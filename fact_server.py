#!/usr/bin/env python3
"""
FACT Dance Server - Runs on GPU cloud (H100/A100).

Receives audio file via WebSocket, runs FACT model, returns servo commands.

Usage:
    python fact_server.py --port 8765

Deploy to cloud with GPU for real-time performance.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import argparse
import numpy as np
from collections import deque

import websockets

# Add mint to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mint'))

import tensorflow as tf

# Import core functions from generate_dance.py (single source of truth)
from generate_dance import (
    extract_audio_features,
    load_model,
    create_neutral_motion,
    generate_dance,
)

# Retargeting
from retarget_to_tonypi import TONYPI_SERVO_MAP, ACTIVE_SERVOS, SMPL_JOINTS, radians_to_pulse


class FACTDanceServer:
    """FACT dance generation server."""

    def __init__(self, config_path, checkpoint_dir):
        print("Loading FACT model...")
        self.model = load_model(config_path, checkpoint_dir)
        print("Model loaded!")

        self.fps = 60
        self.cancel_flag = False  # Set to True to cancel current job
        self.is_processing = False  # Track if a job is in progress

    def _motion_to_servos(self, motion_frame):
        """Convert a single motion frame to servo commands."""
        from scipy.spatial.transform import Rotation

        servo_commands = {}

        for servo_name in ACTIVE_SERVOS:
            config = TONYPI_SERVO_MAP[servo_name]
            smpl_joint = config['smpl_joint']
            smpl_idx = SMPL_JOINTS.index(smpl_joint)
            axis = config['axis']

            # Extract rotation matrix for this joint
            # Motion format: [0:6] padding, [6:9] translation, [9:225] 24 joints x 9 rotmat
            start_idx = 9 + smpl_idx * 9
            rotmat = motion_frame[start_idx:start_idx + 9].reshape(3, 3)

            # Convert to euler angles
            try:
                r = Rotation.from_matrix(rotmat)
                euler = r.as_euler('xyz', degrees=False)
                angle_rad = euler[axis]
            except:
                angle_rad = 0.0

            # Convert to pulse
            pulse = radians_to_pulse(angle_rad, config)
            servo_commands[config['servo_id']] = pulse

        return servo_commands

    def cancel_current_job(self):
        """Cancel any in-progress job."""
        if self.is_processing:
            print("Cancelling current job...")
            self.cancel_flag = True

    async def process_audio_file(self, audio_path, websocket=None):
        """Process audio file and generate complete dance.

        Args:
            audio_path: Path to audio file on server
            websocket: Optional websocket for progress updates

        Returns:
            dict with frames, fps, and metadata
        """
        # Reset cancel flag and mark as processing
        self.cancel_flag = False
        self.is_processing = True

        print(f"Processing audio file: {audio_path}")
        start_time = time.time()

        # Step 1: Extract audio features using generate_dance.py
        if websocket:
            await websocket.send(json.dumps({
                'type': 'progress',
                'stage': 'features',
                'message': 'Extracting audio features...'
            }))

        audio_features = extract_audio_features(audio_path, target_fps=60)
        n_feature_frames = len(audio_features)
        print(f"Extracted {n_feature_frames} feature frames")

        if websocket:
            await websocket.send(json.dumps({
                'type': 'progress',
                'stage': 'features_done',
                'message': f'Extracted {n_feature_frames} feature frames',
                'feature_frames': n_feature_frames
            }))

        # Step 2: Calculate steps to generate
        audio_seq_length = 240
        steps_to_generate = n_feature_frames - audio_seq_length

        if steps_to_generate <= 0:
            steps_to_generate = 60
            pad_amount = audio_seq_length - n_feature_frames + steps_to_generate
            audio_features = np.pad(audio_features, ((0, pad_amount), (0, 0)), mode='edge')

        print(f"Generating {steps_to_generate} motion frames...")

        if websocket:
            await websocket.send(json.dumps({
                'type': 'progress',
                'stage': 'generating',
                'message': f'Generating {steps_to_generate} motion frames...',
                'total_frames': steps_to_generate
            }))

        # Step 3: Generate motion
        seed_motion = create_neutral_motion(n_frames=120)
        gen_start = time.time()

        # Generate in chunks for progress updates
        all_motion = []
        chunk_size = 120
        remaining = steps_to_generate
        frames_done = 0

        while remaining > 0:
            # Check for cancellation
            if self.cancel_flag:
                print("Job cancelled!")
                self.is_processing = False
                if websocket:
                    await websocket.send(json.dumps({
                        'type': 'cancelled',
                        'message': 'Job cancelled - new file uploaded'
                    }))
                return None

            current_chunk = min(chunk_size, remaining)

            if all_motion:
                recent = np.array(all_motion[-120:]) if len(all_motion) >= 120 else np.array(all_motion)
                if len(recent) < 120:
                    pad = create_neutral_motion(120 - len(recent))
                    seed_motion = np.vstack([pad, recent])
                else:
                    seed_motion = recent

            motion = generate_dance(
                self.model,
                audio_features,
                seed_motion,
                steps=current_chunk
            )

            all_motion.extend(motion)
            frames_done += current_chunk
            remaining -= current_chunk

            elapsed = time.time() - gen_start
            fps = frames_done / elapsed if elapsed > 0 else 0
            eta = remaining / fps if fps > 0 else 0

            print(f"Generated {frames_done}/{steps_to_generate} frames ({fps:.1f} fps, ETA {eta:.0f}s)")

            if websocket:
                await websocket.send(json.dumps({
                    'type': 'progress',
                    'stage': 'generating',
                    'frames_done': frames_done,
                    'total_frames': steps_to_generate,
                    'fps': round(fps, 1),
                    'eta_seconds': round(eta)
                }))

            # Yield to allow other tasks (like receiving cancel signal)
            await asyncio.sleep(0)

        gen_time = time.time() - gen_start
        print(f"Generated {len(all_motion)} frames in {gen_time:.1f}s")

        # Step 4: Convert motion to servo commands
        if websocket:
            await websocket.send(json.dumps({
                'type': 'progress',
                'stage': 'converting',
                'message': 'Converting to servo commands...'
            }))

        servo_frames = []
        for motion_frame in all_motion:
            servo_commands = self._motion_to_servos(motion_frame)
            servo_frames.append({
                'servos': servo_commands,
                'time_ms': int(1000 / self.fps)
            })

        total_time = time.time() - start_time
        print(f"Batch complete: {len(servo_frames)} frames in {total_time:.1f}s")

        self.is_processing = False

        return {
            'type': 'batch_result',
            'frames': servo_frames,
            'fps': self.fps,
            'n_frames': len(servo_frames),
            'duration_seconds': len(servo_frames) / self.fps,
            'generation_time_seconds': round(total_time, 1),
        }


async def handle_client(websocket, server):
    """Handle a connected client."""

    client_addr = websocket.remote_address
    print(f"Client connected: {client_addr}")

    try:
        async for message in websocket:
            # Check if binary (file upload) or JSON
            if isinstance(message, bytes):
                # Binary file upload - save to temp and process
                print(f"Received binary file: {len(message)} bytes")

                # Cancel any in-progress job
                server.cancel_current_job()

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(message)
                    temp_path = f.name

                try:
                    result = await server.process_audio_file(temp_path, websocket)
                    if result:  # None if cancelled
                        await websocket.send(json.dumps(result))
                        print(f"Sent batch result: {result['n_frames']} frames")
                finally:
                    os.unlink(temp_path)

            else:
                # JSON message
                data = json.loads(message)

                if data['type'] == 'batch_file':
                    # Base64 encoded file
                    print("Received base64 encoded file")

                    # Cancel any in-progress job
                    server.cancel_current_job()

                    file_data = base64.b64decode(data['file'])
                    filename = data.get('filename', 'audio.wav')
                    ext = os.path.splitext(filename)[1] or '.wav'

                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                        f.write(file_data)
                        temp_path = f.name

                    try:
                        result = await server.process_audio_file(temp_path, websocket)
                        if result:  # None if cancelled
                            await websocket.send(json.dumps(result))
                            print(f"Sent batch result: {result['n_frames']} frames")
                    finally:
                        os.unlink(temp_path)

                elif data['type'] == 'batch_path':
                    # File path on server (for CLI usage)
                    audio_path = data['path']
                    if not os.path.exists(audio_path):
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'File not found: {audio_path}'
                        }))
                        continue

                    result = await server.process_audio_file(audio_path, websocket)
                    await websocket.send(json.dumps(result))
                    print(f"Sent batch result: {result['n_frames']} frames")

                elif data['type'] == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))

                elif data['type'] == 'status':
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'model_loaded': server.model is not None,
                        'fps': server.fps
                    }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {client_addr}")
    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass


async def main(args):
    print("=" * 60)
    print("FACT Dance Server")
    print("=" * 60)
    print(f"\nStarting server on port {args.port}...")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
    else:
        print("No GPU detected - will be slow!")

    server = FACTDanceServer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint
    )

    async with websockets.serve(
        lambda ws: handle_client(ws, server),
        "0.0.0.0",
        args.port,
        max_size=100 * 1024 * 1024  # 100MB max for file uploads
    ):
        print(f"\nServer running on ws://0.0.0.0:{args.port}")
        print("Accepts: binary file, {'type':'batch_file','file':base64,'filename':...}, or {'type':'batch_path','path':...}")
        print("Waiting for connections...")
        await asyncio.Future()


def get_default_path(relative_path):
    """Try multiple locations for config/checkpoint paths."""
    candidates = [
        os.path.join(os.path.expanduser('~'), relative_path),
        os.path.join(os.path.dirname(__file__), relative_path),
        relative_path,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FACT Dance Server')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = get_default_path('mint/configs/fact_v5_deeper_t10_cm12.config')
    if args.checkpoint is None:
        args.checkpoint = get_default_path('mint/checkpoints')

    asyncio.run(main(args))
