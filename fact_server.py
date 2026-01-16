#!/usr/bin/env python3
"""
FACT Dance Server - Runs on GPU cloud (H100/A100).

Receives audio stream via WebSocket, runs FACT model,
returns servo commands in real-time.

Usage:
    python fact_server.py --port 8765

Deploy to cloud with GPU for real-time performance.
"""

import asyncio
import json
import os
import sys
import time
import argparse
import numpy as np
from collections import deque

import websockets

# Add mint to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mint'))

# Audio processing
from scipy import signal
from scipy.fftpack import dct

# FACT model
import tensorflow as tf
from mint.utils import config_util
from mint.core import model_builder

# Retargeting
from retarget_to_tonypi import TONYPI_SERVO_MAP, ACTIVE_SERVOS, SMPL_JOINTS, radians_to_pulse


class AudioFeatureExtractor:
    """Extract audio features for FACT model."""

    def __init__(self, sample_rate=30720, hop_length=512, n_fft=2048):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Pre-compute mel filterbank
        self.mel_fb = self._mel_filterbank(sample_rate, n_fft, n_mels=128)
        self.fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.window = signal.windows.hann(n_fft)

    def _mel_filterbank(self, sr, n_fft, n_mels=128, fmin=0, fmax=None):
        if fmax is None:
            fmax = sr / 2

        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        freqs = mel_to_hz(mels)

        fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        filterbank = np.zeros((n_mels, len(fft_freqs)))

        for i in range(n_mels):
            lower, center, upper = freqs[i], freqs[i + 1], freqs[i + 2]
            for j, freq in enumerate(fft_freqs):
                if lower <= freq <= center:
                    filterbank[i, j] = (freq - lower) / (center - lower)
                elif center <= freq <= upper:
                    filterbank[i, j] = (upper - freq) / (upper - center)

        return filterbank

    def extract_frame_features(self, audio_frame):
        """Extract 35-dim features from a single audio frame."""

        # Ensure correct size
        if len(audio_frame) < self.n_fft:
            audio_frame = np.pad(audio_frame, (0, self.n_fft - len(audio_frame)))

        frame = audio_frame[:self.n_fft] * self.window
        spectrum = np.abs(np.fft.rfft(frame))

        # Envelope (energy)
        envelope = np.sqrt(np.mean(frame ** 2))

        # MFCC (20 dims)
        mel_spec = np.dot(spectrum, self.mel_fb.T)
        mel_spec = np.log(mel_spec + 1e-8)
        mfcc = dct(mel_spec, type=2, norm='ortho')[:20]

        # Chroma (12 dims)
        chroma = np.zeros(12)
        for i, freq in enumerate(self.fft_freqs):
            if freq > 0:
                pitch_class = int(round(12 * np.log2(freq / 440) + 69)) % 12
                chroma[pitch_class] += spectrum[i]
        chroma = chroma / (np.max(chroma) + 1e-8)

        # Peak/beat (simplified - server doesn't track history)
        peak_onehot = 1.0 if envelope > 0.1 else 0.0
        beat_onehot = 0.0  # Would need history for proper beat detection

        features = np.concatenate([
            [envelope],
            mfcc,
            chroma,
            [peak_onehot],
            [beat_onehot]
        ])

        return features.astype(np.float32)


class FACTDanceServer:
    """Real-time FACT dance generation server."""

    def __init__(self, config_path, checkpoint_dir):
        print("Loading FACT model...")
        self.model = self._load_model(config_path, checkpoint_dir)
        print("Model loaded!")

        self.feature_extractor = AudioFeatureExtractor()

        # Rolling buffers
        self.audio_buffer = deque(maxlen=30720 * 4)  # 4 seconds of audio
        self.feature_buffer = deque(maxlen=240)  # FACT needs 240 frames of audio context
        self.motion_buffer = deque(maxlen=120)  # 120 frames of motion history (seed)

        # Initialize with neutral pose
        self._init_neutral_motion()

        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.last_inference_time = 0

    def _load_model(self, config_path, checkpoint_dir):
        configs = config_util.get_configs_from_pipeline_file(config_path)
        model_config = configs['model']

        model = model_builder.build(model_config, is_training=False)
        model.global_step = tf.Variable(initial_value=0, dtype=tf.int64)

        checkpoint = tf.train.Checkpoint(model=model, global_step=model.global_step)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=5
        )
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"Loaded checkpoint: {checkpoint_manager.latest_checkpoint}")

        return model

    def _init_neutral_motion(self):
        """Initialize with neutral standing pose."""
        identity_rotmat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        neutral_frame = np.zeros(225, dtype=np.float32)
        neutral_frame[0:3] = [0, 0, 0]  # Translation
        for joint in range(24):
            start_idx = 3 + joint * 9
            neutral_frame[start_idx:start_idx + 9] = identity_rotmat

        for _ in range(120):
            self.motion_buffer.append(neutral_frame.copy())

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
            start_idx = 3 + smpl_idx * 9
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

    async def process_audio(self, audio_chunk):
        """Process incoming audio and generate dance."""

        # Add to audio buffer
        self.audio_buffer.extend(audio_chunk)

        # Extract features for new audio
        hop = self.feature_extractor.hop_length
        n_fft = self.feature_extractor.n_fft

        # Process as many frames as we can
        audio_array = np.array(self.audio_buffer)
        while len(audio_array) >= n_fft:
            frame = audio_array[:n_fft]
            features = self.feature_extractor.extract_frame_features(frame)
            self.feature_buffer.append(features)
            audio_array = audio_array[hop:]

        # Update audio buffer (remove processed samples)
        self.audio_buffer.clear()
        self.audio_buffer.extend(audio_array)

        # Check if we have enough context to generate
        if len(self.feature_buffer) < 240 or len(self.motion_buffer) < 120:
            return None

        # Run FACT inference
        start_time = time.time()

        # Prepare inputs
        motion_input = np.array(list(self.motion_buffer))[np.newaxis, :, :]  # [1, 120, 225]
        audio_input = np.array(list(self.feature_buffer))[np.newaxis, :, :]  # [1, 240, 35]

        motion_tensor = tf.constant(motion_input, dtype=tf.float32)
        audio_tensor = tf.constant(audio_input, dtype=tf.float32)

        inputs = {
            "motion_input": motion_tensor,
            "audio_input": audio_tensor
        }

        # Generate next frames (generate a small batch for efficiency)
        outputs = self.model.infer_auto_regressive(inputs, steps=10)
        generated = outputs[0].numpy()  # [10, 225]

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time

        # Update motion buffer with generated frames
        for frame in generated:
            self.motion_buffer.append(frame)

        # Convert latest frame to servo commands
        servo_commands = self._motion_to_servos(generated[-1])

        return {
            'servos': servo_commands,
            'inference_time_ms': int(inference_time * 1000),
            'avg_inference_ms': int(np.mean(list(self.inference_times)) * 1000),
            'fps': 10 / inference_time if inference_time > 0 else 0
        }


async def handle_client(websocket, server):
    """Handle a connected client."""

    client_addr = websocket.remote_address
    print(f"Client connected: {client_addr}")

    try:
        async for message in websocket:
            # Parse incoming message
            data = json.loads(message)

            if data['type'] == 'audio':
                # Decode audio chunk (sent as list of floats)
                audio_chunk = np.array(data['audio'], dtype=np.float32)

                # Process and get servo commands
                result = await server.process_audio(audio_chunk)

                if result:
                    await websocket.send(json.dumps({
                        'type': 'servos',
                        **result
                    }))

            elif data['type'] == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))

            elif data['type'] == 'status':
                await websocket.send(json.dumps({
                    'type': 'status',
                    'feature_buffer': len(server.feature_buffer),
                    'motion_buffer': len(server.motion_buffer),
                    'avg_inference_ms': int(np.mean(list(server.inference_times)) * 1000) if server.inference_times else 0
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {client_addr}")
    except Exception as e:
        print(f"Error handling client: {e}")


async def main(args):
    print("=" * 60)
    print("🤖 FACT Dance Server")
    print("=" * 60)
    print(f"\nStarting server on port {args.port}...")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus}")
    else:
        print("⚠️  No GPU detected - will be slow!")

    # Initialize server
    server = FACTDanceServer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint
    )

    # Start WebSocket server
    async with websockets.serve(
        lambda ws: handle_client(ws, server),
        "0.0.0.0",
        args.port,
        max_size=10 * 1024 * 1024  # 10MB max message
    ):
        print(f"\n✅ Server running on ws://0.0.0.0:{args.port}")
        print("Waiting for connections...")
        await asyncio.Future()  # Run forever


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FACT Dance Server')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--config', type=str,
                       default='mint/configs/fact_v5_deeper_t10_cm12.config')
    parser.add_argument('--checkpoint', type=str, default='mint/checkpoints')
    args = parser.parse_args()

    asyncio.run(main(args))
