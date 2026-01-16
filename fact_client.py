#!/usr/bin/env python3
"""
FACT Dance Client - Runs on robot or Mac.

Captures audio from microphone, streams to FACT server,
receives servo commands and executes them.

Usage:
    # Test with simulation (no robot)
    python fact_client.py --server ws://SERVER_IP:8765 --simulate

    # On robot
    python fact_client.py --server ws://SERVER_IP:8765

Environment variable:
    FACT_SERVER=ws://your-server:8765
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
import sounddevice as sd

# Try to import robot SDK
try:
    sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False


class RobotController:
    """Control robot servos."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self.controller = None
        self.last_servos = {}

        if not simulate and ROBOT_AVAILABLE:
            print("Connecting to robot...")
            board = rrc.Board()
            self.controller = Controller(board)
            print("Robot connected!")
        elif not simulate:
            print("⚠️  Robot SDK not available - running in simulation")
            self.simulate = True

    def set_servos(self, servo_dict, time_ms=50):
        """Set servo positions."""
        self.last_servos = servo_dict

        if self.simulate:
            return

        if self.controller:
            for servo_id, pulse in servo_dict.items():
                self.controller.set_bus_servo_pulse(int(servo_id), int(pulse), int(time_ms))

    def print_status(self):
        """Print current servo state."""
        if self.last_servos:
            servos_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.last_servos.items()))
            print(f"  Servos: {servos_str}")


class AudioStreamer:
    """Stream audio to server."""

    def __init__(self, sample_rate=30720, chunk_duration=0.1):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = asyncio.Queue()

    def start(self):
        """Start audio capture."""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            # Put audio in queue (will be sent to server)
            audio = indata[:, 0].astype(np.float32)
            asyncio.get_event_loop().call_soon_threadsafe(
                self.audio_queue.put_nowait, audio.copy()
            )

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=callback
        )
        self.stream.start()

    def stop(self):
        """Stop audio capture."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    async def get_chunk(self):
        """Get next audio chunk."""
        return await self.audio_queue.get()


async def run_client(server_url, robot, simulate=False):
    """Main client loop."""

    print(f"\n🔗 Connecting to {server_url}...")

    # Stats
    latencies = deque(maxlen=100)
    fps_times = deque(maxlen=100)
    last_time = time.time()

    try:
        async with websockets.connect(server_url, max_size=10*1024*1024) as ws:
            print("✅ Connected to server!")

            # Start audio streaming
            streamer = AudioStreamer()
            streamer.start()

            print("\n🎤 Streaming audio... (Ctrl+C to stop)")
            print("-" * 50)

            async def send_audio():
                """Send audio chunks to server."""
                while True:
                    chunk = await streamer.get_chunk()
                    await ws.send(json.dumps({
                        'type': 'audio',
                        'audio': chunk.tolist(),
                        'timestamp': time.time()
                    }))

            async def receive_servos():
                """Receive and execute servo commands."""
                nonlocal last_time

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    if data['type'] == 'servos':
                        now = time.time()
                        fps_times.append(now)

                        # Calculate FPS
                        if len(fps_times) > 1:
                            fps = len(fps_times) / (fps_times[-1] - fps_times[0])
                        else:
                            fps = 0

                        # Execute servo commands
                        robot.set_servos(data['servos'])

                        # Print status
                        print(f"🦾 FPS: {fps:.1f} | "
                              f"Inference: {data.get('inference_time_ms', 0)}ms | "
                              f"Server FPS: {data.get('fps', 0):.1f}")

                        if simulate:
                            robot.print_status()

            # Run send and receive concurrently
            await asyncio.gather(
                send_audio(),
                receive_servos()
            )

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ Connection closed: {e}")
    except ConnectionRefusedError:
        print(f"\n❌ Could not connect to {server_url}")
        print("   Make sure the server is running!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'streamer' in dir():
            streamer.stop()


async def test_connection(server_url):
    """Test server connection."""
    print(f"Testing connection to {server_url}...")

    try:
        async with websockets.connect(server_url) as ws:
            # Send ping
            await ws.send(json.dumps({'type': 'ping'}))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            if data.get('type') == 'pong':
                print("✅ Server responding!")

            # Get status
            await ws.send(json.dumps({'type': 'status'}))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"   Feature buffer: {data.get('feature_buffer', 0)}/240")
            print(f"   Motion buffer: {data.get('motion_buffer', 0)}/120")
            print(f"   Avg inference: {data.get('avg_inference_ms', 0)}ms")

            return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='FACT Dance Client')
    parser.add_argument('--server', type=str,
                       default=os.environ.get('FACT_SERVER', 'ws://localhost:8765'),
                       help='Server WebSocket URL')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulation mode (no robot)')
    parser.add_argument('--test', action='store_true',
                       help='Test connection only')
    args = parser.parse_args()

    print("=" * 60)
    print("🎵 FACT Dance Client")
    print("=" * 60)

    if args.test:
        asyncio.run(test_connection(args.server))
        return

    # Initialize robot
    robot = RobotController(simulate=args.simulate)

    print(f"\nServer: {args.server}")
    print(f"Mode: {'Simulation' if args.simulate else 'Robot'}")

    # Run client
    try:
        asyncio.run(run_client(args.server, robot, simulate=args.simulate))
    except KeyboardInterrupt:
        print("\n\nStopped by user")


if __name__ == '__main__':
    main()
