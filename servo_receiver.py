#!/usr/bin/env python3
"""
Servo Receiver - Runs on the TonyPi robot.
Receives servo commands over WebSocket and executes them.

Usage on Pi:
    python3 servo_receiver.py --port 8766

Then from laptop:
    python3 fact_client.py --server ws://LAMBDA:8765 --robot ws://192.168.149.1:8766
"""

import asyncio
import json
import argparse
import sys

# Import robot SDK
sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("Warning: Robot SDK not available, running in test mode")

import websockets

class ServoController:
    def __init__(self):
        self.board = None
        if ROBOT_AVAILABLE:
            print("Connecting to robot servos...")
            self.board = rrc.Board()
            print("Robot connected!")
        else:
            print("Running in test mode (no robot)")

    def set_servos(self, servo_dict, time_ms=50):
        """Set servo positions.

        Handles both bus servos (1-16) and PWM servos (pwm1, pwm2 for head).
        """
        # Always print for debugging
        servos_str = ", ".join(f"{k}:{v}" for k, v in sorted(servo_dict.items(), key=lambda x: str(x[0])))
        print(f"Servos: {servos_str}", flush=True)

        if not self.board:
            return

        bus_cmds = []
        pwm_cmds = []

        for servo_id, pulse in servo_dict.items():
            # Check if it's a PWM servo (head)
            if isinstance(servo_id, str) and servo_id.startswith('pwm'):
                pwm_id = int(servo_id[3:])
                pwm_cmds.append((pwm_id, int(pulse)))
            else:
                # Bus servo (body)
                bus_cmds.append((int(servo_id), int(pulse)))

        # Send commands
        time_sec = time_ms / 1000.0
        if bus_cmds:
            self.board.bus_servo_set_position(time_sec, tuple(bus_cmds))
        if pwm_cmds:
            self.board.pwm_servo_set_position(time_sec, tuple(pwm_cmds))


async def handle_client(websocket, controller):
    """Handle incoming servo commands."""
    addr = websocket.remote_address
    print(f"Client connected: {addr}")

    try:
        async for message in websocket:
            data = json.loads(message)

            if data.get('type') == 'servos':
                controller.set_servos(data['servos'])

            elif data.get('type') == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {addr}")
    except Exception as e:
        print(f"Error: {e}")


async def main(port):
    controller = ServoController()

    print(f"\nServo Receiver starting on port {port}...")
    async with websockets.serve(
        lambda ws: handle_client(ws, controller),
        "0.0.0.0",
        port
    ):
        print(f"Ready! Waiting for connections on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()

    asyncio.run(main(args.port))
