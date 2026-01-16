#!/usr/bin/env python3
"""
Play dance commands on Tony Pro robot.

This script reads the JSON file from retarget_to_tonypi.py and sends
servo commands to the robot via the HiwonderSDK.

Usage (on the robot):
    python play_dance.py --input dance_tonypi.json

For testing without robot:
    python play_dance.py --input dance_tonypi.json --simulate
"""

import argparse
import json
import time
import sys

# Import the Tony Pro controller
from tony_pro import TonyProController, get_neutral, ALL_SERVOS


class RobotController:
    """Control Tony Pro servos (wrapper for backwards compatibility)."""

    def __init__(self, simulate=False):
        self._controller = TonyProController(simulate=simulate)
        self.simulate = simulate

    def set_servo(self, servo_id, pulse, time_ms):
        """Set a servo to a position.

        Args:
            servo_id: Servo ID (1-16)
            pulse: Pulse value (typically 0-1000, center=500)
            time_ms: Time to reach position in milliseconds
        """
        self._controller.set_servo(servo_id, pulse, time_ms)

    def set_servos(self, servo_dict, time_ms):
        """Set multiple servos at once.

        Args:
            servo_dict: {servo_id: pulse, ...}
            time_ms: Time to reach positions
        """
        self._controller.set_servos(servo_dict, time_ms)

    def go_to_neutral(self):
        """Move all servos to neutral position (500)."""
        self._controller.go_neutral(time_ms=1000)
        time.sleep(1.2)


def play_sequence(controller, frames, fps=60, loop=False):
    """Play a sequence of frames on the robot.

    Args:
        controller: RobotController instance
        frames: List of frame dicts from JSON
        fps: Playback frame rate
        loop: Whether to loop the sequence
    """
    frame_time = 1.0 / fps
    n_frames = len(frames)

    print(f"Playing {n_frames} frames at {fps} FPS...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            for i, frame in enumerate(frames):
                start = time.time()

                # Send servo commands
                controller.set_servos(frame['servos'], frame['time_ms'])

                # Progress indicator
                if i % 30 == 0:
                    print(f"Frame {i}/{n_frames} ({100*i/n_frames:.0f}%)", end='\r')

                # Wait for frame time
                elapsed = time.time() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            print(f"\nSequence complete!")

            if not loop:
                break
            print("Looping...")

    except KeyboardInterrupt:
        print("\nStopped by user")


def main():
    parser = argparse.ArgumentParser(description='Play dance on TonyPi robot')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode (no robot)')
    parser.add_argument('--loop', action='store_true', help='Loop the sequence')
    parser.add_argument('--fps', type=int, default=None, help='Override FPS')
    parser.add_argument('--neutral-first', action='store_true', help='Go to neutral before playing')
    parser.add_argument('--neutral-after', action='store_true', help='Go to neutral after playing')
    args = parser.parse_args()

    # Load dance data
    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    fps = args.fps if args.fps else data['fps']

    print(f"Loaded {len(frames)} frames, {fps} FPS")
    print(f"Duration: {len(frames)/fps:.1f} seconds")
    print(f"Active servos: {data['active_servos']}")

    # Initialize robot
    controller = RobotController(simulate=args.simulate)

    # Go to neutral first if requested
    if args.neutral_first:
        controller.go_to_neutral()

    # Play the sequence
    play_sequence(controller, frames, fps=fps, loop=args.loop)

    # Return to neutral if requested
    if args.neutral_after:
        controller.go_to_neutral()


if __name__ == '__main__':
    main()
