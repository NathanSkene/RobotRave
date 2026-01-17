#!/usr/bin/env python3
"""
Play dance commands on Tony Pro robot.

Supports:
- Servo JSON from `retarget_to_tonypi.py` ("frames" format)
- SMPL joint angles `.npy` (shape `[n_frames, 24, 3]`) from `generate_dance.py` (`*_angles.npy`)

Usage (on the robot):
    python play_dance.py --input dance_tonypi.json
    python play_dance.py --input dance_output_angles.npy

For testing without robot:
    python play_dance.py --input dance_tonypi.json --simulate
    python play_dance.py --input dance_output_angles.npy --simulate
"""

import argparse
import json
import time
import sys

# Import the Tony Pro controller
from tony_pro import TonyProController, get_neutral

# Optional imports for playing back .npy outputs from generate_dance.py
try:
    import numpy as np
except ImportError:
    np = None

try:
    from retarget_to_tonypi import retarget as retarget_smpl_angles, ACTIVE_SERVOS as RETARGET_ACTIVE_SERVOS
except ImportError:
    retarget_smpl_angles = None
    RETARGET_ACTIVE_SERVOS = None


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


def _load_frames_from_json(input_path, fps_override=None):
    with open(input_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    fps = fps_override if fps_override else data['fps']
    active_servos = data.get('active_servos')
    return frames, fps, active_servos


def _load_frames_from_npy(input_path, fps_override=None, no_smooth=False):
    if np is None:
        raise RuntimeError("numpy is required to load .npy files")
    if retarget_smpl_angles is None:
        raise RuntimeError("retarget_to_tonypi.py is required to play .npy files")

    arr = np.load(input_path)

    # generate_dance.py saves either raw motion [N,225] or angles [N,24,3]
    if arr.ndim == 3 and arr.shape[1:] == (24, 3):
        smpl_angles = arr
    elif arr.ndim == 2 and arr.shape[1] == 225:
        raise ValueError(
            "Got raw FACT motion [N,225]. Use the '*_angles.npy' output (shape [N,24,3]) instead."
        )
    else:
        raise ValueError(f"Unsupported .npy shape: {arr.shape}. Expected [N,24,3] angles.")

    fps = fps_override if fps_override else 60
    frames = retarget_smpl_angles(smpl_angles, fps=fps, smooth=not no_smooth)
    active_servos = RETARGET_ACTIVE_SERVOS
    return frames, fps, active_servos


def main():
    parser = argparse.ArgumentParser(description='Play dance on TonyPi robot')
    parser.add_argument('--input', type=str, required=True, help='Input dance file (.json or .npy)')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode (no robot)')
    parser.add_argument('--loop', action='store_true', help='Loop the sequence')
    parser.add_argument('--fps', type=int, default=None, help='Override FPS (default: from file or 60)')
    parser.add_argument('--neutral-first', action='store_true', help='Go to neutral before playing')
    parser.add_argument('--neutral-after', action='store_true', help='Go to neutral after playing')
    parser.add_argument('--no-smooth', action='store_true', help='Disable smoothing (only applies to .npy inputs)')
    args = parser.parse_args()

    # Load dance data
    print(f"Loading {args.input}...")
    if args.input.endswith('.json'):
        frames, fps, active_servos = _load_frames_from_json(args.input, fps_override=args.fps)
    elif args.input.endswith('.npy'):
        frames, fps, active_servos = _load_frames_from_npy(
            args.input,
            fps_override=args.fps,
            no_smooth=args.no_smooth,
        )
    else:
        raise ValueError("--input must end with .json or .npy")

    print(f"Loaded {len(frames)} frames, {fps} FPS")
    print(f"Duration: {len(frames)/fps:.1f} seconds")
    if active_servos is not None:
        print(f"Active servos: {active_servos}")

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
