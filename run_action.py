#!/usr/bin/env python3
"""
Interactive action runner for TonyPi robot.

Lists available pre-scripted actions and lets you run them by number.

Usage (on robot):
    python run_action.py

With simulation (no robot):
    python run_action.py --simulate
"""

import argparse
import time
import sys

from read_d6a import read_d6a
from tony_pro import TonyProController

# Curated list of interesting actions (name, description)
ACTIONS = [
    ("wave", "Wave hello"),
    ("bow", "Take a bow"),
    ("twist", "Dance twist"),
    ("stepping", "Step dance"),
    ("left_kick", "Kick left"),
    ("right_kick", "Kick right"),
    ("wing_chun", "Wing chun combo"),
    ("left_shot", "Punch left"),
    ("right_shot", "Punch right"),
    ("sit_ups", "Do sit-ups"),
    ("stand_up_front", "Get up from front"),
    ("stand_up_back", "Get up from back"),
    ("go_forward", "Walk forward"),
    ("back", "Walk backward"),
    ("turn_left", "Turn left"),
    ("turn_right", "Turn right"),
    ("squat_down", "Squat down"),
    ("squat_up", "Stand from squat"),
    ("grab_left", "Grab with left"),
    ("grab_right", "Grab with right"),
]


def print_menu():
    """Print the action menu."""
    print("\n" + "=" * 50)
    print("  TonyPi Action Runner")
    print("=" * 50)
    for i, (name, desc) in enumerate(ACTIONS):
        print(f"  {i+1:2d}. {name:20s} - {desc}")
    print("=" * 50)
    print("  0. Exit")
    print("  n. Go to neutral position")
    print("  s. Stand")
    print("=" * 50)


def play_action(controller, action_name, action_dir="./action_groups"):
    """Load and play a .d6a action file.

    Args:
        controller: TonyProController instance
        action_name: Name of the action (without .d6a extension)
        action_dir: Directory containing .d6a files
    """
    filepath = f"{action_dir}/{action_name}.d6a"

    try:
        frames = read_d6a(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    total_ms = sum(f['time_ms'] for f in frames)
    print(f"Playing '{action_name}' ({len(frames)} frames, {total_ms/1000:.1f}s)...")

    for i, frame in enumerate(frames):
        time_ms = frame['time_ms']
        servos = frame['servos']

        # Send all servo commands for this frame
        controller.set_servos(servos, time_ms)

        # Wait for the frame duration
        time.sleep(time_ms / 1000.0)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Interactive TonyPi action runner')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode (no robot)')
    parser.add_argument('--action-dir', type=str, default='/home/pi/TonyPi/ActionGroups',
                        help='Directory containing .d6a files')
    args = parser.parse_args()

    # Initialize controller
    controller = TonyProController(simulate=args.simulate)

    if args.simulate:
        print("SIMULATION MODE - no robot commands sent")

    print_menu()

    while True:
        try:
            choice = input("\nEnter number (or 'q' to quit): ").strip().lower()

            if choice in ('q', '0', 'quit', 'exit'):
                print("Goodbye!")
                break

            if choice == 'n':
                print("Going to neutral position...")
                controller.go_neutral(time_ms=1000)
                time.sleep(1.2)
                print("Done!")
                continue

            if choice == 's':
                # Run the stand action
                play_action(controller, "stand", args.action_dir)
                continue

            if choice == 'm':
                print_menu()
                continue

            # Try to parse as number
            try:
                num = int(choice)
                if 1 <= num <= len(ACTIONS):
                    action_name, _ = ACTIONS[num - 1]
                    play_action(controller, action_name, args.action_dir)
                else:
                    print(f"Invalid number. Enter 1-{len(ACTIONS)}, or 'q' to quit")
            except ValueError:
                print("Enter a number, 'n' for neutral, 's' for stand, 'm' for menu, or 'q' to quit")

        except KeyboardInterrupt:
            print("\n\nInterrupted! Going to neutral...")
            controller.go_neutral(time_ms=1000)
            time.sleep(1.2)
            print("Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
