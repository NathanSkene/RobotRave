#!/usr/bin/env python3
"""
Servo Discovery Script for TonyPi Robot.

Run this on Saturday to figure out which servo ID controls which joint.
The script will move each servo one at a time and ask you what moved.

Usage (on the robot):
    python discover_servos.py

The results are saved to servo_mapping.json for use in retargeting.
"""

import json
import time
import sys

# Import HiwonderSDK
try:
    sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("ERROR: HiwonderSDK not found. Run this script on the robot!")
    sys.exit(1)


def discover_servos():
    """Interactively discover servo mappings."""

    print("=" * 60)
    print("TonyPi Servo Discovery Tool")
    print("=" * 60)
    print()
    print("This script will move each servo one at a time.")
    print("Watch what moves and enter the joint name.")
    print()
    print("Joint names to use:")
    print("  Arms: r_shoulder_pitch, r_shoulder_roll, r_elbow")
    print("        l_shoulder_pitch, l_shoulder_roll, l_elbow")
    print("  Legs: r_hip, r_knee, r_ankle")
    print("        l_hip, l_knee, l_ankle")
    print("  Head: head_pan, head_tilt")
    print("  Other: waist, r_hand, l_hand")
    print()
    print("Enter 'skip' to skip a servo, 'quit' to exit early.")
    print()

    # Initialize robot
    print("Connecting to robot...")
    board = rrc.Board()
    controller = Controller(board)
    print("Connected!")
    print()

    # First, move all to neutral
    print("Moving all servos to neutral (500)...")
    for servo_id in range(1, 19):
        controller.set_bus_servo_pulse(servo_id, 500, 500)
    time.sleep(1)
    print("Ready!")
    print()

    mapping = {}

    for servo_id in range(1, 19):
        print(f"\n--- Testing Servo {servo_id} ---")
        input("Press Enter to move this servo...")

        # Move to one direction
        print(f"Moving servo {servo_id} to 650...")
        controller.set_bus_servo_pulse(servo_id, 650, 500)
        time.sleep(0.8)

        # Return to center
        print(f"Returning to center...")
        controller.set_bus_servo_pulse(servo_id, 500, 500)
        time.sleep(0.5)

        # Ask what moved
        joint = input(f"What joint moved? (or 'skip'/'quit'): ").strip().lower()

        if joint == 'quit':
            print("Exiting early...")
            break
        elif joint == 'skip' or joint == '':
            print(f"Skipped servo {servo_id}")
            continue
        else:
            mapping[joint] = servo_id
            print(f"Recorded: {joint} = servo {servo_id}")

    # Return all to neutral
    print("\nReturning all servos to neutral...")
    for servo_id in range(1, 19):
        controller.set_bus_servo_pulse(servo_id, 500, 500)
    time.sleep(1)

    # Save mapping
    output_file = 'servo_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n{'='*60}")
    print("Discovery Complete!")
    print(f"{'='*60}")
    print(f"\nMapping saved to {output_file}:")
    for joint, servo_id in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"  {joint}: servo {servo_id}")

    print(f"\nTotal servos mapped: {len(mapping)}/18")
    print("\nNext step: Update TONYPI_SERVO_MAP in retarget_to_tonypi.py")

    return mapping


def test_specific_servo():
    """Test a specific servo interactively."""

    print("Connecting to robot...")
    board = rrc.Board()
    controller = Controller(board)
    print("Connected!")

    while True:
        try:
            servo_input = input("\nEnter servo ID (1-18) or 'quit': ").strip()
            if servo_input == 'quit':
                break

            servo_id = int(servo_input)
            if not 1 <= servo_id <= 18:
                print("Invalid servo ID")
                continue

            pulse_input = input(f"Enter pulse value for servo {servo_id} (0-1000, default 500): ").strip()
            pulse = int(pulse_input) if pulse_input else 500

            print(f"Moving servo {servo_id} to {pulse}...")
            controller.set_bus_servo_pulse(servo_id, pulse, 500)

        except ValueError:
            print("Invalid input")
        except KeyboardInterrupt:
            break

    print("\nReturning all servos to neutral...")
    for servo_id in range(1, 19):
        controller.set_bus_servo_pulse(servo_id, 500, 500)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TonyPi Servo Discovery')
    parser.add_argument('--test', action='store_true', help='Test specific servo')
    args = parser.parse_args()

    if args.test:
        test_specific_servo()
    else:
        discover_servos()


if __name__ == '__main__':
    main()
