#!/usr/bin/env python3
"""
Retarget SMPL joint angles to Tony Pro servo commands.

SMPL has 24 joints, Tony Pro has 16 servos.
This script maps the relevant joints and scales angles to servo pulse values.

Usage: python retarget_to_tonypi.py --input dance_test_angles.npy --output tonypi_commands.json
"""

import argparse
import json
import numpy as np

# Import servo configuration from tony_pro module
from tony_pro import (
    SMPL_JOINTS,
    RETARGET_MAP,
    ACTIVE_RETARGET_JOINTS,
    PULSE_CENTER,
)

# For backwards compatibility, create the old-style mapping
TONYPI_SERVO_MAP = {}
for joint_name, config in RETARGET_MAP.items():
    TONYPI_SERVO_MAP[joint_name] = {
        'servo_id': config['servo_id'],
        'smpl_joint': config['smpl_joint'],
        'axis': config['axis'],
        'scale': config['scale'],
        'offset': PULSE_CENTER,
        'min': config['min'],
        'max': config['max'],
    }

ACTIVE_SERVOS = ACTIVE_RETARGET_JOINTS


def radians_to_pulse(angle_rad, config):
    """Convert angle in radians to servo pulse value.

    TonyPi servos typically use pulse values 0-1000 (center = 500).
    Full range is approximately -90 to +90 degrees (-1.57 to +1.57 rad).

    The conversion is calculated dynamically based on each servo's actual
    min/max range, since different servos have different safe ranges.
    """
    import math

    # Calculate pulse_per_rad based on this servo's actual range
    # A servo typically covers π radians (180°) for its full mechanical range
    servo_range = config['max'] - config['min']  # e.g., 800-200 = 600
    pulse_per_rad = servo_range / math.pi  # Scale to this servo's range

    # Apply scale and convert to pulse units
    scaled_angle = angle_rad * config['scale']
    pulse = config['offset'] + int(scaled_angle * pulse_per_rad)

    # Clamp to safe range
    pulse = max(config['min'], min(config['max'], pulse))

    return pulse


def smooth_trajectory(angles, window_size=5):
    """Apply smoothing to reduce jerky motion."""
    if window_size <= 1:
        return angles

    kernel = np.ones(window_size) / window_size
    smoothed = np.zeros_like(angles)

    for joint in range(angles.shape[1]):
        for axis in range(angles.shape[2]):
            smoothed[:, joint, axis] = np.convolve(
                angles[:, joint, axis], kernel, mode='same'
            )

    return smoothed


def retarget(smpl_angles, fps=60, smooth=True):
    """Convert SMPL joint angles to TonyPi servo commands.

    Args:
        smpl_angles: [n_frames, 24, 3] array of euler angles
        fps: frames per second
        smooth: whether to smooth the trajectory

    Returns:
        List of frame dictionaries with servo commands
    """
    if smooth:
        smpl_angles = smooth_trajectory(smpl_angles, window_size=5)

    n_frames = smpl_angles.shape[0]
    frames = []

    for frame_idx in range(n_frames):
        frame_data = {
            'time_ms': int(1000 / fps),  # Time to reach this pose
            'servos': {}
        }

        for servo_name in ACTIVE_SERVOS:
            config = TONYPI_SERVO_MAP[servo_name]
            smpl_joint = config['smpl_joint']
            smpl_idx = SMPL_JOINTS.index(smpl_joint)
            axis = config['axis']

            angle_rad = smpl_angles[frame_idx, smpl_idx, axis]
            pulse = radians_to_pulse(angle_rad, config)

            frame_data['servos'][config['servo_id']] = pulse

        frames.append(frame_data)

    return frames


def main():
    parser = argparse.ArgumentParser(description='Retarget SMPL angles to TonyPi')
    parser.add_argument('--input', type=str, required=True, help='Input angles .npy file')
    parser.add_argument('--output', type=str, default='tonypi_commands.json', help='Output JSON file')
    parser.add_argument('--fps', type=int, default=60, help='Frame rate')
    parser.add_argument('--no-smooth', action='store_true', help='Disable smoothing')
    parser.add_argument('--preview', action='store_true', help='Print first few frames')
    args = parser.parse_args()

    print(f"Loading angles from {args.input}...")
    smpl_angles = np.load(args.input)
    print(f"Shape: {smpl_angles.shape} (frames, joints, xyz)")

    print("Retargeting to TonyPi servos...")
    frames = retarget(smpl_angles, fps=args.fps, smooth=not args.no_smooth)

    # Save to JSON
    output_data = {
        'fps': args.fps,
        'n_frames': len(frames),
        'active_servos': ACTIVE_SERVOS,
        'servo_map': {name: TONYPI_SERVO_MAP[name]['servo_id'] for name in ACTIVE_SERVOS},
        'frames': frames
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(frames)} frames to {args.output}")

    if args.preview:
        print("\nFirst 5 frames:")
        for i, frame in enumerate(frames[:5]):
            print(f"  Frame {i}: {frame['servos']}")

    # Print servo ranges used
    print("\nServo pulse ranges in this sequence:")
    for servo_name in ACTIVE_SERVOS:
        servo_id = TONYPI_SERVO_MAP[servo_name]['servo_id']
        pulses = [f['servos'][servo_id] for f in frames]
        print(f"  {servo_name} (ID {servo_id}): {min(pulses)} - {max(pulses)}")


if __name__ == '__main__':
    main()
