#!/usr/bin/env python3
"""
Test FACT model with live microphone recording.

Records audio from mic, then runs it through the FACT model
to generate dance motion. Shows what kind of movements FACT creates.

Usage:
    python test_fact_live.py --duration 10
"""

import argparse
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# Add mint to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mint'))

from generate_dance import (
    extract_audio_features,
    create_neutral_motion,
    load_model,
    generate_dance,
    motion_to_joint_angles
)

# SMPL joint names for readable output
JOINT_NAMES = [
    'root', 'l_hip', 'r_hip', 'belly', 'l_knee', 'r_knee',
    'spine', 'l_ankle', 'r_ankle', 'chest', 'l_toes', 'r_toes',
    'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand'
]

# Joints we care about for dancing
DANCE_JOINTS = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'head', 'l_hip', 'r_hip']


def record_audio(duration, sample_rate=30720):
    """Record audio from microphone."""
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   Play some music!")
    print()

    # Countdown
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("   🔴 RECORDING!")

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    print("   ✅ Recording complete!")
    return audio.flatten(), sample_rate


def analyze_motion(joint_angles):
    """Analyze the generated motion to describe the dance."""
    n_frames = joint_angles.shape[0]

    print("\n" + "=" * 60)
    print("📊 MOTION ANALYSIS - What FACT generated")
    print("=" * 60)

    # Analyze movement intensity per joint
    print("\n🦾 Movement intensity by joint:")
    print("-" * 40)

    intensities = {}
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        if joint_name not in DANCE_JOINTS:
            continue

        angles = joint_angles[:, joint_idx, :]

        # Calculate total movement (sum of absolute changes)
        movement = np.sum(np.abs(np.diff(angles, axis=0)))

        # Calculate range of motion
        range_x = angles[:, 0].max() - angles[:, 0].min()
        range_y = angles[:, 1].max() - angles[:, 1].min()
        range_z = angles[:, 2].max() - angles[:, 2].min()
        total_range = range_x + range_y + range_z

        intensities[joint_name] = {
            'movement': movement,
            'range': total_range,
            'range_xyz': (range_x, range_y, range_z)
        }

    # Sort by movement intensity
    sorted_joints = sorted(intensities.items(), key=lambda x: x[1]['movement'], reverse=True)

    max_movement = max(j[1]['movement'] for j in sorted_joints) if sorted_joints else 1

    for joint_name, data in sorted_joints:
        bar_len = int(20 * data['movement'] / max_movement)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {joint_name:12s} {bar} {data['movement']:.1f}")

    # Determine dance character
    print("\n🎭 Dance character:")
    print("-" * 40)

    arm_movement = (intensities.get('l_shoulder', {}).get('movement', 0) +
                   intensities.get('r_shoulder', {}).get('movement', 0) +
                   intensities.get('l_elbow', {}).get('movement', 0) +
                   intensities.get('r_elbow', {}).get('movement', 0))

    leg_movement = (intensities.get('l_hip', {}).get('movement', 0) +
                   intensities.get('r_hip', {}).get('movement', 0))

    head_movement = intensities.get('head', {}).get('movement', 0)

    total = arm_movement + leg_movement + head_movement + 0.001

    print(f"  Arms:  {100*arm_movement/total:5.1f}% {'🙌' if arm_movement > leg_movement else ''}")
    print(f"  Legs:  {100*leg_movement/total:5.1f}% {'🦵' if leg_movement > arm_movement else ''}")
    print(f"  Head:  {100*head_movement/total:5.1f}% {'😎' if head_movement > total*0.2 else ''}")

    # Describe the style
    print("\n💃 Style assessment:")
    print("-" * 40)

    if arm_movement > leg_movement * 2:
        print("  → Upper body focused (arm-heavy dance)")
    elif leg_movement > arm_movement:
        print("  → Lower body focused (footwork dance)")
    else:
        print("  → Balanced full-body movement")

    if max_movement > 50:
        print("  → High energy / intense movement")
    elif max_movement > 20:
        print("  → Moderate energy")
    else:
        print("  → Subtle / minimal movement")

    # Show sample frames
    print("\n📈 Sample joint angles (first 5 frames):")
    print("-" * 40)
    print("  Frame | R_Shoulder (x,y,z) | L_Shoulder (x,y,z)")
    r_idx = JOINT_NAMES.index('r_shoulder')
    l_idx = JOINT_NAMES.index('l_shoulder')
    for i in range(min(5, n_frames)):
        r = joint_angles[i, r_idx]
        l = joint_angles[i, l_idx]
        print(f"  {i:5d} | ({r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f}) | ({l[0]:+.2f},{l[1]:+.2f},{l[2]:+.2f})")


def main():
    parser = argparse.ArgumentParser(description='Test FACT with live mic input')
    parser.add_argument('--duration', type=int, default=10, help='Recording duration (seconds)')
    parser.add_argument('--generate', type=int, default=5, help='Dance duration to generate (seconds)')
    parser.add_argument('--skip-record', type=str, help='Skip recording, use this audio file instead')
    args = parser.parse_args()

    print("=" * 60)
    print("🎵 FACT Model Test - Live Microphone")
    print("=" * 60)
    print("\nThis will:")
    print(f"  1. Record {args.duration}s of audio from your mic")
    print(f"  2. Extract audio features (what FACT 'sees')")
    print(f"  3. Generate {args.generate}s of dance motion")
    print("  4. Analyze what kind of dance FACT created")

    # Record or load audio
    temp_audio = '/tmp/fact_test_recording.wav'

    if args.skip_record:
        temp_audio = args.skip_record
        print(f"\nUsing existing audio: {temp_audio}")
    else:
        audio, sr = record_audio(args.duration)
        sf.write(temp_audio, audio, sr)
        print(f"Saved recording to {temp_audio}")

    # Extract features
    print("\n🔍 Extracting audio features...")
    audio_features = extract_audio_features(temp_audio)

    print("\n📊 Audio feature summary:")
    print(f"  Shape: {audio_features.shape}")
    print(f"  Envelope (energy): min={audio_features[:,0].min():.3f}, max={audio_features[:,0].max():.3f}")
    print(f"  MFCC range: {audio_features[:,1:21].min():.2f} to {audio_features[:,1:21].max():.2f}")
    print(f"  Peak onsets detected: {int(audio_features[:,33].sum())}")
    print(f"  Beat onsets detected: {int(audio_features[:,34].sum())}")

    # Load model and generate
    print("\n🤖 Loading FACT model...")
    model = load_model(
        'mint/configs/fact_v5_deeper_t10_cm12.config',
        'mint/checkpoints'
    )

    seed_motion = create_neutral_motion(n_frames=120)
    steps = int(args.generate * 60)

    print(f"\n💃 Generating {args.generate} seconds of dance...")
    print("   (This takes ~30 seconds on CPU)")

    start = time.time()
    generated_motion = generate_dance(model, audio_features, seed_motion, steps=steps)
    elapsed = time.time() - start

    print(f"   Generated {generated_motion.shape[0]} frames in {elapsed:.1f}s")
    print(f"   ({generated_motion.shape[0]/elapsed:.1f} frames/sec)")

    # Convert to joint angles and analyze
    joint_angles = motion_to_joint_angles(generated_motion)

    analyze_motion(joint_angles)

    # Save outputs
    print("\n💾 Saved outputs:")
    np.save('fact_test_motion.npy', generated_motion)
    np.save('fact_test_angles.npy', joint_angles)
    print("  → fact_test_motion.npy")
    print("  → fact_test_angles.npy")

    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
