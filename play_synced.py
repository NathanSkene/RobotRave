#!/usr/bin/env python3
"""
Play music on laptop while robot dances - synchronized playback.

Usage:
    python3 play_synced.py
    python3 play_synced.py --audio mHO2.mp3
    python3 play_synced.py --countdown 5
    python3 play_synced.py --no-audio  # Robot only, no music
"""

import subprocess
import threading
import time
import argparse
import os

ROBOT_HOST = "pi@raspberrypi.local"
ROBOT_CHOREO = "choreo.json"
ROBOT_ACTION_DIR = "/home/pi/TonyPi/ActionGroups"


def play_audio(audio_path):
    """Play audio on laptop."""
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(audio_path)
        print(f"Playing audio: {audio_path} ({len(data)/sr:.1f}s)")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"Audio error: {e}")


def start_robot():
    """Start choreography on robot via SSH."""
    cmd = f"cd ~ && python3 play_choreography.py --input {ROBOT_CHOREO} --action-dir {ROBOT_ACTION_DIR} --verbose"
    print(f"Starting robot: {cmd}")
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', ROBOT_HOST, cmd],
        capture_output=False
    )
    return result.returncode


def check_ssh():
    """Check if passwordless SSH works."""
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', ROBOT_HOST, 'echo ok'],
        capture_output=True,
        text=True
    )
    return result.returncode == 0 and 'ok' in result.stdout


def main():
    parser = argparse.ArgumentParser(description='Synced robot dance with laptop audio')
    parser.add_argument('--audio', type=str, default='mHO2.mp3', help='Audio file path')
    parser.add_argument('--countdown', type=int, default=3, help='Countdown seconds')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio playback')
    parser.add_argument('--test-ssh', action='store_true', help='Test SSH connection only')
    args = parser.parse_args()

    # Test SSH connection
    print("Checking SSH connection...")
    if not check_ssh():
        print("ERROR: Passwordless SSH not working!")
        print("Run this to set it up:")
        print("  ssh-copy-id pi@raspberrypi.local")
        return 1

    print("SSH connection OK")

    if args.test_ssh:
        return 0

    # Check audio file
    if not args.no_audio and not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        print("Copy it from robot with:")
        print(f"  scp pi@raspberrypi.local:~/mHO2.mp3 .")
        return 1

    # Countdown
    for i in range(args.countdown, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("GO!")

    # Start both threads
    robot_thread = threading.Thread(target=start_robot)
    robot_thread.start()

    # Small delay to account for SSH connection time
    time.sleep(0.3)

    if not args.no_audio:
        play_audio(args.audio)

    robot_thread.join()
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
