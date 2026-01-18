#!/usr/bin/env python3
"""
Play choreography on TonyPi robot with optional audio sync.

Reads a choreography JSON file (from generate_choreography.py) and triggers
pre-scripted .d6a actions at the specified times.

Usage (on robot):
    python play_choreography.py --input choreo.json
    python play_choreography.py --input choreo.json --audio song.mp3

For testing without robot:
    python play_choreography.py --input choreo.json --simulate
"""

import argparse
import json
import time
import threading
from pathlib import Path

from read_d6a import read_d6a
from tony_pro import TonyProController

# Optional audio playback
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def load_choreography(choreo_path):
    """Load choreography from JSON file.

    Args:
        choreo_path: Path to choreography JSON

    Returns:
        Choreography dict with 'choreography' list
    """
    with open(choreo_path, 'r') as f:
        return json.load(f)


def load_action(action_name, action_dir):
    """Load a .d6a action file.

    Args:
        action_name: Name of action (without .d6a extension)
        action_dir: Directory containing .d6a files

    Returns:
        List of frame dicts, or None if not found
    """
    filepath = Path(action_dir) / f"{action_name}.d6a"

    if not filepath.exists():
        print(f"Warning: Action not found: {filepath}")
        return None

    try:
        return read_d6a(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def play_action_frames(controller, frames):
    """Play a sequence of frames on the robot.

    Args:
        controller: TonyProController instance
        frames: List of frame dicts from .d6a file
    """
    for frame in frames:
        time_ms = frame['time_ms']
        servos = frame['servos']

        # Send servo commands
        controller.set_servos(servos, time_ms)

        # Wait for frame duration
        time.sleep(time_ms / 1000.0)


def play_choreography(controller, choreo_data, action_dir, verbose=False):
    """Play choreography without audio.

    Args:
        controller: TonyProController instance
        choreo_data: Choreography dict
        action_dir: Directory containing .d6a files
        verbose: Print action triggers
    """
    choreography = choreo_data['choreography']
    total_duration_ms = choreo_data.get('total_duration_ms', 0)

    print(f"Playing {len(choreography)} action triggers over {total_duration_ms/1000:.1f}s")
    print("Press Ctrl+C to stop")

    # Pre-load all unique actions
    unique_actions = set(entry['action'] for entry in choreography)
    action_cache = {}

    print(f"Loading {len(unique_actions)} unique actions...")
    for action_name in unique_actions:
        frames = load_action(action_name, action_dir)
        if frames:
            action_cache[action_name] = frames
            if verbose:
                duration = sum(f['time_ms'] for f in frames)
                print(f"  {action_name}: {len(frames)} frames, {duration}ms")

    # Schedule and play
    start_time = time.time()

    try:
        for i, entry in enumerate(choreography):
            trigger_time_s = entry['time_ms'] / 1000.0
            action_name = entry['action']

            # Wait until trigger time
            current_time = time.time() - start_time
            wait_time = trigger_time_s - current_time

            if wait_time > 0:
                time.sleep(wait_time)

            # Play the action
            frames = action_cache.get(action_name)
            if frames:
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Playing: {action_name}")
                play_action_frames(controller, frames)
            else:
                if verbose:
                    print(f"[{trigger_time_s:.1f}s] Skipping: {action_name} (not loaded)")

        print("\nChoreography complete!")

    except KeyboardInterrupt:
        print("\n\nStopped by user")


def play_choreography_with_audio(controller, choreo_data, action_dir,
                                  audio_path, verbose=False):
    """Play choreography synced with audio.

    Audio and actions run in parallel, with actions triggered at their
    specified times relative to audio start.

    Args:
        controller: TonyProController instance
        choreo_data: Choreography dict
        action_dir: Directory containing .d6a files
        audio_path: Path to audio file
        verbose: Print action triggers
    """
    if not AUDIO_AVAILABLE:
        print("Audio playback requires sounddevice and soundfile")
        print("Install with: pip install sounddevice soundfile")
        return

    choreography = choreo_data['choreography']

    # Load audio
    try:
        audio_data, sample_rate = sf.read(audio_path)
        audio_duration = len(audio_data) / sample_rate
        print(f"Audio: {audio_duration:.1f}s at {sample_rate}Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # Pre-load actions
    unique_actions = set(entry['action'] for entry in choreography)
    action_cache = {}

    print(f"Loading {len(unique_actions)} unique actions...")
    for action_name in unique_actions:
        frames = load_action(action_name, action_dir)
        if frames:
            action_cache[action_name] = frames

    print(f"Playing {len(choreography)} action triggers synced with audio")
    print("Press Ctrl+C to stop")

    # Start audio in background thread
    audio_started = threading.Event()

    def play_audio():
        audio_started.set()
        sd.play(audio_data, sample_rate)
        sd.wait()

    audio_thread = threading.Thread(target=play_audio, daemon=True)
    audio_thread.start()

    # Wait for audio to start
    audio_started.wait(timeout=1.0)
    start_time = time.time()

    try:
        for entry in choreography:
            trigger_time_s = entry['time_ms'] / 1000.0
            action_name = entry['action']

            # Wait until trigger time
            current_time = time.time() - start_time
            wait_time = trigger_time_s - current_time

            if wait_time > 0:
                time.sleep(wait_time)
            elif wait_time < -0.5:
                # We're behind, skip this action
                if verbose:
                    print(f"[SKIP] {action_name} (behind by {-wait_time:.1f}s)")
                continue

            # Play the action
            frames = action_cache.get(action_name)
            if frames:
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Playing: {action_name}")
                play_action_frames(controller, frames)

        print("\nChoreography complete!")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        sd.stop()


def preview_choreography(choreo_data, action_dir):
    """Preview choreography timeline without playing.

    Args:
        choreo_data: Choreography dict
        action_dir: Directory containing .d6a files
    """
    choreography = choreo_data['choreography']
    total_ms = choreo_data.get('total_duration_ms', 0)

    print(f"\nChoreography Preview ({total_ms/1000:.1f}s total)")
    print("=" * 60)

    # Check which actions exist
    for entry in choreography:
        time_s = entry['time_ms'] / 1000.0
        action_name = entry['action']
        score = entry.get('score', 0)

        # Check if action exists
        action_path = Path(action_dir) / f"{action_name}.d6a"
        exists = action_path.exists()
        status = "" if exists else " [NOT FOUND]"

        print(f"  {time_s:6.1f}s: {action_name:25s} (score: {score:.2f}){status}")

    # Summary
    unique_actions = set(entry['action'] for entry in choreography)
    missing = [a for a in unique_actions
               if not (Path(action_dir) / f"{a}.d6a").exists()]

    print("=" * 60)
    print(f"Total triggers: {len(choreography)}")
    print(f"Unique actions: {len(unique_actions)}")

    if missing:
        print(f"\nMissing actions ({len(missing)}):")
        for name in sorted(missing):
            print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Play choreography on TonyPi robot"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input choreography JSON file")
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="Audio file to play with choreography")
    parser.add_argument("--action-dir", type=str, default="./action_groups",
                        help="Directory containing .d6a action files")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulation mode (no robot)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print action triggers as they play")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Preview timeline without playing")
    parser.add_argument("--neutral-first", action="store_true",
                        help="Go to neutral position before playing")
    parser.add_argument("--neutral-after", action="store_true",
                        help="Go to neutral position after playing")

    args = parser.parse_args()

    # Load choreography
    print(f"Loading choreography from {args.input}...")
    choreo_data = load_choreography(args.input)

    n_triggers = len(choreo_data.get('choreography', []))
    duration_ms = choreo_data.get('total_duration_ms', 0)
    print(f"Loaded: {n_triggers} triggers, {duration_ms/1000:.1f}s duration")

    # Preview mode
    if args.preview:
        preview_choreography(choreo_data, args.action_dir)
        return

    # Initialize controller
    controller = TonyProController(simulate=args.simulate)

    if args.simulate:
        print("SIMULATION MODE - no robot commands sent")

    # Go to neutral first if requested
    if args.neutral_first:
        print("Going to neutral position...")
        controller.go_neutral(time_ms=1000)
        time.sleep(1.2)

    # Play
    try:
        if args.audio:
            play_choreography_with_audio(
                controller, choreo_data, args.action_dir,
                args.audio, verbose=args.verbose
            )
        else:
            play_choreography(
                controller, choreo_data, args.action_dir,
                verbose=args.verbose
            )
    finally:
        # Go to neutral after if requested
        if args.neutral_after:
            print("Going to neutral position...")
            controller.go_neutral(time_ms=1000)
            time.sleep(1.2)


if __name__ == "__main__":
    main()
