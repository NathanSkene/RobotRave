#!/usr/bin/env python3
"""
Play dance commands on Tony Pro robot with optional audio sync.

This script reads dance JSON files (from retarget_to_tonypi.py or batch_dance.py)
and sends servo commands to the robot via the HiwonderSDK.

Usage (on the robot):
    python play_dance.py --input dance.json
    python play_dance.py --input dance.json --audio song.mp3

For testing without robot:
    python play_dance.py --input dance.json --simulate
"""

import argparse
import json
import time
import sys
import threading

# Import the Tony Pro controller
from tony_pro import TonyProController, get_neutral

# Optional audio playback
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# Safe mode limits
SAFE_BUS_MIN = 300
SAFE_BUS_MAX = 700
SAFE_PWM_MIN = 1200
SAFE_PWM_MAX = 1800


def clamp_to_safe(frames):
    """Clamp all servo values to safe central ranges.

    Bus servos (1-16): 300-700 (instead of 0-1000)
    PWM servos (pwm1, pwm2): 1200-1800 (instead of 500-2500)
    """
    safe_frames = []
    for frame in frames:
        safe_servos = {}
        for servo_id, pulse in frame['servos'].items():
            if str(servo_id).startswith('pwm'):
                # PWM servo (head)
                safe_servos[servo_id] = max(SAFE_PWM_MIN, min(SAFE_PWM_MAX, pulse))
            else:
                # Bus servo (body)
                safe_servos[servo_id] = max(SAFE_BUS_MIN, min(SAFE_BUS_MAX, pulse))
        safe_frames.append({'time_ms': frame['time_ms'], 'servos': safe_servos})
    return safe_frames


def remove_servos(frames, skip_list):
    """Remove specified servos from all frames.

    Args:
        frames: List of frame dicts
        skip_list: List of servo IDs to skip (can be int or string like 'pwm1')
    """
    # Normalize skip list to strings for comparison
    skip_set = set(str(s).strip() for s in skip_list if str(s).strip())

    filtered_frames = []
    for frame in frames:
        filtered_servos = {}
        for servo_id, pulse in frame['servos'].items():
            if str(servo_id) not in skip_set:
                filtered_servos[servo_id] = pulse
        filtered_frames.append({'time_ms': frame['time_ms'], 'servos': filtered_servos})
    return filtered_frames


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


def load_audio(audio_path):
    """Load audio file for playback.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)

    Returns:
        (audio_data, sample_rate) or (None, None) if unavailable
    """
    if not AUDIO_AVAILABLE:
        print("Warning: Audio playback requires sounddevice and soundfile")
        print("Install with: pip install sounddevice soundfile")
        return None, None

    try:
        audio_data, sample_rate = sf.read(audio_path)
        print(f"Loaded audio: {len(audio_data)/sample_rate:.1f}s at {sample_rate}Hz")
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


def play_sequence_with_audio(controller, frames, audio_data, sample_rate, fps=60, loop=False):
    """Play dance sequence synced with audio.

    Args:
        controller: RobotController instance
        frames: List of frame dicts from JSON
        audio_data: Audio array from soundfile
        sample_rate: Audio sample rate
        fps: Dance frame rate
        loop: Whether to loop
    """
    frame_time = 1.0 / fps
    n_frames = len(frames)
    audio_duration = len(audio_data) / sample_rate if audio_data is not None else 0

    print(f"Playing {n_frames} frames at {fps} FPS with audio")
    print(f"Dance duration: {n_frames/fps:.1f}s, Audio duration: {audio_duration:.1f}s")
    print("Press Ctrl+C to stop")

    # Audio playback in separate thread
    audio_event = threading.Event()

    def play_audio():
        audio_event.set()  # Signal that audio is starting
        sd.play(audio_data, sample_rate)
        sd.wait()

    try:
        while True:
            # Start audio playback
            if audio_data is not None:
                audio_event.clear()
                audio_thread = threading.Thread(target=play_audio, daemon=True)
                audio_thread.start()
                # Wait for audio to actually start
                audio_event.wait(timeout=1.0)

            # Play frames synced to wall clock
            start_time = time.time()

            for i, frame in enumerate(frames):
                # Calculate when this frame should play
                target_time = start_time + (i * frame_time)
                current_time = time.time()

                # If we're behind, skip frame
                if current_time > target_time + frame_time:
                    continue

                # Wait until it's time for this frame
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Send servo commands
                controller.set_servos(frame['servos'], frame['time_ms'])

                # Progress indicator
                if i % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"Frame {i}/{n_frames} ({100*i/n_frames:.0f}%) - Time: {elapsed:.1f}s", end='\r')

            print(f"\nSequence complete!")

            if not loop:
                break

            # Stop audio before looping
            if audio_data is not None:
                sd.stop()
            print("Looping...")

    except KeyboardInterrupt:
        print("\nStopped by user")
        if audio_data is not None:
            sd.stop()


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
    parser.add_argument('--audio', type=str, default=None, help='Audio file to play with dance')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode (no robot)')
    parser.add_argument('--loop', action='store_true', help='Loop the sequence')
    parser.add_argument('--fps', type=int, default=None, help='Override FPS')
    parser.add_argument('--neutral-first', action='store_true', help='Go to neutral before playing')
    parser.add_argument('--neutral-after', action='store_true', help='Go to neutral after playing')
    parser.add_argument('--safe', action='store_true', help='Safe mode: clamp servos to central range (300-700 bus, 1200-1800 PWM)')
    parser.add_argument('--skip-servos', type=str, default='', help='Comma-separated list of servo IDs to skip (e.g., "15,16,pwm1")')
    args = parser.parse_args()

    # Load dance data
    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    fps = args.fps if args.fps else data.get('fps', 60)

    # Apply safe mode clamping if requested
    if args.safe:
        print("SAFE MODE: Clamping servos to central range (300-700 bus, 1200-1800 PWM)")
        frames = clamp_to_safe(frames)

    # Remove skipped servos
    if args.skip_servos:
        skip_list = args.skip_servos.split(',')
        print(f"SKIPPING SERVOS: {skip_list}")
        frames = remove_servos(frames, skip_list)

    print(f"Loaded {len(frames)} frames, {fps} FPS")
    print(f"Duration: {len(frames)/fps:.1f} seconds")
    if 'active_servos' in data:
        print(f"Active servos: {data['active_servos']}")

    # Load audio if provided
    audio_data, sample_rate = None, None
    if args.audio:
        audio_data, sample_rate = load_audio(args.audio)

    # Initialize robot
    controller = RobotController(simulate=args.simulate)

    # Go to neutral first if requested
    if args.neutral_first:
        controller.go_to_neutral()

    # Play the sequence
    if audio_data is not None:
        play_sequence_with_audio(controller, frames, audio_data, sample_rate,
                                  fps=fps, loop=args.loop)
    else:
        play_sequence(controller, frames, fps=fps, loop=args.loop)

    # Return to neutral if requested
    if args.neutral_after:
        controller.go_to_neutral()


if __name__ == '__main__':
    main()
