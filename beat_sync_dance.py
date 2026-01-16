#!/usr/bin/env python3
"""
Fallback Beat-Sync Dance Script for Tony Pro Robot.

This is a simpler alternative to FACT-generated dances.
It uses aubio for real-time beat detection and triggers
pre-defined dance moves on beats and onsets.

Usage (on the robot):
    python beat_sync_dance.py --audio your_song.wav

For real-time mic input:
    python beat_sync_dance.py --live
"""

import argparse
import json
import time
import sys
import random
import numpy as np

# Try to import aubio
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    print("Warning: aubio not found. Install with: pip install aubio")

# Try to import sounddevice for live input
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Import Tony Pro configuration
from tony_pro import (
    TonyProController,
    MOVES as DANCE_MOVES,
    BEAT_SEQUENCES,
    ONSET_MOVES,
)


class BeatSyncDancer:
    """Beat-synchronized dancing controller."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self._controller = TonyProController(simulate=simulate)
        self.current_move = 'neutral'
        self.beat_count = 0
        self.current_sequence = random.choice(BEAT_SEQUENCES)
        self.sequence_index = 0
        self.last_beat_time = 0
        self.min_beat_interval = 0.2  # Minimum time between beats (200ms)

    def set_move(self, move_name, time_ms=200):
        """Execute a dance move."""
        if move_name not in DANCE_MOVES:
            return

        self.current_move = move_name

        if self.simulate:
            print(f"  >> {move_name}")

        self._controller.execute_move(move_name, time_ms)

    def on_beat(self):
        """Called when a beat is detected."""
        now = time.time()
        if now - self.last_beat_time < self.min_beat_interval:
            return  # Too soon after last beat
        self.last_beat_time = now

        self.beat_count += 1

        # Execute next move in sequence
        move = self.current_sequence[self.sequence_index]
        self.set_move(move, time_ms=150)

        # Advance sequence
        self.sequence_index = (self.sequence_index + 1) % len(self.current_sequence)

        # Occasionally change sequence
        if self.beat_count % 16 == 0:
            self.current_sequence = random.choice(BEAT_SEQUENCES)
            self.sequence_index = 0
            print(f"  [New sequence at beat {self.beat_count}]")

    def on_onset(self, strength=1.0):
        """Called when a strong onset/transient is detected."""
        if strength > 0.8:  # Only trigger on strong onsets
            move = random.choice(ONSET_MOVES)
            self.set_move(move, time_ms=100)

    def return_to_neutral(self):
        """Return to neutral position."""
        self._controller.execute_move('neutral', time_ms=500)


def process_audio_file(audio_path, dancer, sample_rate=44100):
    """Process an audio file and trigger dance moves on beats."""
    import soundfile as sf

    print(f"Loading audio: {audio_path}")
    audio_data, orig_sr = sf.read(audio_path)

    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if orig_sr != sample_rate:
        from scipy import signal
        num_samples = int(len(audio_data) * sample_rate / orig_sr)
        audio_data = signal.resample(audio_data, num_samples)

    audio_data = audio_data.astype(np.float32)

    # Setup aubio beat tracker
    win_size = 1024
    hop_size = 512

    tempo_detector = aubio.tempo("default", win_size, hop_size, sample_rate)
    onset_detector = aubio.onset("energy", win_size, hop_size, sample_rate)
    onset_detector.set_threshold(0.5)

    print(f"Playing {len(audio_data)/sample_rate:.1f} seconds of audio...")
    print("Press Ctrl+C to stop")
    print()

    # Process in chunks
    n_samples = len(audio_data)
    position = 0
    start_time = time.time()

    try:
        while position < n_samples - hop_size:
            # Get chunk
            chunk = audio_data[position:position + hop_size]

            # Detect beat
            is_beat = tempo_detector(chunk)
            if is_beat:
                bpm = tempo_detector.get_bpm()
                print(f"Beat! BPM: {bpm:.0f}")
                dancer.on_beat()

            # Detect onset
            is_onset = onset_detector(chunk)
            if is_onset:
                dancer.on_onset(onset_detector.get_last())

            position += hop_size

            # Sync with real time
            elapsed = time.time() - start_time
            audio_time = position / sample_rate
            if audio_time > elapsed:
                time.sleep(audio_time - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user")

    dancer.return_to_neutral()
    print(f"\nDone! Total beats: {dancer.beat_count}")


def process_live_audio(dancer, sample_rate=44100):
    """Process live microphone input and trigger dance moves."""
    if not SOUNDDEVICE_AVAILABLE:
        print("Error: sounddevice not available for live input")
        return

    win_size = 1024
    hop_size = 512

    tempo_detector = aubio.tempo("default", win_size, hop_size, sample_rate)
    onset_detector = aubio.onset("energy", win_size, hop_size, sample_rate)
    onset_detector.set_threshold(0.5)

    print("Listening for music...")
    print("Press Ctrl+C to stop")
    print()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        # Convert to mono float32
        audio = indata[:, 0].astype(np.float32)

        # Process in hop_size chunks
        for i in range(0, len(audio) - hop_size, hop_size):
            chunk = audio[i:i + hop_size]

            is_beat = tempo_detector(chunk)
            if is_beat:
                bpm = tempo_detector.get_bpm()
                print(f"Beat! BPM: {bpm:.0f}")
                dancer.on_beat()

            is_onset = onset_detector(chunk)
            if is_onset:
                dancer.on_onset(onset_detector.get_last())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=win_size,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user")

    dancer.return_to_neutral()
    print(f"\nDone! Total beats: {dancer.beat_count}")


def test_moves(dancer):
    """Test all dance moves one by one."""
    print("Testing all dance moves...")
    print()

    for move_name in DANCE_MOVES:
        print(f"Move: {move_name}")
        dancer.set_move(move_name, time_ms=500)
        time.sleep(1.0)

    dancer.return_to_neutral()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Beat-sync dancing for TonyPi')
    parser.add_argument('--audio', type=str, help='Audio file to process')
    parser.add_argument('--live', action='store_true', help='Use live microphone input')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode (no robot)')
    parser.add_argument('--test', action='store_true', help='Test all dance moves')
    args = parser.parse_args()

    if not AUBIO_AVAILABLE and not args.test:
        print("Error: aubio required for beat detection")
        print("Install with: pip install aubio")
        sys.exit(1)

    dancer = BeatSyncDancer(simulate=args.simulate)

    if args.test:
        test_moves(dancer)
    elif args.live:
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice required for live input")
            print("Install with: pip install sounddevice")
            sys.exit(1)
        process_live_audio(dancer)
    elif args.audio:
        process_audio_file(args.audio, dancer)
    else:
        print("Usage:")
        print("  python beat_sync_dance.py --audio your_song.wav")
        print("  python beat_sync_dance.py --live")
        print("  python beat_sync_dance.py --test --simulate")


if __name__ == '__main__':
    main()
