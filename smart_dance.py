#!/usr/bin/env python3
"""
Smart Beat-Sync Dance - Adapts style based on music genre/vibe.

Detects:
- Tempo (BPM): slow/medium/fast
- Energy: chill/moderate/intense
- Brightness: bass-heavy vs treble-heavy

Maps to dance styles:
- CHILL: Slow, smooth wave movements
- POP: Classic upbeat dance moves
- EDM: Fast, energetic, punchy moves
- HIPHOP: Head bobs, punches, swagger

Usage:
    python smart_dance.py --live
    python smart_dance.py --audio song.wav
"""

import argparse
import time
import sys
import random
import numpy as np
from collections import deque

try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    print("Error: aubio required. Install with: pip install aubio")
    sys.exit(1)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Import Tony Pro configuration
from tony_pro import SERVO, TonyProController


# =============================================================================
# DANCE STYLES - Different move sets for different music vibes
# =============================================================================

# Each move uses the correct Tony Pro servo IDs from tony_pro.py
# SERVO.R_SHOULDER=7, SERVO.R_ELBOW=5, SERVO.L_SHOULDER=15, SERVO.L_ELBOW=13
# SERVO.HEAD_PITCH=8, SERVO.HEAD_YAW=16

STYLE_CHILL = {
    'name': 'CHILL',
    'description': 'Slow, smooth, flowing movements',
    'moves': {
        'sway_right': {
            SERVO.R_SHOULDER: 550, SERVO.R_ELBOW: 550,
            SERVO.L_SHOULDER: 450, SERVO.L_ELBOW: 450,
            SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 500,
        },
        'sway_left': {
            SERVO.R_SHOULDER: 450, SERVO.R_ELBOW: 450,
            SERVO.L_SHOULDER: 550, SERVO.L_ELBOW: 550,
            SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 500,
        },
        'gentle_wave': {
            SERVO.R_SHOULDER: 400, SERVO.R_ELBOW: 550,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 450,
        },
        'neutral': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 500,
        },
    },
    'sequences': [
        ['sway_right', 'neutral', 'sway_left', 'neutral'],
        ['gentle_wave', 'neutral', 'sway_right', 'sway_left'],
    ],
    'move_time_ms': 400,  # Slower movements
}

STYLE_POP = {
    'name': 'POP',
    'description': 'Classic upbeat dance moves',
    'moves': {
        'arms_up': {
            SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 400,
        },
        'arms_out': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 300,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 700,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 500,
        },
        'clap_ready': {
            SERVO.R_SHOULDER: 400, SERVO.R_ELBOW: 600,
            SERVO.L_SHOULDER: 600, SERVO.L_ELBOW: 400,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 450,
        },
        'point_right': {
            SERVO.R_SHOULDER: 350, SERVO.R_ELBOW: 400,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 600, SERVO.HEAD_PITCH: 500,
        },
        'point_left': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 650, SERVO.L_ELBOW: 600,
            SERVO.HEAD_YAW: 400, SERVO.HEAD_PITCH: 500,
        },
        'neutral': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 500,
        },
    },
    'sequences': [
        ['arms_up', 'neutral', 'arms_out', 'neutral'],
        ['point_right', 'neutral', 'point_left', 'neutral'],
        ['clap_ready', 'neutral', 'arms_up', 'neutral'],
    ],
    'move_time_ms': 200,
}

STYLE_EDM = {
    'name': 'EDM',
    'description': 'Fast, energetic, pumping moves',
    'moves': {
        'pump_up': {
            SERVO.R_SHOULDER: 250, SERVO.R_ELBOW: 400,
            SERVO.L_SHOULDER: 750, SERVO.L_ELBOW: 600,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 350,
        },
        'pump_down': {
            SERVO.R_SHOULDER: 400, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 600, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 550,
        },
        'rave_hands': {
            SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 350,
            SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 650,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 400,
        },
        'fist_pump_r': {
            SERVO.R_SHOULDER: 200, SERVO.R_ELBOW: 300,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 400,
        },
        'fist_pump_l': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 800, SERVO.L_ELBOW: 700,
            SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 400,
        },
        'neutral': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 500,
        },
    },
    'sequences': [
        ['pump_up', 'pump_down', 'pump_up', 'pump_down'],
        ['fist_pump_r', 'neutral', 'fist_pump_l', 'neutral'],
        ['rave_hands', 'neutral', 'pump_up', 'pump_down'],
    ],
    'move_time_ms': 120,  # Fast movements
}

STYLE_HIPHOP = {
    'name': 'HIP-HOP',
    'description': 'Head bobs, punches, swagger',
    'moves': {
        'head_bob': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 600,
        },
        'head_up': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 400,
        },
        'right_punch': {
            SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 250,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 600, SERVO.HEAD_PITCH: 500,
        },
        'left_punch': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 750,
            SERVO.HEAD_YAW: 400, SERVO.HEAD_PITCH: 500,
        },
        'swagger_r': {
            SERVO.R_SHOULDER: 450, SERVO.R_ELBOW: 450,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 400,
            SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 520,
        },
        'swagger_l': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 400,
            SERVO.L_SHOULDER: 550, SERVO.L_ELBOW: 550,
            SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 520,
        },
        'neutral': {
            SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
            SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
            SERVO.HEAD_YAW: 500, SERVO.HEAD_PITCH: 500,
        },
    },
    'sequences': [
        ['head_bob', 'head_up', 'head_bob', 'head_up'],
        ['right_punch', 'neutral', 'left_punch', 'neutral'],
        ['swagger_r', 'swagger_l', 'swagger_r', 'swagger_l'],
        ['head_bob', 'right_punch', 'head_bob', 'left_punch'],
    ],
    'move_time_ms': 180,
}

ALL_STYLES = {
    'chill': STYLE_CHILL,
    'pop': STYLE_POP,
    'edm': STYLE_EDM,
    'hiphop': STYLE_HIPHOP,
}


# =============================================================================
# MUSIC ANALYZER - Detects genre/vibe from audio features
# =============================================================================

class MusicAnalyzer:
    """Analyzes audio to detect music style/vibe."""

    def __init__(self, sample_rate=44100, history_seconds=3):
        self.sample_rate = sample_rate
        self.win_size = 1024
        self.hop_size = 512

        # Aubio analyzers
        self.tempo_detector = aubio.tempo("default", self.win_size, self.hop_size, sample_rate)
        self.onset_detector = aubio.onset("energy", self.win_size, self.hop_size, sample_rate)

        # Rolling history for analysis
        history_frames = int(history_seconds * sample_rate / self.hop_size)
        self.energy_history = deque(maxlen=history_frames)
        self.brightness_history = deque(maxlen=history_frames)
        self.onset_history = deque(maxlen=history_frames)
        self.bpm_history = deque(maxlen=30)  # Last 30 BPM readings

        # Current detected values
        self.current_bpm = 120
        self.current_energy = 0.5
        self.current_brightness = 0.5
        self.onset_density = 0
        self.current_style = 'pop'

        # FFT for spectral analysis (use hop_size for analysis chunks)
        self.fft_freqs = np.fft.rfftfreq(self.hop_size, 1.0 / sample_rate)

    def analyze_chunk(self, audio_chunk):
        """Analyze a chunk of audio and update features."""

        # Ensure correct type
        audio = np.ascontiguousarray(audio_chunk, dtype=np.float32)

        # Beat detection
        is_beat = self.tempo_detector(audio)
        if is_beat:
            bpm = self.tempo_detector.get_bpm()
            if 60 < bpm < 200:  # Sanity check
                self.bpm_history.append(bpm)
                if self.bpm_history:
                    self.current_bpm = np.median(list(self.bpm_history))

        # Energy (RMS)
        energy = np.sqrt(np.mean(audio ** 2))
        self.energy_history.append(energy)

        # Spectral brightness (centroid)
        spectrum = np.abs(np.fft.rfft(audio))
        if spectrum.sum() > 0:
            brightness = np.sum(self.fft_freqs * spectrum) / np.sum(spectrum)
            brightness_norm = min(brightness / 4000, 1.0)  # Normalize to 0-1
            self.brightness_history.append(brightness_norm)

        # Onset detection
        is_onset = self.onset_detector(audio)
        self.onset_history.append(1 if is_onset else 0)

        # Update aggregated values
        if self.energy_history:
            energies = list(self.energy_history)
            self.current_energy = np.mean(energies) / (np.max(energies) + 1e-8)

        if self.brightness_history:
            self.current_brightness = np.mean(list(self.brightness_history))

        if self.onset_history:
            self.onset_density = np.mean(list(self.onset_history))

        return is_beat, is_onset

    def detect_style(self):
        """Determine the current music style based on features."""

        bpm = self.current_bpm
        energy = self.current_energy
        brightness = self.current_brightness

        # Classification logic
        if bpm < 100:
            # Slow music
            style = 'chill'
        elif bpm > 140 and energy > 0.6:
            # Fast and energetic
            style = 'edm'
        elif brightness < 0.4 and bpm < 130:
            # Bass-heavy, moderate tempo
            style = 'hiphop'
        else:
            # Default upbeat
            style = 'pop'

        # Only change style if confident (avoid rapid switching)
        if style != self.current_style:
            self.current_style = style

        return self.current_style

    def get_status(self):
        """Get current analysis status as string."""
        return (f"BPM: {self.current_bpm:.0f} | "
                f"Energy: {self.current_energy:.2f} | "
                f"Brightness: {self.current_brightness:.2f}")


# =============================================================================
# SMART DANCER - Dances according to detected style
# =============================================================================

class SmartDancer:
    """Dances with style-aware movements."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self._controller = TonyProController(simulate=simulate)

        self.current_style_name = 'pop'
        self.current_style = STYLE_POP
        self.current_sequence = self.current_style['sequences'][0]
        self.sequence_index = 0
        self.beat_count = 0
        self.last_beat_time = 0
        self.min_beat_interval = 0.15

    def set_style(self, style_name):
        """Change to a new dance style."""
        if style_name == self.current_style_name:
            return

        if style_name in ALL_STYLES:
            self.current_style_name = style_name
            self.current_style = ALL_STYLES[style_name]
            self.current_sequence = random.choice(self.current_style['sequences'])
            self.sequence_index = 0
            print(f"\n🎭 STYLE CHANGE → {self.current_style['name']}")
            print(f"   {self.current_style['description']}")

    def execute_move(self, move_name):
        """Execute a dance move."""
        moves = self.current_style['moves']
        if move_name not in moves:
            return

        move = moves[move_name]
        time_ms = self.current_style['move_time_ms']

        if self.simulate:
            print(f"  -> {move_name}")

        self._controller.set_servos(move, time_ms)

    def on_beat(self):
        """Handle beat detection."""
        now = time.time()
        if now - self.last_beat_time < self.min_beat_interval:
            return False
        self.last_beat_time = now
        self.beat_count += 1

        # Execute next move in sequence
        move_name = self.current_sequence[self.sequence_index]
        self.execute_move(move_name)

        # Advance sequence
        self.sequence_index = (self.sequence_index + 1) % len(self.current_sequence)

        # Change sequence every 16 beats
        if self.beat_count % 16 == 0:
            self.current_sequence = random.choice(self.current_style['sequences'])
            self.sequence_index = 0

        return True

    def on_strong_onset(self):
        """Handle strong onset/transient."""
        # Pick a random energetic move from current style
        moves = list(self.current_style['moves'].keys())
        moves = [m for m in moves if m != 'neutral']
        if moves:
            self.execute_move(random.choice(moves))

    def return_to_neutral(self):
        """Return to neutral position."""
        self._controller.go_neutral(time_ms=500)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run_live(dancer, analyzer, duration=None):
    """Run with live microphone input."""

    if not SOUNDDEVICE_AVAILABLE:
        print("Error: sounddevice required for live input")
        return

    sample_rate = analyzer.sample_rate
    hop_size = analyzer.hop_size

    print("\n" + "=" * 60)
    print("🎤 SMART DANCE - Listening to music...")
    print("=" * 60)
    print("\nDetecting style from: tempo, energy, brightness")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    last_status_time = start_time

    def callback(indata, frames, time_info, status):
        nonlocal last_status_time

        audio = indata[:, 0].astype(np.float32)

        # Process in chunks
        for i in range(0, len(audio) - hop_size, hop_size):
            chunk = audio[i:i + hop_size]

            is_beat, is_onset = analyzer.analyze_chunk(chunk)

            if is_beat:
                # Update style based on analysis
                style = analyzer.detect_style()
                dancer.set_style(style)

                # Dance!
                if dancer.on_beat():
                    bpm = analyzer.current_bpm
                    print(f"🥁 Beat {dancer.beat_count} | BPM: {bpm:.0f} | Style: {dancer.current_style['name']}")

            if is_onset and analyzer.onset_detector.get_last() > 0.9:
                dancer.on_strong_onset()

        # Print status every 2 seconds
        now = time.time()
        if now - last_status_time > 2:
            print(f"   [{analyzer.get_status()}]")
            last_status_time = now

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=analyzer.win_size,
            callback=callback
        ):
            if duration:
                time.sleep(duration)
            else:
                while True:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    dancer.return_to_neutral()
    print(f"\n{'=' * 60}")
    print(f"Done! Total beats: {dancer.beat_count}")
    print(f"Final style: {dancer.current_style['name']}")


def run_file(audio_path, dancer, analyzer):
    """Run with audio file input."""

    if not SOUNDFILE_AVAILABLE:
        print("Error: soundfile required")
        return

    print(f"Loading: {audio_path}")
    audio_data, orig_sr = sf.read(audio_path)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if orig_sr != analyzer.sample_rate:
        from scipy import signal
        num_samples = int(len(audio_data) * analyzer.sample_rate / orig_sr)
        audio_data = signal.resample(audio_data, num_samples)

    audio_data = audio_data.astype(np.float32)
    duration = len(audio_data) / analyzer.sample_rate

    print(f"Duration: {duration:.1f} seconds")
    print("\n" + "=" * 60)
    print("🎵 SMART DANCE - Analyzing music...")
    print("=" * 60 + "\n")

    hop_size = analyzer.hop_size
    position = 0
    start_time = time.time()
    last_status_time = start_time

    try:
        while position < len(audio_data) - hop_size:
            chunk = audio_data[position:position + hop_size]

            is_beat, is_onset = analyzer.analyze_chunk(chunk)

            if is_beat:
                style = analyzer.detect_style()
                dancer.set_style(style)

                if dancer.on_beat():
                    bpm = analyzer.current_bpm
                    pct = 100 * position / len(audio_data)
                    print(f"🥁 Beat {dancer.beat_count} | BPM: {bpm:.0f} | Style: {dancer.current_style['name']} | {pct:.0f}%")

            if is_onset and analyzer.onset_detector.get_last() > 0.9:
                dancer.on_strong_onset()

            position += hop_size

            # Real-time sync
            elapsed = time.time() - start_time
            audio_time = position / analyzer.sample_rate
            if audio_time > elapsed:
                time.sleep(audio_time - elapsed)

            # Status update
            now = time.time()
            if now - last_status_time > 2:
                print(f"   [{analyzer.get_status()}]")
                last_status_time = now

    except KeyboardInterrupt:
        pass

    dancer.return_to_neutral()
    print(f"\n{'=' * 60}")
    print(f"Done! Total beats: {dancer.beat_count}")


def demo_styles(dancer):
    """Demo all dance styles."""

    print("\n" + "=" * 60)
    print("🎭 STYLE DEMO - Showing all dance styles")
    print("=" * 60)

    for style_name, style in ALL_STYLES.items():
        print(f"\n{style['name']}")
        print(f"  {style['description']}")
        print(f"  Move time: {style['move_time_ms']}ms")

        dancer.set_style(style_name)

        # Do one sequence
        sequence = style['sequences'][0]
        for move in sequence:
            print(f"    → {move}")
            dancer.execute_move(move)
            time.sleep(style['move_time_ms'] / 1000 + 0.3)

        time.sleep(0.5)

    dancer.return_to_neutral()
    print("\nDemo complete!")


def main():
    parser = argparse.ArgumentParser(description='Smart beat-sync dance with style detection')
    parser.add_argument('--live', action='store_true', help='Use live microphone')
    parser.add_argument('--audio', type=str, help='Audio file to process')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode')
    parser.add_argument('--demo', action='store_true', help='Demo all styles')
    parser.add_argument('--duration', type=int, help='Duration in seconds (live mode)')
    args = parser.parse_args()

    dancer = SmartDancer(simulate=args.simulate)
    analyzer = MusicAnalyzer()

    if args.demo:
        demo_styles(dancer)
    elif args.live:
        run_live(dancer, analyzer, duration=args.duration)
    elif args.audio:
        run_file(args.audio, dancer, analyzer)
    else:
        print("Usage:")
        print("  python smart_dance.py --live [--simulate]")
        print("  python smart_dance.py --audio song.wav [--simulate]")
        print("  python smart_dance.py --demo --simulate")


if __name__ == '__main__':
    main()
