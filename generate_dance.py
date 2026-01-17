#!/usr/bin/env python3
"""
Generate dance motion from audio using FACT model.
Usage: python generate_dance.py --audio your_song.mp3 --output dance.npy
"""

import os
import sys
import argparse
import gc
import subprocess
import numpy as np
import tensorflow as tf

# Enable GPU memory growth BEFORE any TF operations
# This prevents TF from allocating all GPU memory upfront
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Enabled memory growth for {len(gpus)} GPU(s)")


def check_gpu_memory(min_free_gb=10, auto_kill=False):
    """Check GPU memory and optionally kill other processes.

    Args:
        min_free_gb: Minimum free GPU memory required (in GB)
        auto_kill: If True, automatically kill other GPU processes

    Returns:
        True if enough memory is available, False otherwise
    """
    try:
        # Get GPU memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Warning: Could not query GPU memory")
            return True

        total, free, used = map(int, result.stdout.strip().split(','))
        free_gb = free / 1024
        used_gb = used / 1024
        total_gb = total / 1024

        print(f"GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_gb:.1f}GB used)")

        if free_gb >= min_free_gb:
            return True

        # Not enough memory - check what's using it
        print(f"Warning: Only {free_gb:.1f}GB free, need {min_free_gb}GB")

        # Get processes using GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )

        if result.stdout.strip():
            print("Processes using GPU:")
            current_pid = os.getpid()
            other_pids = []

            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 3:
                    pid, mem, name = int(parts[0].strip()), parts[1].strip(), parts[2].strip()
                    marker = " (this process)" if pid == current_pid else ""
                    print(f"  PID {pid}: {mem}MB - {name}{marker}")
                    if pid != current_pid:
                        other_pids.append(pid)

            if other_pids and auto_kill:
                print(f"Auto-killing {len(other_pids)} other GPU process(es)...")
                for pid in other_pids:
                    try:
                        os.kill(pid, 9)
                        print(f"  Killed PID {pid}")
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        print(f"  Cannot kill PID {pid} (permission denied)")

                # Wait and recheck
                import time
                time.sleep(2)
                return check_gpu_memory(min_free_gb, auto_kill=False)
            elif other_pids:
                print(f"Run with --clear-gpu to auto-kill other processes, or manually: kill {' '.join(map(str, other_pids))}")
                return False

        return free_gb >= min_free_gb

    except FileNotFoundError:
        print("Warning: nvidia-smi not found, skipping GPU memory check")
        return True

import soundfile as sf
from scipy import signal
from scipy.fftpack import dct

# Add mint to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mint'))

from mint.utils import config_util
from mint.core import model_builder


def mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """Create mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)

    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    filterbank = np.zeros((n_mels, len(fft_freqs)))

    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]

        for j, freq in enumerate(fft_freqs):
            if lower <= freq <= center:
                filterbank[i, j] = (freq - lower) / (center - lower)
            elif center <= freq <= upper:
                filterbank[i, j] = (upper - freq) / (upper - center)

    return filterbank


def extract_audio_features_from_array(audio_data, sample_rate, target_fps=60):
    """Extract 35-dim audio features from audio array.

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        target_fps: Target frame rate (default 60)

    Returns:
        [n_frames, 35] audio features
    """
    HOP_LENGTH = 512
    N_FFT = 2048
    SR = target_fps * HOP_LENGTH  # 30720 Hz

    data = audio_data.copy()

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sample_rate != SR:
        print(f"Resampling from {sample_rate} to {SR}...")
        num_samples = int(len(data) * SR / sample_rate)
        data = signal.resample(data, num_samples)

    data = data.astype(np.float32)

    return _extract_features_from_samples(data, target_fps)


def extract_audio_features(audio_path, target_fps=60):
    """Extract 35-dim audio features from audio file (without librosa/numba)."""
    HOP_LENGTH = 512
    N_FFT = 2048
    SR = target_fps * HOP_LENGTH  # 30720 Hz

    print(f"Loading audio: {audio_path}")
    data, orig_sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if orig_sr != SR:
        print(f"Resampling from {orig_sr} to {SR}...")
        num_samples = int(len(data) * SR / orig_sr)
        data = signal.resample(data, num_samples)

    data = data.astype(np.float32)

    return _extract_features_from_samples(data, target_fps)


def _extract_features_from_samples(data, target_fps=60):
    """Internal function to extract features from preprocessed audio samples."""
    HOP_LENGTH = 512
    N_FFT = 2048
    SR = target_fps * HOP_LENGTH  # 30720 Hz

    print("Extracting features...")
    n_frames = 1 + (len(data) - N_FFT) // HOP_LENGTH

    # Compute STFT magnitude
    stft_mag = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    window = signal.windows.hann(N_FFT)

    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = data[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        stft_mag[i] = np.abs(spectrum)

    # Onset envelope (spectral flux)
    diff = np.diff(stft_mag, axis=0, prepend=stft_mag[0:1])
    envelope = np.sum(np.maximum(diff, 0), axis=1)
    envelope = envelope / (np.max(envelope) + 1e-8)  # Normalize

    # MFCC (20 dims)
    mel_fb = mel_filterbank(SR, N_FFT, n_mels=128)
    mel_spec = np.dot(stft_mag, mel_fb.T)
    mel_spec = np.log(mel_spec + 1e-8)
    mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, :20]

    # Chroma (12 dims) - simplified version
    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    for i, freq in enumerate(fft_freqs):
        if freq > 0:
            pitch_class = int(round(12 * np.log2(freq / 440) + 69)) % 12
            chroma[:, pitch_class] += stft_mag[:, i]
    chroma = chroma / (np.max(chroma) + 1e-8)

    # Peak detection (onset)
    peak_threshold = np.mean(envelope) + 0.5 * np.std(envelope)
    peaks = signal.find_peaks(envelope, height=peak_threshold, distance=int(SR / HOP_LENGTH / 8))[0]
    peak_onehot = np.zeros(n_frames, dtype=np.float32)
    peak_onehot[peaks] = 1.0

    # Beat detection (simple autocorrelation-based)
    # Look for periodicity around 120 BPM (2 Hz)
    beat_period = int(SR / HOP_LENGTH / 2)  # ~30 frames for 120 BPM
    beat_onehot = np.zeros(n_frames, dtype=np.float32)
    for i in range(beat_period, n_frames - beat_period):
        if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]:
            if envelope[i] > peak_threshold * 0.5:
                beat_onehot[i] = 1.0

    # Concatenate: 1 + 20 + 12 + 1 + 1 = 35 dims
    audio_features = np.concatenate([
        envelope[:, None],
        mfcc,
        chroma,
        peak_onehot[:, None],
        beat_onehot[:, None]
    ], axis=-1)

    print(f"Audio features shape: {audio_features.shape} (frames, 35)")

    return audio_features.astype(np.float32)


def create_neutral_motion(n_frames=120):
    """Create neutral standing pose for seed motion.

    Motion format (225 dims):
    - [0:6]   → Padding (zeros)
    - [6:9]   → Translation (x, y, z)
    - [9:225] → 24 joints × 9 rotation matrix values = 216 dims
    """
    motion = np.zeros((n_frames, 225), dtype=np.float32)

    # Set identity rotation matrices for each joint (I_3x3 flattened = [1,0,0,0,1,0,0,0,1])
    identity_rotmat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)

    for frame in range(n_frames):
        # Padding at [0:6] - already zeros
        # Translation at [6:9]
        motion[frame, 6:9] = [0, 0, 0]
        # 24 identity rotation matrices starting at index 9
        for joint in range(24):
            start_idx = 9 + joint * 9
            motion[frame, start_idx:start_idx + 9] = identity_rotmat

    return motion


def load_model(config_path, checkpoint_dir):
    """Load FACT model with pre-trained weights."""
    print("Loading model configuration...")
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']

    print("Building model...")
    model = model_builder.build(model_config, is_training=False)
    model.global_step = tf.Variable(initial_value=0, dtype=tf.int64)

    print("Loading checkpoint...")
    checkpoint = tf.train.Checkpoint(model=model, global_step=model.global_step)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    print(f"Loaded checkpoint: {checkpoint_manager.latest_checkpoint}")

    return model


def generate_dance(model, audio_features, seed_motion, steps=600):
    """Generate dance motion from audio using FACT model.

    The model uses infer_auto_regressive() which:
    - Takes FULL audio sequence upfront (model slides 240-frame window internally)
    - Maintains 120-frame motion buffer (rolling, not full history)
    - Each step: 360 total frames (120 motion + 240 audio)

    DO NOT chunk by restarting - this breaks the sliding window mechanism!

    Args:
        model: FACT model
        audio_features: [seq_len, 35] audio features (must be >= steps + 240)
        seed_motion: [120, 225] seed motion (2 seconds)
        steps: number of frames to generate (60fps, so 600 = 10 seconds)

    Returns:
        Generated motion [steps, 225]
    """
    print(f"Generating {steps} frames of dance ({steps/60:.1f} seconds)...")

    # Prepare inputs - pass FULL audio sequence
    motion_input = tf.constant(seed_motion[np.newaxis, :, :], dtype=tf.float32)
    audio_input = tf.constant(audio_features[np.newaxis, :, :], dtype=tf.float32)

    inputs = {
        "motion_input": motion_input,
        "audio_input": audio_input
    }

    # Single call - model handles the autoregressive loop internally
    outputs = model.infer_auto_regressive(inputs, steps=steps)
    motion = outputs[0].numpy()

    # Clean up to free GPU memory
    del outputs, inputs, motion_input, audio_input
    gc.collect()
    tf.keras.backend.clear_session()

    print(f"  Generated {len(motion)} frames")
    return motion


def motion_to_joint_angles(motion):
    """Convert motion (rotation matrices) to euler angles.

    Args:
        motion: [n_frames, 225] - padding (6) + translation (3) + 24 joints × 9 rotation matrix

    Returns:
        joint_angles: [n_frames, 24, 3] euler angles in radians
    """
    from scipy.spatial.transform import Rotation

    n_frames = motion.shape[0]
    joint_angles = np.zeros((n_frames, 24, 3), dtype=np.float32)

    for frame in range(n_frames):
        for joint in range(24):
            start_idx = 9 + joint * 9  # Rotation matrices start at index 9
            rotmat = motion[frame, start_idx:start_idx + 9].reshape(3, 3)

            # Convert rotation matrix to euler angles (XYZ order)
            try:
                r = Rotation.from_matrix(rotmat)
                joint_angles[frame, joint] = r.as_euler('xyz', degrees=False)
            except:
                joint_angles[frame, joint] = [0, 0, 0]

    return joint_angles


def main():
    parser = argparse.ArgumentParser(description='Generate dance from audio using FACT model')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output', type=str, default='dance_output.npy', help='Output path for motion')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration to generate (seconds)')
    parser.add_argument('--config', type=str, default='mint/configs/fact_v5_deeper_t10_cm12.config')
    parser.add_argument('--checkpoint', type=str, default='mint/checkpoints')
    parser.add_argument('--no-clear-gpu', action='store_true', help='Do NOT auto-kill other GPU processes')
    parser.add_argument('--min-gpu-memory', type=float, default=10.0, help='Minimum free GPU memory in GB (default: 10)')
    args = parser.parse_args()

    # Check GPU memory before loading model (auto-kill other processes by default)
    auto_kill = not args.no_clear_gpu
    if not check_gpu_memory(min_free_gb=args.min_gpu_memory, auto_kill=auto_kill):
        print("Error: Not enough GPU memory available.")
        sys.exit(1)

    # Extract audio features
    audio_features = extract_audio_features(args.audio)

    # Create seed motion (neutral pose)
    seed_motion = create_neutral_motion(n_frames=120)

    # Load model
    model = load_model(args.config, args.checkpoint)

    # Calculate steps (60fps)
    steps = int(args.duration * 60)

    # Check if we have enough audio
    audio_seq_length = 240  # Model's audio sequence length
    max_steps = audio_features.shape[0] - audio_seq_length
    if steps > max_steps:
        print(f"Warning: Audio too short. Generating {max_steps} frames instead of {steps}")
        steps = max_steps

    # Generate dance
    generated_motion = generate_dance(model, audio_features, seed_motion, steps=steps)

    # Save raw motion
    np.save(args.output, generated_motion)
    print(f"Saved motion to {args.output}")

    # Also save as joint angles
    joint_angles = motion_to_joint_angles(generated_motion)
    angles_path = args.output.replace('.npy', '_angles.npy')
    np.save(angles_path, joint_angles)
    print(f"Saved joint angles to {angles_path}")

    print(f"\nGenerated {generated_motion.shape[0]} frames ({generated_motion.shape[0]/60:.1f} seconds)")
    print(f"Motion shape: {generated_motion.shape}")
    print(f"Joint angles shape: {joint_angles.shape}")

    # Print some stats
    print(f"\nJoint angle ranges (radians):")
    for joint_idx, joint_name in enumerate(['root', 'l_hip', 'r_hip', 'belly', 'l_knee', 'r_knee',
                                            'spine', 'l_ankle', 'r_ankle', 'chest', 'l_toes', 'r_toes',
                                            'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
                                            'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']):
        angles = joint_angles[:, joint_idx, :]
        print(f"  {joint_name}: x=[{angles[:,0].min():.2f}, {angles[:,0].max():.2f}] "
              f"y=[{angles[:,1].min():.2f}, {angles[:,1].max():.2f}] "
              f"z=[{angles[:,2].min():.2f}, {angles[:,2].max():.2f}]")


if __name__ == '__main__':
    main()
