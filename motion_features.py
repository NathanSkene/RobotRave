#!/usr/bin/env python3
"""
Feature extraction for motion matching.

Extracts high-level motion features from servo data to enable
semantic matching between FACT-generated dances and .d6a actions.

Features extracted:
- Body part activity (which parts move: arms, legs, etc.)
- Motion type (oscillating, single gesture, cyclic, static)
- Periodicity (is it repetitive? at what frequency?)
- Symmetry (left-right relationship)
- Energy/velocity (how much motion, how fast)
- Complexity (how many joints moving together)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft


# Servo groupings by body part
SERVO_GROUPS = {
    'left_arm': ['6', '7', '8'],      # elbow, shoulder_roll, shoulder_pitch
    'right_arm': ['11', '15', '16'],  # elbow, shoulder_roll, shoulder_pitch
    'left_leg': ['1', '2', '3', '4', '5'],  # ankle_roll, ankle_pitch, knee, hip_pitch, hip_roll
    'right_leg': ['9', '10', '14', '12', '13'],  # ankle_roll, ankle_pitch, knee, hip_pitch, hip_roll
}

# Symmetric pairs (left, right)
SYMMETRIC_PAIRS = [
    ('6', '11'),   # elbows
    ('7', '15'),   # shoulder_roll
    ('8', '16'),   # shoulder_pitch
    ('1', '9'),    # ankle_roll
    ('2', '10'),   # ankle_pitch
    ('3', '14'),   # knee
    ('4', '12'),   # hip_pitch
    ('5', '13'),   # hip_roll
]


def frames_to_array(frames, servo_ids=None):
    """Convert frames list to numpy array.

    Args:
        frames: List of frame dicts with 'servos' key
        servo_ids: Optional list of servo IDs to include (default: 1-16)

    Returns:
        np.array of shape (n_frames, n_servos)
        List of servo IDs in column order
    """
    if servo_ids is None:
        servo_ids = [str(i) for i in range(1, 17)]

    # Filter to only servos that exist in frames
    available = set(frames[0]['servos'].keys()) if frames else set()
    servo_ids = [s for s in servo_ids if s in available]

    data = np.zeros((len(frames), len(servo_ids)))
    for i, frame in enumerate(frames):
        for j, sid in enumerate(servo_ids):
            data[i, j] = frame['servos'].get(sid, 500)  # Default to neutral

    return data, servo_ids


def compute_activity_mask(data, servo_ids, threshold=20):
    """Compute which body parts are active (moving).

    Args:
        data: Servo data array (n_frames, n_servos)
        servo_ids: List of servo ID strings
        threshold: Minimum movement range to consider "active"

    Returns:
        Dict of {group_name: bool} indicating activity
    """
    # Calculate movement range per servo
    servo_range = {}
    for j, sid in enumerate(servo_ids):
        col = data[:, j]
        servo_range[sid] = np.ptp(col)  # peak-to-peak

    # Determine activity per group
    activity = {}
    for group, servos in SERVO_GROUPS.items():
        group_ranges = [servo_range.get(s, 0) for s in servos]
        activity[group] = bool(max(group_ranges) > threshold)

    # Aggregate
    activity['arms'] = bool(activity['left_arm'] or activity['right_arm'])
    activity['legs'] = bool(activity['left_leg'] or activity['right_leg'])
    activity['both'] = bool(activity['arms'] and activity['legs'])

    return activity


def classify_motion_type(data, fps=60):
    """Classify the type of motion.

    Returns:
        'static' - minimal movement
        'single_gesture' - one-shot movement (like a kick or punch)
        'oscillating' - back-and-forth movement (like waving)
        'cyclic' - repeating pattern (like walking)
    """
    if data.shape[0] < 10:
        return 'static'

    # Calculate total movement
    total_movement = np.sum(np.abs(np.diff(data, axis=0)))
    avg_per_frame = total_movement / (data.shape[0] * data.shape[1])

    if avg_per_frame < 2:
        return 'static'

    # Analyze frequency content of most active servo
    ranges = np.ptp(data, axis=0)
    most_active = np.argmax(ranges)
    active_col = data[:, most_active]

    # Detrend and compute FFT
    detrended = active_col - np.mean(active_col)
    if np.std(detrended) < 1:
        return 'static'

    n = len(detrended)
    freqs = np.fft.fftfreq(n, 1/fps)
    fft_vals = np.abs(fft(detrended))[:n//2]
    freqs = freqs[:n//2]

    # Find dominant frequency (excluding DC)
    fft_vals[0] = 0  # ignore DC
    if len(fft_vals) < 2:
        return 'single_gesture'

    peak_idx = np.argmax(fft_vals)
    peak_freq = freqs[peak_idx]

    # Check if there's significant periodic content
    peak_power = fft_vals[peak_idx]
    total_power = np.sum(fft_vals)

    if total_power < 1:
        return 'single_gesture'

    periodicity_ratio = peak_power / total_power

    # Detect zero crossings in velocity
    velocity = np.diff(active_col)
    sign_changes = np.sum(np.abs(np.diff(np.sign(velocity))) > 0)

    # Classification logic
    duration_s = data.shape[0] / fps

    if periodicity_ratio > 0.3 and peak_freq > 0.3:
        # Check if it's walking-like (alternating legs) vs oscillating
        if 0.5 < peak_freq < 2.5 and sign_changes > 4:
            return 'cyclic'
        elif sign_changes >= 3:
            return 'oscillating'

    if sign_changes <= 2 and duration_s < 2:
        return 'single_gesture'

    if sign_changes >= 4:
        return 'oscillating'

    return 'single_gesture'


def detect_periodicity(data, fps=60):
    """Detect if motion is periodic and estimate frequency.

    Returns:
        (is_periodic: bool, frequency: float in Hz)
    """
    if data.shape[0] < 20:
        return False, 0.0

    # Use most active servo
    ranges = np.ptp(data, axis=0)
    most_active = np.argmax(ranges)
    active_col = data[:, most_active]

    if np.std(active_col) < 5:
        return False, 0.0

    # Compute autocorrelation
    detrended = active_col - np.mean(active_col)
    autocorr = np.correlate(detrended, detrended, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find peaks in autocorrelation (excluding lag 0)
    min_lag = int(fps * 0.15)  # Minimum period ~150ms
    max_lag = min(len(autocorr) - 1, int(fps * 3))  # Maximum period 3s

    if max_lag <= min_lag:
        return False, 0.0

    # Find local maxima
    peaks, properties = signal.find_peaks(autocorr[min_lag:max_lag], height=0.3)

    if len(peaks) == 0:
        return False, 0.0

    # Get the first significant peak (fundamental period)
    peak_lag = peaks[0] + min_lag
    frequency = float(fps / peak_lag)

    return True, frequency


def compute_symmetry(data, servo_ids):
    """Analyze left-right symmetry of motion.

    Returns:
        'mirror' - left and right move in opposite directions
        'parallel' - left and right move together
        'one_sided' - only one side moves
        'asymmetric' - uncorrelated movement
    """
    symmetry_scores = []

    for left_id, right_id in SYMMETRIC_PAIRS:
        if left_id not in servo_ids or right_id not in servo_ids:
            continue

        left_idx = servo_ids.index(left_id)
        right_idx = servo_ids.index(right_id)

        left_col = data[:, left_idx]
        right_col = data[:, right_idx]

        left_range = np.ptp(left_col)
        right_range = np.ptp(right_col)

        # Skip if neither side moves
        if left_range < 10 and right_range < 10:
            continue

        # Check for one-sided movement
        if left_range < 10:
            symmetry_scores.append(('one_sided', right_range))
            continue
        if right_range < 10:
            symmetry_scores.append(('one_sided', left_range))
            continue

        # Calculate correlation
        left_norm = (left_col - np.mean(left_col)) / (np.std(left_col) + 1e-6)
        right_norm = (right_col - np.mean(right_col)) / (np.std(right_col) + 1e-6)
        correlation = np.mean(left_norm * right_norm)

        weight = max(left_range, right_range)

        if correlation > 0.6:
            symmetry_scores.append(('parallel', weight))
        elif correlation < -0.6:
            symmetry_scores.append(('mirror', weight))
        else:
            symmetry_scores.append(('asymmetric', weight))

    if not symmetry_scores:
        return 'static'

    # Weighted voting
    type_weights = {}
    for sym_type, weight in symmetry_scores:
        type_weights[sym_type] = type_weights.get(sym_type, 0) + weight

    return max(type_weights, key=type_weights.get)


def compute_energy_features(data, fps=60):
    """Compute energy and velocity features.

    Returns:
        dict with total_displacement, avg_velocity, peak_velocity
    """
    if data.shape[0] < 2:
        return {
            'total_displacement': 0,
            'avg_velocity': 0,
            'peak_velocity': 0,
        }

    # Velocity = frame-to-frame differences
    velocity = np.diff(data, axis=0)

    # Total displacement across all servos
    total_displacement = np.sum(np.abs(velocity))

    # Duration in seconds
    duration = data.shape[0] / fps

    # Average velocity (per servo per second)
    avg_velocity = total_displacement / (data.shape[1] * duration) if duration > 0 else 0

    # Peak velocity (max single-frame movement for any servo)
    peak_velocity = np.max(np.abs(velocity)) if velocity.size > 0 else 0

    return {
        'total_displacement': float(total_displacement),
        'avg_velocity': float(avg_velocity),
        'peak_velocity': float(peak_velocity),
    }


def count_active_servos(data, threshold=20):
    """Count how many servos are actively moving.

    Args:
        data: Servo data array
        threshold: Minimum range to consider active

    Returns:
        Number of active servos (0-16)
    """
    ranges = np.ptp(data, axis=0)
    return int(np.sum(ranges > threshold))


def extract_features(frames, fps=60):
    """Extract feature vector from motion (FACT or .d6a).

    Args:
        frames: List of frame dicts with 'servos' key
        fps: Frame rate (default 60)

    Returns:
        Dict of motion features
    """
    if not frames:
        return {
            'activity': {},
            'motion_type': 'static',
            'is_periodic': False,
            'frequency': 0.0,
            'symmetry_type': 'static',
            'total_displacement': 0,
            'avg_velocity': 0,
            'peak_velocity': 0,
            'active_dof': 0,
            'n_frames': 0,
            'duration_ms': 0,
        }

    # Convert to numpy array
    data, servo_ids = frames_to_array(frames)

    # Calculate duration
    duration_ms = sum(f.get('time_ms', int(1000/fps)) for f in frames)

    # Extract all features
    activity = compute_activity_mask(data, servo_ids)
    motion_type = classify_motion_type(data, fps)
    is_periodic, frequency = detect_periodicity(data, fps)
    symmetry_type = compute_symmetry(data, servo_ids)
    energy = compute_energy_features(data, fps)
    active_dof = count_active_servos(data)

    return {
        'activity': activity,
        'motion_type': motion_type,
        'is_periodic': is_periodic,
        'frequency': frequency,
        'symmetry_type': symmetry_type,
        'total_displacement': energy['total_displacement'],
        'avg_velocity': energy['avg_velocity'],
        'peak_velocity': energy['peak_velocity'],
        'active_dof': active_dof,
        'n_frames': len(frames),
        'duration_ms': duration_ms,
    }


def features_to_summary(features):
    """Generate human-readable summary of features."""
    lines = []

    # Activity
    activity = features['activity']
    active_parts = []
    if activity.get('left_arm'):
        active_parts.append('L-arm')
    if activity.get('right_arm'):
        active_parts.append('R-arm')
    if activity.get('left_leg'):
        active_parts.append('L-leg')
    if activity.get('right_leg'):
        active_parts.append('R-leg')
    lines.append(f"Active: {', '.join(active_parts) if active_parts else 'none'}")

    # Motion type
    lines.append(f"Type: {features['motion_type']}")

    # Periodicity
    if features['is_periodic']:
        lines.append(f"Periodic: {features['frequency']:.2f} Hz")
    else:
        lines.append("Periodic: no")

    # Symmetry
    lines.append(f"Symmetry: {features['symmetry_type']}")

    # Energy
    lines.append(f"Velocity: {features['avg_velocity']:.1f} avg, {features['peak_velocity']:.0f} peak")

    # Complexity
    lines.append(f"Active DOF: {features['active_dof']}/16")

    return '\n'.join(lines)


if __name__ == '__main__':
    # Test with a sample .d6a file
    import sys
    from read_d6a import read_d6a

    if len(sys.argv) < 2:
        print("Usage: python motion_features.py <action.d6a>")
        sys.exit(1)

    filepath = sys.argv[1]
    frames = read_d6a(filepath)

    print(f"File: {filepath}")
    print(f"Frames: {len(frames)}")
    print()

    features = extract_features(frames)
    print(features_to_summary(features))
