#!/usr/bin/env python3
"""
Generate choreography from FACT dance output.

Analyzes FACT-generated dance data and converts it to a sequence of
pre-scripted .d6a action triggers based on feature matching.

Usage:
    python generate_choreography.py \
        --dance dance_house_tonypi.json \
        --library library.json \
        --output choreo.json

    # With visualization of matches
    python generate_choreography.py \
        --dance dance_house_tonypi.json \
        --library library.json \
        --output choreo.json \
        --verbose
"""

import argparse
import json
from pathlib import Path

from motion_features import extract_features, features_to_summary
from match_motion import find_best_action, find_top_matches, load_library, compute_similarity


def load_dance(dance_path):
    """Load FACT dance JSON file.

    Args:
        dance_path: Path to dance JSON file

    Returns:
        Dict with 'frames', 'fps', etc.
    """
    with open(dance_path, 'r') as f:
        return json.load(f)


def segment_by_action_duration(frames, library, fps=60, min_score=0.3):
    """Segment dance by trying each action's duration as segment length.

    For each position, tries all actions and picks the best match,
    then advances by that action's duration.

    Args:
        frames: List of frame dicts from FACT dance
        library: Action library dict
        fps: Frame rate
        min_score: Minimum match score to accept

    Returns:
        List of choreography entries
    """
    actions = library.get('actions', library)
    choreography = []
    position = 0
    n_frames = len(frames)

    # Pre-compute action durations in frames
    action_frames = {}
    for name, action in actions.items():
        duration_ms = action.get('duration_ms', 1000)
        action_frames[name] = max(1, int(duration_ms * fps / 1000))

    while position < n_frames:
        best_action = None
        best_score = 0
        best_duration = 60  # Default 1 second

        for name, action in actions.items():
            duration = action_frames[name]

            # Skip if segment would extend past end
            if position + duration > n_frames:
                continue

            # Extract segment
            segment = frames[position:position + duration]
            segment_features = extract_features(segment, fps=fps)

            # Compare to action features
            action_features = action.get('features', {})
            score = compute_similarity(segment_features, action_features)

            if score > best_score:
                best_score = score
                best_action = name
                best_duration = duration

        # Use the best match, or default to 'stand' for 1 second
        if best_action is None or best_score < min_score:
            best_action = 'stand'
            best_score = 0.0
            best_duration = fps  # 1 second

        time_ms = int(position * 1000 / fps)
        choreography.append({
            'time_ms': time_ms,
            'action': best_action,
            'score': round(best_score, 3),
            'duration_frames': best_duration,
        })

        position += best_duration

    return choreography


def segment_fixed_window(frames, library, fps=60, window_ms=1000, overlap=0.5):
    """Segment dance using fixed-size sliding windows.

    Simpler approach that uses consistent window sizes.

    Args:
        frames: List of frame dicts
        library: Action library dict
        fps: Frame rate
        window_ms: Window size in milliseconds
        overlap: Overlap ratio (0-1)

    Returns:
        List of choreography entries
    """
    window_frames = int(window_ms * fps / 1000)
    step_frames = int(window_frames * (1 - overlap))
    n_frames = len(frames)

    choreography = []
    position = 0

    while position < n_frames:
        end_pos = min(position + window_frames, n_frames)
        segment = frames[position:end_pos]

        segment_features = extract_features(segment, fps=fps)
        action, score = find_best_action(segment_features, library)

        time_ms = int(position * 1000 / fps)
        choreography.append({
            'time_ms': time_ms,
            'action': action,
            'score': round(score, 3),
        })

        position += step_frames

    return choreography


def segment_adaptive(frames, library, fps=60, min_window_ms=500, max_window_ms=3000):
    """Adaptive segmentation that finds natural motion boundaries.

    Detects motion changes and segments accordingly.

    Args:
        frames: List of frame dicts
        library: Action library dict
        fps: Frame rate
        min_window_ms: Minimum segment duration
        max_window_ms: Maximum segment duration

    Returns:
        List of choreography entries
    """
    import numpy as np
    from motion_features import frames_to_array

    data, servo_ids = frames_to_array(frames)
    n_frames = len(frames)

    # Compute velocity
    velocity = np.diff(data, axis=0)
    total_velocity = np.sum(np.abs(velocity), axis=1)

    # Smooth velocity
    kernel_size = max(1, fps // 10)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(total_velocity, kernel, mode='same')

    # Find local minima (motion boundaries)
    from scipy.signal import find_peaks
    min_distance = int(min_window_ms * fps / 1000)
    peaks, _ = find_peaks(-smoothed, distance=min_distance)

    # Add start and end
    boundaries = [0] + list(peaks) + [n_frames]

    # Merge segments that are too short or split those too long
    max_frames = int(max_window_ms * fps / 1000)
    min_frames = int(min_window_ms * fps / 1000)

    final_boundaries = [0]
    for i in range(1, len(boundaries)):
        seg_len = boundaries[i] - final_boundaries[-1]
        if seg_len > max_frames:
            # Split into chunks
            n_splits = int(np.ceil(seg_len / max_frames))
            chunk_size = seg_len // n_splits
            for j in range(1, n_splits):
                final_boundaries.append(final_boundaries[-1] + chunk_size)
            final_boundaries.append(boundaries[i])
        elif seg_len >= min_frames:
            final_boundaries.append(boundaries[i])
        # else: merge with previous (skip this boundary)

    if final_boundaries[-1] != n_frames:
        final_boundaries.append(n_frames)

    # Match each segment
    choreography = []
    for i in range(len(final_boundaries) - 1):
        start = final_boundaries[i]
        end = final_boundaries[i + 1]

        segment = frames[start:end]
        segment_features = extract_features(segment, fps=fps)
        action, score = find_best_action(segment_features, library)

        time_ms = int(start * 1000 / fps)
        choreography.append({
            'time_ms': time_ms,
            'action': action,
            'score': round(score, 3),
            'duration_frames': int(end - start),
        })

    return choreography


def consolidate_repeats(choreography, max_repeats=3):
    """Consolidate consecutive identical actions.

    Args:
        choreography: List of choreography entries
        max_repeats: Maximum times to repeat before forcing variety

    Returns:
        Consolidated choreography
    """
    if not choreography:
        return choreography

    consolidated = [choreography[0]]
    repeat_count = 1

    for entry in choreography[1:]:
        if entry['action'] == consolidated[-1]['action']:
            repeat_count += 1
            if repeat_count <= max_repeats:
                # Keep as separate entry (will repeat action)
                consolidated.append(entry)
        else:
            repeat_count = 1
            consolidated.append(entry)

    return consolidated


def generate_choreography(dance_path, library_path, method='adaptive', verbose=False, **kwargs):
    """Generate choreography from FACT dance.

    Args:
        dance_path: Path to FACT dance JSON
        library_path: Path to action library JSON
        method: 'adaptive', 'action_duration', or 'fixed_window'
        verbose: Print progress info
        **kwargs: Additional arguments for segmentation method

    Returns:
        Choreography dict ready to save
    """
    # Load data
    if verbose:
        print(f"Loading dance from {dance_path}...")
    dance = load_dance(dance_path)
    frames = dance['frames']
    fps = dance.get('fps', 60)

    if verbose:
        print(f"Loading library from {library_path}...")
    library = load_library(library_path)

    if verbose:
        print(f"Dance: {len(frames)} frames at {fps} FPS ({len(frames)/fps:.1f}s)")
        print(f"Library: {library.get('n_actions', 0)} actions")
        print(f"Method: {method}")

    # Segment and match
    if method == 'action_duration':
        choreography = segment_by_action_duration(frames, library, fps=fps, **kwargs)
    elif method == 'fixed_window':
        choreography = segment_fixed_window(frames, library, fps=fps, **kwargs)
    elif method == 'adaptive':
        choreography = segment_adaptive(frames, library, fps=fps, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Consolidate
    choreography = consolidate_repeats(choreography)

    if verbose:
        print(f"\nGenerated {len(choreography)} action triggers:")
        for entry in choreography[:20]:
            time_s = entry['time_ms'] / 1000
            print(f"  {time_s:6.1f}s: {entry['action']:25s} (score: {entry['score']:.2f})")
        if len(choreography) > 20:
            print(f"  ... and {len(choreography)-20} more")

    # Build output
    total_duration = len(frames) * 1000 / fps
    return {
        'source': str(dance_path),
        'library': str(library_path),
        'method': method,
        'fps': fps,
        'total_duration_ms': int(total_duration),
        'n_triggers': len(choreography),
        'choreography': choreography,
    }


def print_choreography_stats(choreo_data):
    """Print statistics about generated choreography."""
    choreography = choreo_data['choreography']

    # Count action usage
    action_counts = {}
    for entry in choreography:
        action = entry['action']
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"\nChoreography Statistics:")
    print(f"  Total triggers: {len(choreography)}")
    print(f"  Unique actions: {len(action_counts)}")
    print(f"  Duration: {choreo_data['total_duration_ms']/1000:.1f}s")

    avg_score = sum(e['score'] for e in choreography) / len(choreography) if choreography else 0
    print(f"  Average match score: {avg_score:.2f}")

    print(f"\nMost used actions:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    for action, count in sorted_actions[:10]:
        print(f"  {action:25s} {count:3d} times")


def main():
    parser = argparse.ArgumentParser(
        description="Generate choreography from FACT dance output"
    )
    parser.add_argument("--dance", "-d", type=str, required=True,
                        help="Input FACT dance JSON file")
    parser.add_argument("--library", "-l", type=str, default="library.json",
                        help="Action library JSON file")
    parser.add_argument("--output", "-o", type=str, default="choreo.json",
                        help="Output choreography JSON file")
    parser.add_argument("--method", "-m", type=str, default="adaptive",
                        choices=['adaptive', 'action_duration', 'fixed_window'],
                        help="Segmentation method")
    parser.add_argument("--window-ms", type=int, default=1000,
                        help="Window size for fixed_window method (ms)")
    parser.add_argument("--min-window-ms", type=int, default=500,
                        help="Minimum segment size for adaptive method (ms)")
    parser.add_argument("--max-window-ms", type=int, default=3000,
                        help="Maximum segment size for adaptive method (ms)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--stats", "-s", action="store_true",
                        help="Print choreography statistics")

    args = parser.parse_args()

    # Generate choreography
    kwargs = {}
    if args.method == 'fixed_window':
        kwargs['window_ms'] = args.window_ms
    elif args.method == 'adaptive':
        kwargs['min_window_ms'] = args.min_window_ms
        kwargs['max_window_ms'] = args.max_window_ms

    choreo_data = generate_choreography(
        args.dance,
        args.library,
        method=args.method,
        verbose=args.verbose,
        **kwargs
    )

    # Save
    with open(args.output, 'w') as f:
        json.dump(choreo_data, f, indent=2)

    print(f"\nSaved choreography to {args.output}")

    if args.stats or args.verbose:
        print_choreography_stats(choreo_data)


if __name__ == "__main__":
    main()
