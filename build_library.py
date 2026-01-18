#!/usr/bin/env python3
"""
Build action feature library from .d6a files.

Pre-computes motion features for all available actions, enabling fast
matching during choreography generation.

Usage:
    python build_library.py --input ./action_groups --output library.json
    python build_library.py --input ./action_groups --output library.json --verbose
"""

import argparse
import json
from pathlib import Path

from read_d6a import read_d6a
from motion_features import extract_features, features_to_summary


def build_action_library(d6a_dir, verbose=False):
    """Build feature library from all .d6a files in directory.

    Args:
        d6a_dir: Path to directory containing .d6a files
        verbose: Print progress and feature summaries

    Returns:
        Dict mapping action names to their features and metadata
    """
    d6a_dir = Path(d6a_dir)
    d6a_files = sorted(d6a_dir.glob("*.d6a"))

    if not d6a_files:
        print(f"No .d6a files found in {d6a_dir}")
        return {}

    library = {}
    errors = []

    for i, filepath in enumerate(d6a_files):
        name = filepath.stem  # filename without extension

        if verbose:
            print(f"[{i+1}/{len(d6a_files)}] Processing {name}...", end='')

        try:
            frames = read_d6a(filepath)
            features = extract_features(frames)

            # Calculate total duration
            duration_ms = sum(f['time_ms'] for f in frames)

            library[name] = {
                'features': features,
                'duration_ms': duration_ms,
                'n_frames': len(frames),
                'source': str(filepath),
            }

            if verbose:
                print(f" {len(frames)} frames, {duration_ms}ms")
                if features['activity'].get('arms') or features['activity'].get('legs'):
                    active = []
                    if features['activity'].get('arms'):
                        active.append('arms')
                    if features['activity'].get('legs'):
                        active.append('legs')
                    print(f"    {features['motion_type']}, {'+'.join(active)}, "
                          f"{features['symmetry_type']}")

        except Exception as e:
            errors.append((name, str(e)))
            if verbose:
                print(f" ERROR: {e}")

    if errors:
        print(f"\nWarning: {len(errors)} files failed to process:")
        for name, err in errors:
            print(f"  {name}: {err}")

    return library


def categorize_actions(library):
    """Categorize actions by their features for easier browsing.

    Returns dict of category -> list of action names
    """
    categories = {
        'static': [],
        'arm_movement': [],
        'leg_movement': [],
        'full_body': [],
        'walking': [],
        'gestures': [],
    }

    for name, action in library.items():
        features = action['features']
        activity = features.get('activity', {})

        # Categorize
        if features['motion_type'] == 'static':
            categories['static'].append(name)
        elif features['motion_type'] == 'cyclic' and activity.get('legs'):
            categories['walking'].append(name)
        elif activity.get('both'):
            categories['full_body'].append(name)
        elif activity.get('arms') and not activity.get('legs'):
            categories['arm_movement'].append(name)
        elif activity.get('legs') and not activity.get('arms'):
            categories['leg_movement'].append(name)
        else:
            categories['gestures'].append(name)

    return categories


def print_library_summary(library):
    """Print summary of action library."""
    print(f"\nLibrary Summary: {len(library)} actions")
    print("=" * 50)

    # Group by motion type
    by_type = {}
    for name, action in library.items():
        motion_type = action['features']['motion_type']
        by_type.setdefault(motion_type, []).append(name)

    for motion_type in ['static', 'single_gesture', 'oscillating', 'cyclic']:
        if motion_type in by_type:
            actions = by_type[motion_type]
            print(f"\n{motion_type.upper()} ({len(actions)}):")
            for name in sorted(actions)[:10]:
                action = library[name]
                duration = action['duration_ms']
                print(f"  {name:25s} {duration:5d}ms")
            if len(actions) > 10:
                print(f"  ... and {len(actions)-10} more")

    # Categorize
    categories = categorize_actions(library)
    print("\n" + "=" * 50)
    print("By Category:")
    for cat, names in categories.items():
        if names:
            print(f"  {cat}: {len(names)} actions")


def main():
    parser = argparse.ArgumentParser(
        description="Build action feature library from .d6a files"
    )
    parser.add_argument("--input", "-i", type=str, default="./action_groups",
                        help="Directory containing .d6a files")
    parser.add_argument("--output", "-o", type=str, default="library.json",
                        help="Output JSON file for library")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print library summary after building")

    args = parser.parse_args()

    print(f"Building action library from {args.input}...")
    library = build_action_library(args.input, verbose=args.verbose)

    if not library:
        print("No actions found!")
        return

    # Add metadata
    output = {
        'n_actions': len(library),
        'source_dir': args.input,
        'categories': categorize_actions(library),
        'actions': library,
    }

    # Save
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved library with {len(library)} actions to {args.output}")

    if args.summary or not args.verbose:
        print_library_summary(library)


if __name__ == "__main__":
    main()
