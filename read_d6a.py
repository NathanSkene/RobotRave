#!/usr/bin/env python3
"""
Utility to read and convert TonyPi's .d6a action files.

.d6a files are SQLite3 databases with an ActionGroup table containing:
- index: frame number
- time_ms: duration to hold this pose (milliseconds)
- servo1-16: pulse values for each servo (0-1000 range)

Usage:
    python read_d6a.py wave.d6a                    # Print frame info
    python read_d6a.py wave.d6a -o wave.json       # Convert to JSON
    python read_d6a.py --list                      # List available actions
"""

import sqlite3
import json
import argparse
from pathlib import Path


def read_d6a(filepath):
    """Read a .d6a action file and return frames.

    Args:
        filepath: Path to .d6a file

    Returns:
        List of frame dicts with 'time_ms' and 'servos' keys
    """
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()
    cursor.execute("select * from ActionGroup")

    frames = []
    for row in cursor.fetchall():
        frame = {
            'time_ms': row[1],
            'servos': {str(i+1): row[2+i] for i in range(16)}
        }
        frames.append(frame)

    conn.close()
    return frames


def d6a_to_json(d6a_path, output_path=None):
    """Convert .d6a to JSON format compatible with play_dance.py.

    Args:
        d6a_path: Path to .d6a file
        output_path: Optional path for JSON output

    Returns:
        Dict with fps, n_frames, and frames list
    """
    frames = read_d6a(d6a_path)

    # Calculate effective FPS from average time between frames
    # Note: .d6a files use variable timing per frame, not fixed FPS
    if len(frames) > 1:
        avg_time = sum(f['time_ms'] for f in frames) / len(frames)
        fps = 1000.0 / avg_time if avg_time > 0 else 60
    else:
        fps = 60

    # Total duration
    total_ms = sum(f['time_ms'] for f in frames)

    output = {
        'fps': round(fps, 2),
        'n_frames': len(frames),
        'total_duration_ms': total_ms,
        'source': str(d6a_path),
        'frames': frames
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_path}")

    return output


def list_actions(directory):
    """List all .d6a files in a directory."""
    path = Path(directory)
    d6a_files = sorted(path.glob("*.d6a"))

    if not d6a_files:
        print(f"No .d6a files found in {directory}")
        return

    print(f"Found {len(d6a_files)} action files in {directory}:\n")

    for f in d6a_files:
        try:
            frames = read_d6a(f)
            total_ms = sum(frame['time_ms'] for frame in frames)
            print(f"  {f.stem:25s} {len(frames):3d} frames, {total_ms/1000:.1f}s")
        except Exception as e:
            print(f"  {f.stem:25s} ERROR: {e}")


def print_frames(filepath, verbose=False):
    """Print frame information from a .d6a file."""
    frames = read_d6a(filepath)
    total_ms = sum(f['time_ms'] for f in frames)

    print(f"File: {filepath}")
    print(f"Frames: {len(frames)}")
    print(f"Total duration: {total_ms}ms ({total_ms/1000:.2f}s)")
    print()

    if verbose:
        print("Frame details:")
        for i, frame in enumerate(frames):
            print(f"\n  Frame {i}: {frame['time_ms']}ms")
            servos = frame['servos']
            # Print servos in rows of 4
            for row_start in range(1, 17, 4):
                row = [f"S{j}:{servos[str(j)]:4d}" for j in range(row_start, min(row_start+4, 17))]
                print(f"    {' '.join(row)}")
    else:
        print("First frame servo values:")
        if frames:
            servos = frames[0]['servos']
            for row_start in range(1, 17, 4):
                row = [f"S{j}:{servos[str(j)]:4d}" for j in range(row_start, min(row_start+4, 17))]
                print(f"  {' '.join(row)}")
        print("\n(Use -v for all frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Read and convert TonyPi .d6a action files"
    )
    parser.add_argument("input", nargs="?", help="Input .d6a file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show all frame details")
    parser.add_argument("--list", metavar="DIR", nargs="?", const="./action_groups",
                        help="List all .d6a files in directory (default: ./action_groups)")

    args = parser.parse_args()

    if args.list is not None:
        list_actions(args.list)
        return

    if not args.input:
        parser.print_help()
        return

    if args.output:
        d6a_to_json(args.input, args.output)
    else:
        print_frames(args.input, verbose=args.verbose)


if __name__ == "__main__":
    main()
