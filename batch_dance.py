#!/usr/bin/env python3
"""
Batch Dance Generator - Send audio file to FACT server and get complete dance.

This script sends an audio file to the FACT server, which generates
all dance motion frames at once. The result is saved as a JSON file that
can be played back with play_dance.py.

Usage:
    python batch_dance.py song.mp3 --output dance.json
    python batch_dance.py song.wav --server ws://132.145.180.105:8765

The server must be running on a GPU machine for reasonable performance.
"""

import argparse
import asyncio
import json
import os
import sys
import time

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Run: pip install websockets")
    sys.exit(1)


async def generate_dance(audio_path, server_url, verbose=True):
    """Send audio file to server and receive complete dance.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        server_url: WebSocket URL of FACT server
        verbose: Print progress updates

    Returns:
        dict with dance frames and metadata
    """
    if verbose:
        print(f"\nConnecting to {server_url}...")

    # Read file as binary
    with open(audio_path, 'rb') as f:
        file_data = f.read()

    file_size_mb = len(file_data) / (1024 * 1024)
    if verbose:
        print(f"File size: {file_size_mb:.1f} MB")

    async with websockets.connect(
        server_url,
        max_size=100 * 1024 * 1024,  # 100MB for large responses
        ping_timeout=300,  # 5 min timeout for long processing
        close_timeout=10
    ) as ws:
        if verbose:
            print("Connected! Sending file...")

        # Send file as binary
        start_time = time.time()
        await ws.send(file_data)

        if verbose:
            print(f"Sent {len(file_data)} bytes")
            print("Waiting for server to generate dance...")

        # Receive progress updates and final result
        result = None
        while True:
            response = await ws.recv()
            data = json.loads(response)

            if data['type'] == 'progress':
                if verbose:
                    stage = data.get('stage', '')
                    msg = data.get('message', '')

                    if stage == 'generating':
                        done = data.get('frames_done', 0)
                        total = data.get('total_frames', 1)
                        fps = data.get('fps', 0)
                        eta = data.get('eta_seconds', 0)
                        pct = 100 * done / total if total > 0 else 0
                        print(f"  Generating: {done}/{total} ({pct:.0f}%) - {fps:.1f} fps, ETA {eta}s", end='\r')
                    else:
                        print(f"  {msg}")

            elif data['type'] == 'batch_result':
                result = data
                break

            elif data['type'] == 'error':
                raise RuntimeError(f"Server error: {data.get('message', 'Unknown error')}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n\nDance generation complete!")
            print(f"  Frames: {result['n_frames']}")
            print(f"  Duration: {result['duration_seconds']:.1f}s")
            print(f"  FPS: {result['fps']}")
            print(f"  Server time: {result['generation_time_seconds']:.1f}s")
            print(f"  Total time: {elapsed:.1f}s")

        return result


def save_dance(result, output_path, audio_path=None):
    """Save dance result to JSON file.

    Args:
        result: Result dict from generate_dance
        output_path: Output JSON file path
        audio_path: Optional audio file path to include as reference
    """
    # Add audio reference if provided
    if audio_path:
        result['audio_file'] = os.path.basename(audio_path)

    # Add active servos info for play_dance.py compatibility
    try:
        from tony_pro import ACTIVE_RETARGET_JOINTS, RETARGET_MAP
        result['active_servos'] = ACTIVE_RETARGET_JOINTS
        result['servo_map'] = {
            name: RETARGET_MAP[name]['servo_id']
            for name in ACTIVE_RETARGET_JOINTS
        }
    except ImportError:
        pass  # tony_pro not available, skip servo info

    with open(output_path, 'w') as f:
        json.dump(result, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")


async def main():
    parser = argparse.ArgumentParser(
        description='Generate dance from audio using FACT model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_dance.py song.mp3
    python batch_dance.py music.wav --output my_dance.json
    python batch_dance.py track.mp3 --server ws://132.145.180.105:8765

The output JSON can be played back with:
    python play_dance.py --input dance.json --audio song.mp3
        """
    )
    parser.add_argument('audio', type=str, help='Input audio file (mp3, wav, etc.)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSON file (default: <audio_name>_dance.json)')
    parser.add_argument('--server', '-s', type=str, default='ws://132.145.180.105:8765',
                       help='FACT server WebSocket URL')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Generate output filename if not specified
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        args.output = f"{base}_dance.json"

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("FACT Batch Dance Generator")
        print("=" * 60)
        print(f"Audio: {args.audio}")

    # Generate dance
    result = await generate_dance(args.audio, args.server, verbose=verbose)

    # Save result
    save_dance(result, args.output, audio_path=args.audio)

    if verbose:
        print("\nTo play back on robot:")
        print(f"  python play_dance.py --input {args.output} --audio {args.audio}")


if __name__ == '__main__':
    asyncio.run(main())
