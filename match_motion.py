#!/usr/bin/env python3
"""
Motion matching - find the best action for a given motion segment.

Uses feature-based similarity scoring to match FACT-generated motion
to pre-scripted .d6a actions.

Usage:
    from match_motion import find_best_action, compute_similarity
"""

import json


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets.

    Returns 0-1 where 1 = identical sets
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def activity_similarity(act1, act2):
    """Compute similarity between two activity masks.

    Args:
        act1, act2: Dicts with 'left_arm', 'right_arm', 'left_leg', 'right_leg' bools

    Returns:
        0-1 similarity score
    """
    keys = ['left_arm', 'right_arm', 'left_leg', 'right_leg']

    active1 = {k for k in keys if act1.get(k, False)}
    active2 = {k for k in keys if act2.get(k, False)}

    return jaccard_similarity(active1, active2)


def compute_similarity(f1, f2, weights=None):
    """Compute similarity score between two feature sets.

    Args:
        f1: Features of FACT segment
        f2: Features of library action
        weights: Optional dict of feature weights (default uses balanced weights)

    Returns:
        Score 0-1 where 1 = perfect match
    """
    if weights is None:
        weights = {
            'activity': 0.25,
            'motion_type': 0.20,
            'periodicity': 0.15,
            'symmetry': 0.15,
            'energy': 0.15,
            'complexity': 0.10,
        }

    score = 0.0

    # 1. Activity match (Jaccard similarity on active body parts)
    act1 = f1.get('activity', {})
    act2 = f2.get('activity', {})
    activity_score = activity_similarity(act1, act2)
    score += weights['activity'] * activity_score

    # 2. Motion type match
    type1 = f1.get('motion_type', 'static')
    type2 = f2.get('motion_type', 'static')

    if type1 == type2:
        type_score = 1.0
    elif {type1, type2} == {'oscillating', 'cyclic'}:
        # These are similar-ish
        type_score = 0.6
    elif 'static' in {type1, type2}:
        # Static vs anything else is poor match
        type_score = 0.1
    else:
        type_score = 0.3

    score += weights['motion_type'] * type_score

    # 3. Periodicity match
    periodic1 = f1.get('is_periodic', False)
    periodic2 = f2.get('is_periodic', False)

    if periodic1 == periodic2:
        if periodic1:
            # Both periodic - compare frequencies
            freq1 = f1.get('frequency', 0)
            freq2 = f2.get('frequency', 0)
            # Allow for some frequency difference (within 2 Hz)
            freq_diff = abs(freq1 - freq2)
            periodic_score = max(0, 1.0 - freq_diff / 2.0)
        else:
            # Both non-periodic
            periodic_score = 1.0
    else:
        periodic_score = 0.3

    score += weights['periodicity'] * periodic_score

    # 4. Symmetry match
    sym1 = f1.get('symmetry_type', 'static')
    sym2 = f2.get('symmetry_type', 'static')

    if sym1 == sym2:
        sym_score = 1.0
    elif sym1 == 'static' or sym2 == 'static':
        sym_score = 0.2
    elif {sym1, sym2} == {'mirror', 'parallel'}:
        # Both symmetric but different type
        sym_score = 0.5
    else:
        sym_score = 0.3

    score += weights['symmetry'] * sym_score

    # 5. Energy match (normalized velocity)
    vel1 = f1.get('avg_velocity', 0)
    vel2 = f2.get('avg_velocity', 0)

    # Normalize to 0-1 scale (assuming max velocity ~100)
    max_vel = max(vel1, vel2, 1)
    energy_diff = abs(vel1 - vel2) / max(max_vel, 10)
    energy_score = max(0, 1.0 - energy_diff)

    score += weights['energy'] * energy_score

    # 6. Complexity match (active DOF)
    dof1 = f1.get('active_dof', 0)
    dof2 = f2.get('active_dof', 0)

    # Score based on difference in active DOF (out of 16)
    dof_diff = abs(dof1 - dof2)
    complexity_score = max(0, 1.0 - dof_diff / 8.0)  # 8 DOF difference = 0

    score += weights['complexity'] * complexity_score

    return score


def find_best_action(segment_features, library, min_score=0.0, duration_tolerance=0.5):
    """Find the best matching action for a motion segment.

    Args:
        segment_features: Features extracted from FACT segment
        library: Action library dict (from build_library.py output)
        min_score: Minimum score to accept (0-1)
        duration_tolerance: How much duration difference is acceptable (0-1)
            0.5 means action can be 50% shorter or 200% longer

    Returns:
        (action_name, score) tuple, or ('stand', 0.0) if no good match
    """
    actions = library.get('actions', library)  # Handle both formats

    best_action = 'stand'
    best_score = 0.0

    segment_duration = segment_features.get('duration_ms', 0)

    for name, action in actions.items():
        action_features = action.get('features', action)
        action_duration = action.get('duration_ms', 0)

        # Optional: filter by duration
        if segment_duration > 0 and action_duration > 0 and duration_tolerance < 1.0:
            duration_ratio = action_duration / segment_duration
            if duration_ratio < (1 - duration_tolerance) or duration_ratio > (1 + duration_tolerance):
                continue

        score = compute_similarity(segment_features, action_features)

        if score > best_score:
            best_score = score
            best_action = name

    if best_score < min_score:
        return 'stand', 0.0

    return best_action, best_score


def find_top_matches(segment_features, library, top_k=5):
    """Find the top K matching actions for a segment.

    Args:
        segment_features: Features extracted from FACT segment
        library: Action library dict
        top_k: Number of matches to return

    Returns:
        List of (action_name, score) tuples, sorted by score descending
    """
    actions = library.get('actions', library)
    matches = []

    for name, action in actions.items():
        action_features = action.get('features', action)
        score = compute_similarity(segment_features, action_features)
        matches.append((name, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches[:top_k]


def explain_match(segment_features, action_features, action_name):
    """Generate human-readable explanation of why an action was matched.

    Returns string explanation
    """
    lines = [f"Match: {action_name}"]

    # Activity
    act1 = segment_features.get('activity', {})
    act2 = action_features.get('features', action_features).get('activity', {})

    seg_parts = []
    act_parts = []
    for part in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
        if act1.get(part):
            seg_parts.append(part.replace('_', ' '))
        if act2.get(part):
            act_parts.append(part.replace('_', ' '))

    lines.append(f"  Segment active: {', '.join(seg_parts) or 'none'}")
    lines.append(f"  Action active:  {', '.join(act_parts) or 'none'}")

    # Motion type
    type1 = segment_features.get('motion_type', 'static')
    type2 = action_features.get('features', action_features).get('motion_type', 'static')
    lines.append(f"  Motion type: segment={type1}, action={type2}")

    # Periodicity
    periodic1 = segment_features.get('is_periodic', False)
    periodic2 = action_features.get('features', action_features).get('is_periodic', False)
    freq1 = segment_features.get('frequency', 0)
    freq2 = action_features.get('features', action_features).get('frequency', 0)

    if periodic1 or periodic2:
        lines.append(f"  Periodic: segment={periodic1} ({freq1:.1f}Hz), action={periodic2} ({freq2:.1f}Hz)")

    # Symmetry
    sym1 = segment_features.get('symmetry_type', 'static')
    sym2 = action_features.get('features', action_features).get('symmetry_type', 'static')
    lines.append(f"  Symmetry: segment={sym1}, action={sym2}")

    return '\n'.join(lines)


def load_library(library_path):
    """Load action library from JSON file.

    Args:
        library_path: Path to library.json

    Returns:
        Library dict
    """
    with open(library_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Test matching
    import sys
    from motion_features import extract_features

    if len(sys.argv) < 3:
        print("Usage: python match_motion.py <library.json> <test.d6a>")
        print("  Tests matching a .d6a file against the library")
        sys.exit(1)

    from read_d6a import read_d6a

    library_path = sys.argv[1]
    test_path = sys.argv[2]

    print(f"Loading library from {library_path}...")
    library = load_library(library_path)
    print(f"Loaded {library.get('n_actions', len(library))} actions")

    print(f"\nExtracting features from {test_path}...")
    frames = read_d6a(test_path)
    features = extract_features(frames)

    print(f"\nTop 5 matches:")
    matches = find_top_matches(features, library, top_k=5)
    for name, score in matches:
        print(f"  {name:25s} {score:.3f}")

    if matches:
        best_name, best_score = matches[0]
        action = library['actions'][best_name]
        print(f"\n{explain_match(features, action, best_name)}")
