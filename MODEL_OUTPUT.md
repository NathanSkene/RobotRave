# FACT Model Output Format

This document describes the data pipeline from raw FACT model output to servo commands.

## Stage 1: Raw FACT Output

**Shape: `[n_frames, 225]`** at 60 FPS

Each frame is 225 numbers:
- `[0:6]` - Padding (zeros)
- `[6:9]` - Root translation (x, y, z) in meters
- `[9:225]` - 24 joints × 9 values = 216 values (flattened 3×3 rotation matrices)

```
Frame 0: [0, 0, 0, 0, 0, 0, 0.0, 0.92, 0.0, 1, 0, 0, 0, 1, 0, 0, 0, 1, ...]
                              └─translation─┘ └──joint 0 rotmat (identity)──┘
```

---

## Stage 2: Joint Angles

**Shape: `[n_frames, 24, 3]`** - Euler angles in radians

The 24 SMPL joints:
```
 0: root        6: spine       12: neck        18: l_elbow
 1: l_hip       7: l_ankle     13: l_collar    19: r_elbow
 2: r_hip       8: r_ankle     14: r_collar    20: l_wrist
 3: belly       9: chest       15: head        21: r_wrist
 4: l_knee     10: l_toes      16: l_shoulder  22: l_hand
 5: r_knee     11: r_toes      17: r_shoulder  23: r_hand
```

Example output:
```
Frame 0, Joint 17 (r_shoulder): [0.52, -0.31, 0.18]  # radians (x, y, z rotation)
Frame 0, Joint 15 (head):       [0.08, 0.15, -0.02]
```

---

## Stage 3: Tony Pro Servo Commands

**JSON stream** - pulse values (0-1000, center=500)

```json
{
  "type": "servos",
  "servos": {
    "7": 598,
    "5": 523,
    "15": 412,
    "13": 489,
    "8": 515,
    "16": 528,
    "1": 502,
    "2": 497,
    "3": 511,
    "4": 505,
    "6": 498
  },
  "inference_time_ms": 45,
  "fps": 22.2
}
```

### Servo ID Mapping

| ID | Joint | Description |
|----|-------|-------------|
| 7 | right_shoulder_pitch | Right arm forward/back |
| 5 | right_elbow_pitch | Right elbow bend |
| 15 | left_shoulder_pitch | Left arm forward/back |
| 13 | left_elbow_pitch | Left elbow bend |
| 8 | head_pitch | Head up/down |
| 16 | head_yaw | Head left/right |
| 1 | right_hip_roll | Right hip side-to-side |
| 2 | right_hip_yaw | Right hip rotation |
| 3 | right_knee_pitch | Right knee bend |
| 4 | right_hip_pitch | Right hip forward/back |
| 6 | right_ankle_pitch | Right ankle flex |

---

## Real-time Stream Example

What the robot receives at 60 FPS:

```
t=0.000s  {7: 500, 5: 500, 15: 500, 13: 500, 8: 500, 16: 500, ...}  <- neutral
t=0.016s  {7: 512, 5: 503, 15: 488, 13: 498, 8: 502, 16: 505, ...}  <- slight movement
t=0.033s  {7: 538, 5: 515, 15: 462, 13: 491, 8: 508, 16: 512, ...}  <- arms rising
t=0.050s  {7: 589, 5: 542, 15: 411, 13: 478, 8: 518, 16: 521, ...}  <- arms up!
t=0.066s  {7: 621, 5: 568, 15: 379, 13: 462, 8: 525, 16: 528, ...}  <- peak
t=0.083s  {7: 598, 5: 551, 15: 402, 13: 471, 8: 520, 16: 524, ...}  <- coming down
...
```

### Pulse Value Interpretation

- **500** = center/neutral position
- **<500** = one direction (e.g., arm back, head left)
- **>500** = other direction (e.g., arm forward, head right)
- Range typically **200-800** (clamped for safety)

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Audio File (.mp3/.wav)                                                 │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │  Audio Feature Extraction               │                            │
│  │  - 35 dims per frame (60 FPS)           │                            │
│  │  - MFCC (20) + Chroma (12) + beats (3)  │                            │
│  └─────────────────────────────────────────┘                            │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │  FACT Model (GPU)                       │                            │
│  │  - Input: audio features + seed motion  │                            │
│  │  - Output: [n_frames, 225] rotation     │                            │
│  │    matrices for 24 SMPL joints          │                            │
│  └─────────────────────────────────────────┘                            │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │  Rotation Matrix → Euler Angles         │                            │
│  │  - [n_frames, 24, 3] radians            │                            │
│  │  - scipy.spatial.transform.Rotation     │                            │
│  └─────────────────────────────────────────┘                            │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │  Retarget to Tony Pro                   │                            │
│  │  - 24 SMPL joints → 11 active servos    │                            │
│  │  - Radians → pulse values (0-1000)      │                            │
│  │  - Safety clamping (200-800)            │                            │
│  └─────────────────────────────────────────┘                            │
│       │                                                                 │
│       ▼                                                                 │
│  Servo Commands JSON (60 FPS stream)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `generate_dance.py` | Batch: audio → motion `.npy` files |
| `retarget_to_tonypi.py` | Batch: motion `.npy` → servo commands `.json` |
| `fact_server.py` | Real-time: WebSocket server for streaming |
| `fact_client.py` | Real-time: Robot client connects to server |
| `play_dance.py` | Playback: Execute pre-generated `.json` on robot |
