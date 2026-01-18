# RobotRave

Make a TonyPi Pro humanoid robot dance autonomously to music using AI-generated choreography.

## How It Works

```
Audio (.mp3)
    ↓ FACT Model (Lambda GPU server)
SMPL Motion (.npy)
    ↓ Retargeting (retarget_to_tonypi.py)
Dance JSON (servo values)
    ↓ Feature Extraction & Matching (generate_choreography.py)
Choreography JSON (action triggers)
    ↓ Playback (play_synced.py)
Robot Dances + Music Plays
```

We use Google's **FACT model** to generate human dance motion from audio, then convert it to sequences of **pre-scripted robot actions** using feature-based matching. This is more stable than sending raw AI-generated servo values directly.

**Full pipeline documentation:** [CHOREOGRAPHY_README.md](CHOREOGRAPHY_README.md)

---

## Quick Start

```bash
# 1. Generate choreography from FACT dance output
python3 generate_choreography.py \
    --dance examples/dance_house_tonypi.json \
    --library library.json \
    --output choreo.json

# 2. Preview the action sequence
python3 play_choreography.py --input choreo.json --preview

# 3. Copy to robot and play with synced music
scp choreo.json play_choreography.py read_d6a.py pi@raspberrypi.local:~/
python3 play_synced.py
```

---

## The Pipeline Explained

### 1. FACT Model → SMPL Motion

The [FACT model](https://google.github.io/aichoreographer/) (Full-Attention Cross-modal Transformer) generates human dance motion from audio. It was trained on the [AIST++ dataset](https://google.github.io/aistplusplus_dataset/) - 1,408 dance sequences across 10 genres performed by professional dancers.

**What is SMPL?**

SMPL (Skinned Multi-Person Linear Model) is a standard format for describing human body poses. FACT outputs SMPL data: 24 joints × 3 rotation values = 72 numbers per frame at 60fps.

```
         head
           │
         neck
           │
    ┌──────┴──────┐
 L shoulder    R shoulder
    │              │
 L elbow       R elbow
    │              │
 L wrist       R wrist
           │
        spine
           │
    ┌──────┴──────┐
  L hip        R hip
    │              │
  L knee       R knee
    │              │
 L ankle       R ankle
```

### 2. Retargeting → Dance JSON

`retarget_to_tonypi.py` converts SMPL angles to TonyPi servo pulse values, producing a **Dance JSON** file:

```json
{
  "fps": 60,
  "n_frames": 1797,
  "frames": [
    {"time_ms": 16, "servos": {"1": 880, "2": 508, "8": 674, ...}},
    {"time_ms": 16, "servos": {"1": 875, "2": 511, "8": 733, ...}},
    ...
  ]
}
```

Each frame contains servo pulse values (0-1000) for all 16 bus servos, representing what the robot "should" do if we directly played back the FACT output. But we don't use these values directly - see next step.

### 3. Feature Extraction & Matching → Choreography

Instead of sending the Dance JSON servo values directly to the robot (which causes stability issues), we use **feature-based matching** to convert it to a sequence of pre-scripted actions.

**The mathematical approach:**

1. **Segment the dance** into variable-length windows using motion boundary detection (local minima in total velocity)

2. **Extract features** from each segment:
   - **Activity mask**: Which servo groups move (arms/legs) based on movement range > threshold
   - **Motion type**: Classify as static/oscillating/cyclic/gesture using FFT frequency analysis
   - **Periodicity**: Detect via autocorrelation peaks, extract dominant frequency
   - **Symmetry**: Compute left-right correlation (mirror/parallel/one-sided/asymmetric)
   - **Energy**: Total displacement, average velocity, peak velocity
   - **Complexity**: Count of active degrees of freedom

3. **Score similarity** between segment features and each library action using weighted matching:
   ```
   score = 0.25 × activity_jaccard
         + 0.20 × motion_type_match
         + 0.15 × periodicity_match
         + 0.15 × symmetry_match
         + 0.15 × energy_similarity
         + 0.10 × complexity_match
   ```

4. **Select best action** for each segment, generating a choreography timeline:
   ```json
   {"time_ms": 0, "action": "go_forward", "score": 0.65}
   {"time_ms": 1250, "action": "turn_left", "score": 0.73}
   ...
   ```

### 4. Playback

`play_synced.py` plays music on your laptop while SSH-triggering actions on the robot, keeping them synchronized.

---

## Setup

```bash
git clone https://github.com/NathanSkene/RobotRave.git
cd RobotRave
./setup.sh
```

The setup script clones the [MINT repository](https://github.com/google-research/mint) and downloads model checkpoints.

**Passwordless SSH to robot:**
```bash
ssh-copy-id pi@raspberrypi.local
# Password: raspberrypi
```

---

## TonyPi Pro Servo Mapping

**Bus Servos (IDs 1-16, pulse 0-1000, center=500):**

| ID | Joint | ID | Joint |
|----|-------|----|-------|
| 1 | l_ankle_roll | 9 | r_ankle_roll |
| 2 | l_ankle_pitch | 10 | r_ankle_pitch |
| 3 | l_knee_pitch | 11 | r_elbow_pitch |
| 4 | l_hip_pitch | 12 | r_hip_pitch |
| 5 | l_hip_roll | 13 | r_hip_roll |
| 6 | l_elbow_pitch | 14 | r_knee_pitch |
| 7 | l_shoulder_roll | 15 | r_shoulder_roll |
| 8 | l_shoulder_pitch | 16 | r_shoulder_pitch |

**PWM Servos (head, pulse ~1000-2000, center=1500):**
- pwm1: head_pitch
- pwm2: head_yaw

---

## Project Structure

```
RobotRave/
├── CHOREOGRAPHY_README.md   # Full pipeline documentation
├── examples/                # Sample files
│   ├── dance_house_tonypi.json
│   ├── choreo.json
│   └── mHO2.mp3
├── action_groups/           # 117 pre-scripted .d6a actions
├── library.json             # Pre-computed action features
│
│ # Main Pipeline
├── generate_choreography.py # Feature matching
├── play_choreography.py     # Execute on robot
├── play_synced.py           # Synced laptop+robot playback
├── retarget_to_tonypi.py    # SMPL → servo conversion
│
│ # Support Scripts
├── motion_features.py       # Feature extraction
├── match_motion.py          # Similarity scoring
├── build_library.py         # Build action library
├── read_d6a.py              # Read .d6a files
│
│ # Documentation
├── OLD_APPROACHES.md        # Previous approaches we tried
└── mint/                    # Google FACT model (submodule)
```

---

## Server Details

**Lambda Labs (FACT inference):**
- SSH: `ssh ubuntu@132.145.180.105`

**TonyPi Robot:**
- SSH: `ssh pi@raspberrypi.local`
- Password: `raspberrypi`
- Actions: `/home/pi/TonyPi/ActionGroups`

---

## References

- [AI Choreographer (FACT)](https://google.github.io/aichoreographer/) - Google Research
- [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/) - Dance motion data
- [MINT Repository](https://github.com/google-research/mint) - FACT implementation
- [Hiwonder TonyPi](https://www.hiwonder.com/products/tonypi-pro) - Robot hardware

---

## Why Not Direct Servo Playback?

Our original approach was to map SMPL joint angles directly to servo commands and play them back frame-by-frame. This failed because:

- **Collision detection**: SMPL assumes a human body with flesh and joints that can pass through each other in edge cases. The robot's rigid legs would collide, causing it to fall.
- **Balance**: Human dancers constantly make micro-adjustments for balance. FACT doesn't output these, so the robot would tip over during weight shifts.
- **Range mismatch**: FACT outputs use the full human joint range, but the robot's servos have mechanical limits. Clamping caused jerky, unnatural motion.

The feature-based choreography approach sidesteps these issues by using **pre-tested, stable robot actions** that we know work, and just selecting *which* action to play based on what the AI dance is trying to do.

See [OLD_APPROACHES.md](OLD_APPROACHES.md) for more details on approaches we tried.
