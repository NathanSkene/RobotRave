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

### 2. Retargeting → Servo Values

`retarget_to_tonypi.py` converts SMPL angles to TonyPi servo pulse values (0-1000 range). This maps the 24 human joints to the robot's 16 bus servos + 2 PWM head servos.

### 3. Feature Matching → Choreography

Instead of sending retargeted servo values directly (which can be unstable), we:

1. **Extract motion features** from segments of the dance:
   - Body part activity (arms, legs, both)
   - Motion type (static, oscillating, cyclic, gesture)
   - Periodicity and symmetry
   - Energy (velocity, displacement)

2. **Match to pre-scripted actions** from a library of 117 tested .d6a files

3. **Generate choreography** - a timeline of action triggers synced to the music

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

## Other Approaches

See [OLD_APPROACHES.md](OLD_APPROACHES.md) for previous approaches we tried:
- Direct servo playback from FACT output
- Beat-sync dancing with aubio
- Real-time WebSocket streaming
