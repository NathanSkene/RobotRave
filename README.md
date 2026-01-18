# RobotRave

Make a TonyPi Pro humanoid robot dance autonomously to music using AI-generated choreography.

## Current Approach: Feature-Based Choreography

We use Google's **FACT model** to generate human dance motion from audio, then convert it to sequences of **pre-scripted robot actions** using feature-based matching. This approach is more stable than sending raw servo values directly.

```
Audio (.mp3)
    ↓ FACT Model (Lambda GPU)
Motion Data (.npy)
    ↓ Retargeting
Dance JSON (servo values)
    ↓ Feature Matching  ← THIS IS THE KEY STEP
Choreography JSON (action triggers)
    ↓ Playback
Robot Dances + Music Plays
```

**Full pipeline documentation:** [CHOREOGRAPHY_README.md](CHOREOGRAPHY_README.md)

### Quick Start (Choreography Pipeline)

```bash
# Generate choreography from FACT dance
python3 generate_choreography.py \
    --dance examples/dance_house_tonypi.json \
    --library library.json \
    --output choreo.json

# Preview the action sequence
python3 play_choreography.py --input choreo.json --preview

# Play on robot with synced music
python3 play_synced.py
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/RobotRave.git
cd RobotRave
./setup.sh
```

The setup script will:
1. Clone the [MINT repository](https://github.com/google-research/mint) (Google's FACT model implementation)
2. Download pre-trained model checkpoints (~500MB) from [Google Drive](https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm)
3. Optionally install Python dependencies

**Note:** The model weights are not stored in this repo due to size. The setup script uses `gdown` to fetch them automatically.

---

## How It Works

### The FACT Model

We use Google's **FACT model** (Full-Attention Cross-modal Transformer) from their [AI Choreographer](https://google.github.io/aichoreographer/) research to generate human dance movements from music. The model was trained on the AIST++ dataset containing 1,408 dance sequences across 10 genres.

### Feature-Based Motion Matching

Instead of sending FACT's raw servo values directly to the robot (which can cause instability), we:

1. **Extract motion features** from FACT output:
   - Body part activity (arms, legs, both)
   - Motion type (static, oscillating, cyclic, single gesture)
   - Periodicity (frequency of repetitive motion)
   - Symmetry (mirror, parallel, one-sided)
   - Energy (velocity, displacement)

2. **Match to pre-scripted actions** from a library of 117 .d6a files (walks, kicks, waves, turns, etc.)

3. **Generate choreography** - a timeline of action triggers synced to the music

This gives stable, tested robot motion while still capturing the essence of the AI-generated dance.

### Why FACT?
We evaluated three approaches:
- [**MDLT**](https://github.com/meowatthemoon/MDLT) (Music to Dance Language Translation) - No pre-trained weights, requires ~140 hours training ([paper](https://arxiv.org/abs/2403.15569))
- [**DFKI Robot Dance**](https://ieeexplore.ieee.org/document/9981462) - Physically feasible trajectories but code not publicly available
- [**FACT**](https://github.com/google-research/mint) (Google) - Pre-trained checkpoints available, state-of-the-art quality ([paper](https://arxiv.org/abs/2101.08779))

FACT was the clear winner: Google already spent months training it on professional dancers, and we can just download and use it.

### Fallback: Beat-Sync Dancing
If FACT inference is too slow or fails, we have rule-based beat-synchronized dancing using `aubio` for real-time beat/onset detection with pre-defined dance moves.

---

## How It Works (Plain English)

### What is the FACT Model?

FACT is a neural network that learned to dance by watching 1,408 videos of professional dancers. Google researchers trained it to understand the relationship between music and movement - things like:

- "When there's a drum hit, raise your arms"
- "During a bass drop, crouch down"
- "When the melody rises, extend outward"

You feed it a song, and it outputs a sequence of body poses - one for every frame (60 per second). It's like having a professional choreographer that works instantly.

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Music (tempo, beats, melody, energy)            │
│         + 2-second "seed" motion to start from          │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │         FACT Neural Network                      │    │
│  │                                                  │    │
│  │   Learned from watching 5.2 hours of dance      │    │
│  │   videos across 10 genres (pop, hip-hop,        │    │
│  │   breakdance, house, krump, etc.)               │    │
│  └─────────────────────────────────────────────────┘    │
│                          ↓                              │
│  OUTPUT: Body pose every 1/60th of a second             │
│          (72 numbers describing joint angles)           │
└─────────────────────────────────────────────────────────┘
```

### What is AIST++?

[AIST++](https://google.github.io/aistplusplus_dataset/) is a massive dance dataset that Google created:

- **1,408 dance sequences** performed by 30 professional dancers
- **10 dance genres**: pop, lock, waack, jazz, house, middle hip-hop, break, krump, street jazz, LA house
- **5.2 hours** of motion capture data (10.1 million frames)
- Every dance was recorded with 9 cameras and precise motion tracking

This is what FACT learned from. The quality of the output depends on the quality of the training data, and AIST++ contains genuinely skilled dancers.

### What is SMPL?

SMPL (Skinned Multi-Person Linear Model) is a standard way to describe human body poses mathematically. Instead of describing movement in words ("raise your right arm"), SMPL uses numbers.

```
         head (joint 15)
           │
         neck (joint 12)
           │
    ┌──────┴──────┐
 L shoulder    R shoulder
 (joint 16)    (joint 17)
    │              │
 L elbow       R elbow
 (joint 18)    (joint 19)
    │              │
 L wrist       R wrist
 (joint 20)    (joint 21)
           │
        spine
           │
    ┌──────┴──────┐
  L hip        R hip
 (joint 1)    (joint 2)
    │              │
  L knee       R knee
 (joint 4)    (joint 5)
    │              │
 L ankle       R ankle
 (joint 7)    (joint 8)
```

Each of the 24 joints has 3 rotation values (x, y, z axes). So **72 numbers** completely describe any human pose. FACT outputs these 72 numbers for every frame of the dance.

### How We Translate 24 Joints → 16 Servos

The challenge: SMPL describes a human body with 24 joints, but Tony Pro only has 16 servos. They don't match up perfectly.

**What we keep:**
| SMPL Joint | Tony Pro Servo | Why |
|------------|---------------|-----|
| head | head_pitch (8), head_yaw (16) | Direct match |
| r_shoulder | right_shoulder_pitch (7) | Take the forward/back rotation |
| r_elbow | right_elbow_pitch (5) | Take the bend angle |
| l_shoulder | left_shoulder_pitch (15) | Take the forward/back rotation |
| l_elbow | left_elbow_pitch (13) | Take the bend angle |
| r_hip | right_hip_roll/yaw/pitch (1,2,4) | 3 DOF match |
| r_knee | right_knee_pitch (3) | Direct match |
| r_ankle | right_ankle_pitch (6) | Direct match |
| (same for left leg) | ... | ... |

**What we lose:**
- Shoulder roll (Tony Pro arms only move forward/back, not sideways)
- Wrist rotation (no wrist servos)
- Spine/torso twist (no torso servo)
- Finger movements (no hand servos)

**The conversion process:**
```python
# SMPL gives us: right_shoulder rotation = [0.5, -0.3, 0.2] radians (x, y, z)
# Tony Pro only has shoulder pitch (forward/back motion)

# 1. Extract the relevant axis (x = pitch = forward/back)
angle = smpl_right_shoulder[0]  # 0.5 radians

# 2. Convert radians to servo pulse (0-1000 scale, 500 = center)
pulse = 500 + (angle * 191)  # 191 pulse units per radian

# 3. Clamp to safe range (don't break the servo!)
pulse = clamp(pulse, 200, 800)

# 4. Send to robot
servo.set_position(7, pulse)  # Servo 7 = right shoulder
```

**Safety scaling:**
- Arms and head: 100% of SMPL motion (safe to move freely)
- Legs: 30% of SMPL motion (robot can fall over if legs move too much)

The result isn't a perfect recreation of the human dance, but it captures the essence - when the dancer's arms go up, the robot's arms go up. When they bob their head, the robot bobs its head.

**For detailed output format documentation, see [MODEL_OUTPUT.md](MODEL_OUTPUT.md).**

---

## TonyPi Pro Servo Mapping

**Bus Servos (IDs 1-16, pulse range 0-1000, center=500):**

| ID | Joint | Group |
|----|-------|-------|
| 1 | l_ankle_roll | left_leg |
| 2 | l_ankle_pitch | left_leg |
| 3 | l_knee_pitch | left_leg |
| 4 | l_hip_pitch | left_leg |
| 5 | l_hip_roll | left_leg |
| 6 | l_elbow_pitch | left_arm |
| 7 | l_shoulder_roll | left_arm |
| 8 | l_shoulder_pitch | left_arm |
| 9 | r_ankle_roll | right_leg |
| 10 | r_ankle_pitch | right_leg |
| 11 | r_elbow_pitch | right_arm |
| 12 | r_hip_pitch | right_leg |
| 13 | r_hip_roll | right_leg |
| 14 | r_knee_pitch | right_leg |
| 15 | r_shoulder_roll | right_arm |
| 16 | r_shoulder_pitch | right_arm |

**PWM Servos (head, pulse range ~1000-2000, center=1500):**

| ID | Joint |
|----|-------|
| pwm1 | head_pitch |
| pwm2 | head_yaw |

**Summary:**
- Head: 2 DOF (PWM servos, not bus servos)
- Each arm: 3 DOF (shoulder pitch + shoulder roll + elbow pitch)
- Each leg: 5 DOF (hip pitch + hip roll + knee + ankle pitch + ankle roll)

---

## Project Structure

```
RobotRave/
├── CHOREOGRAPHY_README.md   # Full choreography pipeline documentation
├── examples/                # Sample dance, choreography, and audio
│   ├── dance_house_tonypi.json
│   ├── choreo.json
│   └── mHO2.mp3
├── action_groups/           # 117 pre-scripted .d6a robot actions
├── library.json             # Pre-computed action features
│
│ # Choreography Pipeline (Current Approach)
├── generate_choreography.py # Match FACT dance to library actions
├── play_choreography.py     # Execute choreography on robot
├── play_synced.py           # Synced playback (laptop audio + robot)
├── motion_features.py       # Extract motion features
├── match_motion.py          # Similarity scoring
├── build_library.py         # Build action feature library
├── read_d6a.py              # Read .d6a action files
│
│ # Core Scripts
├── tony_pro.py              # Central servo config & controller module
├── retarget_to_tonypi.py    # SMPL → servo command conversion
├── play_dance.py            # Direct servo playback (alternative)
├── generate_dance.py        # Run FACT model on audio files (Lambda)
│
│ # Fallback / Testing
├── beat_sync_dance.py       # Beat-triggered dance moves
├── smart_dance.py           # Style-adaptive dancing
├── discover_servos.py       # Interactive servo discovery
│
│ # Documentation
├── RESEARCH.md              # Research notes & approach analysis
├── CLOUD_SETUP.md           # GPU cloud deployment guide
├── MODEL_OUTPUT.md          # FACT model output format
│
└── (submodules)
    ├── mint/                # Google FACT model
    └── TonyPi/              # Hiwonder SDK
```

---

## Saturday Checklist

### 1. Initial Connection (~15 min)
```bash
# Connect to robot's WiFi network: HW-67B69FFB

# SSH into the robot's Raspberry Pi
ssh pi@192.168.149.1
# Password: raspberrypi

# Copy project files to robot
scp -r *.py tony_pro_config.json pi@192.168.149.1:~/RobotRave/
```

### 2. Verify Servo Mapping (~30 min)
```bash
# On the robot - test the servo map is correct
python tony_pro.py --map

# If servos aren't right, run discovery
python discover_servos.py
```

### 3. Test Dance Moves (~30 min)
```bash
# Simulation mode (prints moves, no robot motion)
python beat_sync_dance.py --test --simulate

# Real robot - test all moves
python beat_sync_dance.py --test
```

### 4. Beat-Sync Dancing (Fallback)
```bash
# Dance to an audio file
python beat_sync_dance.py --audio your_song.wav

# Live microphone input
python beat_sync_dance.py --live
```

### 5. Smart Style-Adaptive Dancing
```bash
# Detects music genre and adapts style
python smart_dance.py --live

# Or with a file
python smart_dance.py --audio song.wav

# Demo all styles
python smart_dance.py --demo --simulate
```

### 6. FACT Model Dancing (If Cloud Server Ready)
```bash
# On cloud GPU server
python fact_server.py --port 8765

# On robot
python fact_client.py --server ws://<server-ip>:8765
```

### 7. Pre-Generated Dances
```bash
# Generate dance from audio (requires FACT)
python generate_dance.py --audio song.mp3 --output dance.json

# Play on robot
python play_dance.py --input dance.json
```

---

## Quick Reference

### Python Imports
```python
from tony_pro import SERVO, MOVES, TonyProController

# Access servo IDs
SERVO.HEAD_PITCH  # 8
SERVO.R_SHOULDER  # 7
SERVO.L_ELBOW     # 13

# Use pre-defined moves
controller = TonyProController(simulate=False)
controller.execute_move('arms_up')
controller.execute_move('celebrate')
controller.go_neutral()
```

### Safety Notes
- Leg servos are limited to 30% range to prevent falls
- Start with `--simulate` flag to test without robot motion
- Use `--neutral-first` to reset pose before playing sequences
- Head movements are limited to prevent strain

---

## Cloud GPU Setup

For real-time FACT inference, see [CLOUD_SETUP.md](CLOUD_SETUP.md).

**Recommended:** AWS g5.xlarge (~$1/hr) with your credits.

---

## Dependencies

**On Robot (Raspberry Pi):**
```bash
pip install numpy aubio sounddevice soundfile
# HiwonderSDK is pre-installed
```

**On Development Machine:**
```bash
pip install numpy aubio sounddevice soundfile scipy
```

**For FACT Model (Cloud GPU):**
```bash
pip install tensorflow numpy scipy websockets soundfile
```

---

## References

- [AI Choreographer (FACT)](https://google.github.io/aichoreographer/) - Google Research
- [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/) - Dance motion data
- [MINT Repository](https://github.com/google-research/mint) - FACT implementation
- [Hiwonder TonyPi](https://www.hiwonder.com/products/tonypi-pro) - Robot hardware

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Servo moves wrong direction | Flip sign in `tony_pro.py` scale factor |
| Movement too jerky | Increase `time_ms` parameter |
| Robot falls over | Reduce leg servo ranges in `SERVO_LIMITS` |
| No beat detection | Check microphone with `--simulate` flag |
| Can't connect to robot | Verify IP, try `ping` first |

---

Built for a 24-hour robot rave hackathon.
