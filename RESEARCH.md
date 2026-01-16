# RobotRave: 24-Hour Dancing Robot Plan

## Project Goal
Make a **Hiwonder TonyPi Pro** humanoid robot dance autonomously to music at a robot rave.

**CRITICAL CONSTRAINT: 24 hours to completion. Robot available Saturday morning.**

**Key Discovery: Pre-trained FACT model checkpoints available!**
- Download: https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm

---

## REVISED 24-HOUR TIMELINE

### Friday (Before Robot Available)
| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1-2 | Download FACT checkpoints + AIST++ API | Working inference setup |
| Hour 2-4 | Run FACT inference on test music | Generated SMPL dance sequences |
| Hour 4-6 | Build SMPL -> TonyPi retargeting | Joint angle converter |
| Hour 6-8 | Write TonyPi servo bridge code | HiwonderSDK integration |
| Hour 8-10 | Test with TonyPi simulator/mock | Validate pipeline end-to-end |
| Hour 10-12 | Prepare fallback: beat-sync rule-based | Backup if ML fails |

### Saturday (With Robot)
| Time | Task | Deliverable |
|------|------|-------------|
| Hour 12-14 | Flash code to robot, initial test | Basic movement working |
| Hour 14-18 | Tune motion, fix servo mapping | Smooth dancing |
| Hour 18-22 | Add real-time audio input | Live music response |
| Hour 22-24 | Polish, add LED effects if time | Rave-ready robot |

---

## 1. Hardware: Hiwonder TonyPi Pro Specifications

### Degrees of Freedom
- **18 DOF base** (expandable to 20 DOF with gripper hands)
- Arm: ~3 DOF per arm (shoulder, elbow, wrist)
- Leg: ~3 DOF per leg (hip, knee, ankle)
- Head: 2 DOF (pan/tilt for camera)

### Servo System
- **Type**: LX-824HV bus servos + LFD-01M anti-blocking servos
- **Communication**: Serial bus protocol (daisy-chained)
- **Feature**: Anti-locking protection prevents burnout

### Processor & Control
- **Brain**: Raspberry Pi 5 (4GB/8GB/16GB options)
- **Vision**: 480P HD wide-angle camera with OpenCV
- **SDK**: Python-based HiwonderSDK
- **Motion Control**: Inverse kinematics + PID control

### Physical Specs
- **Dimensions**: 373 x 186 x 106mm
- **Weight**: ~1.8kg
- **Battery**: 11.1V 2000mAh (60 min runtime)

### SDK Structure (from GitHub/Hiwonder/TonyPi)
```
HiwonderSDK/          # Hardware control
ActionGroups/         # Pre-defined motion sequences
Functions/            # ColorDetect, FaceDetect, etc.
servo_config.yaml     # Servo calibration
```

---

## 2. Available Approaches Analyzed

### Approach A: MDLT (Music to Dance as Language Translation)
**Paper**: Correia & Alexandre, arXiv 2024

**Architecture:**
- Treats music->dance as sequence-to-sequence translation
- Two model options: **Transformer** or **Mamba** (state-space model)
- Input: 438-dim audio features @ 60fps (MFCC, chroma, onset, tempogram)
- Output: 4 joint angles (right arm only) in range [-pi, pi]

**Pros:**
- Code is available and well-structured
- Mamba variant is O(n) complexity - good for real-time
- Output format (joint angles) maps directly to servos

**Cons:**
- Only generates 4 DOF (right arm) - need to extend for full body
- **No pre-trained weights available** - requires ~140 hours training

### Approach B: Robot Dance Generation (DFKI)
**Paper**: Boukheddimi et al., IROS 2022

**Pros:**
- Physically feasible trajectories (respects dynamics/torque limits)
- Generalizable to different robot morphologies

**Cons:**
- **Code not publicly available** (documentation only)
- Requires full robot dynamics model (URDF + Pinocchio)

### Approach C: Google AI Choreographer (FACT) - **CHOSEN APPROACH**
**Paper**: Li et al., ICCV 2021

**Architecture:**
- Full-Attention Cross-modal Transformer (FACT)
- Input: 2-second seed motion + music
- Output: Long-range 3D SMPL motion

**Pros:**
- State-of-the-art quality
- Code available (google-research/mint)
- **Pre-trained checkpoints available!**

**Cons:**
- Outputs SMPL format - requires motion retargeting to TonyPi
- TensorFlow-based

---

## 3. AIST++ Dataset Details

### Contents
- **1,408 dance sequences** across 10 genres
- **30 subjects**, 9 camera views
- **5.2 hours** of motion data (10.1M frames)
- Motion: SMPL format (24 joints) + 3D keypoints (17 COCO joints)
- Audio: Synchronized music files

### Genre Codes
mPO (pop), mLO (lock), mWA (waack), mJB (jazz), mLH (la house),
mMH (middle hip-hop), mBR (break), mKR (krump), mHO (house), mJS (street jazz)

### Data Format
```python
# SMPL joints (24 points):
['root', 'lhip', 'rhip', 'belly', 'lknee', 'rknee', 'spine',
 'lankle', 'rankle', 'chest', 'ltoes', 'rtoes', 'neck',
 'linshoulder', 'rinshoulder', 'head', 'lshoulder', 'rshoulder',
 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhand', 'rhand']
```

---

## 4. CONCEPTUAL OVERVIEW (For Team Explanation)

### What is SMPL?
SMPL (Skinned Multi-Person Linear Model) is a standard way to represent human body poses mathematically.

```
SMPL represents a human body as:
|-- Shape parameters (B): Body proportions (tall/short, wide/narrow)
|-- Pose parameters (O): 24 joint rotations (shoulders, elbows, hips, etc.)
+-- Root translation: Where the body is in 3D space

         head
          |
         neck
          |
    shoulder--shoulder
        |        |
      elbow    elbow
        |        |
      wrist    wrist
          \  /
          spine
          /  \
        hip   hip
        |      |
       knee   knee
        |      |
      ankle  ankle
```

Each joint has 3 rotation values (x, y, z axis). So 24 joints x 3 = **72 numbers describe any human pose**.

### What does the FACT model do?
FACT is a neural network trained by Google on 1,408 dance videos from AIST++.

```
INPUT:  Music audio features (tempo, beats, melody, energy)
        + 2-second "seed" motion to start from
           |
           v
        +====================================+
        |  FACT (Transformer Neural Network) |
        |                                    |
        |  Learned patterns like:            |
        |  "drum hit -> arm raise"           |
        |  "bass drop -> crouch down"        |
        |  "melody rise -> extend arms"      |
        +====================================+
           |
           v
OUTPUT: Sequence of SMPL pose parameters
        (72 numbers per frame, 60 frames/second)
```

**Key**: Google already trained this for months. We just download and use it.

### Motion Retargeting (SMPL -> TonyPi)

The challenge:
- SMPL has 24 joints with human proportions
- TonyPi has 18 servos with robot proportions

Solution (Direct Joint Mapping):
```python
# SMPL gives rotation for "right_shoulder": [0.3, -0.2, 0.5] radians
# TonyPi has 2 shoulder servos (pitch and roll)

# Map relevant axes:
tonypi_r_shoulder_pitch = smpl_r_shoulder[0] * scale_factor
tonypi_r_shoulder_roll  = smpl_r_shoulder[1] * scale_factor

# Clamp to safe servo range:
tonypi_r_shoulder_pitch = clamp(tonypi_r_shoulder_pitch, -90, +90)
```

### Full Pipeline Diagram

```
+--------------------------------------------------------------------------+
|   MUSIC FILE (rave_track.mp3)                                            |
|         |                                                                |
|         v                                                                |
|   +-------------+                                                        |
|   |  librosa    |  -> Extract: tempo, beats, melody, energy              |
|   +-------------+                                                        |
|         |                                                                |
|         v                                                                |
|   +---------------------------------------------+                        |
|   |  FACT Model (pre-trained by Google)         |                        |
|   +---------------------------------------------+                        |
|         |                                                                |
|         v                                                                |
|   SMPL Motion: 72 numbers per frame describing human pose                |
|         |                                                                |
|         v                                                                |
|   +---------------------------------------------+                        |
|   |  Motion Retargeting (our code)              |                        |
|   |  - Map SMPL joints -> TonyPi servos         |                        |
|   |  - Scale rotations to servo ranges          |                        |
|   |  - Limit leg motion for safety              |                        |
|   +---------------------------------------------+                        |
|         |                                                                |
|         v                                                                |
|   TonyPi Servo Commands: [servo1=45deg, servo2=30deg, ...]               |
|         |                                                                |
|         v                                                                |
|   +-------------+                                                        |
|   | HiwonderSDK |  -> Send commands to physical robot                    |
|   +-------------+                                                        |
|         |                                                                |
|         v                                                                |
|   ROBOT DANCES!                                                          |
+--------------------------------------------------------------------------+
```

---

## 5. WHAT WE ACTUALLY DO ON THE DAY (Saturday)

### Step 1: Initial Connection (~30 min)
```bash
# SSH into the robot's Raspberry Pi
ssh pi@<robot-ip-address>

# Or connect via USB serial
screen /dev/ttyUSB0 115200
```

### Step 2: Servo Calibration Test (~1 hour)

**This is the "throw leg up and see if it's shit" part.**

We'll have a test script like this:
```python
# test_servos.py - Run this to check each servo works

from HiwonderSDK import Board

# Test right arm shoulder
print("Testing RIGHT SHOULDER - should raise arm forward")
input("Press Enter to move servo 1 to 45 degrees...")
Board.setBusServoAngle(1, 45, 500)  # servo 1, 45deg, 500ms

print("Did the RIGHT arm move forward? (y/n)")
response = input()
if response == 'n':
    print("PROBLEM: Servo 1 mapping is wrong. Try servo 2?")

# Test each servo one by one...
```

**You'll iterate**:
1. Run test -> "Did right arm go up?" -> "No, it went sideways"
2. Adjust mapping -> Run test again -> "Now it goes up!"
3. Repeat for each joint

### Step 3: Find the Servo Mapping (~2 hours)

**IMPORTANT: Servo ID mapping is NOT available online.**

I searched extensively:
- Hiwonder official docs (docs.hiwonder.com)
- TonyPi GitHub repo (servo_config.yaml only has calibration offsets, not joint names)
- Hiwonder wiki
- Hackster.io projects
- YouTube tutorials

**What we know for certain:**
- IDs 17 & 18 = hands (open/close)
- IDs 1-16 = body servos (exact mapping NOT published)

**How to get the mapping:**
1. **Ask teammate** - if they have the TonyPi PC software, it shows a visual servo layout
2. **Or discover manually** with this script:

```python
# test_servo_discovery.py
from HiwonderSDK import Board
import time

for servo_id in range(1, 19):
    print(f"\n=== Testing Servo {servo_id} ===")
    print("Watch what moves...")
    Board.setBusServoAngle(servo_id, 600, 500)  # Move to 600
    time.sleep(1)
    Board.setBusServoAngle(servo_id, 500, 500)  # Return to center
    joint = input(f"What moved? (e.g., 'r_shoulder', 'l_knee'): ")
    print(f"Servo {servo_id} = {joint}")
```

We need to discover which servo ID controls which joint:
```
TonyPi Servo IDs (must discover on Saturday):
+-------------------------------+
|  Head pan: ID ?               |
|  Head tilt: ID ?              |
|  R shoulder pitch: ID ?       |
|  R shoulder roll: ID ?        |
|  R elbow: ID ?                |
|  L shoulder pitch: ID ?       |
|  L shoulder roll: ID ?        |
|  L elbow: ID ?                |
|  Waist: ID ?                  |
|  R hip: ID ?                  |
|  R knee: ID ?                 |
|  R ankle: ID ?                |
|  L hip: ID ?                  |
|  L knee: ID ?                 |
|  L ankle: ID ?                |
+-------------------------------+
```

### Step 4: Safe Range Discovery (~1 hour)

For each servo, find the safe range before it hits a physical limit:
```python
# Find safe range for servo 1
for angle in range(0, 180, 10):
    print(f"Moving to {angle}...")
    Board.setBusServoAngle(1, angle, 300)
    response = input("Is it straining/stuck? (y/n): ")
    if response == 'y':
        print(f"Safe max for servo 1: {angle - 10}")
        break
```

### Step 5: First Dance Test (~30 min)

```python
# dance_test.py
# Play a short song, send pre-computed movements

import time
from HiwonderSDK import Board

# Load pre-generated dance from FACT
dance_sequence = load_dance("test_song_dance.json")

# Play!
for frame in dance_sequence:
    for servo_id, angle in frame['servos'].items():
        Board.setBusServoAngle(servo_id, angle, 16)  # 60fps = 16ms per frame
    time.sleep(0.016)
```

### Step 6: Iterate on Problems

**Common issues you'll hit:**

| What you see | What's wrong | Fix |
|--------------|--------------|-----|
| Arm goes wrong direction | Axis sign flipped | Multiply angle by -1 |
| Movement too jerky | Angles changing too fast | Add smoothing filter |
| Robot falls over | Leg movements too extreme | Reduce leg scaling to 20% |
| Servo makes grinding noise | Hitting physical limit | Reduce angle range |
| Movements lag behind music | Pipeline too slow | Pre-compute whole song |

### What You'll Be Telling Me

Throughout Saturday, you'll report things like:
- "Servo 3 is the elbow, not servo 5"
- "When I send 90deg, the arm only goes to 45deg"
- "The left leg keeps hitting the right leg"
- "It falls over when both arms go up"

And I'll adjust the code accordingly.

---

## 6. STRATEGY SUMMARY

### Primary Approach: Pre-trained FACT Model
**No training required - use Google's pre-trained checkpoints**

```
+---------------------------------------------------------------+
|  PIPELINE (all runs on laptop/Raspberry Pi)                   |
|                                                               |
|  Music File (.mp3/.wav)                                       |
|       |                                                       |
|       v                                                       |
|  Audio Feature Extraction (librosa)                           |
|       |                                                       |
|       v                                                       |
|  FACT Model Inference (pre-trained checkpoint)                |
|       |                                                       |
|       v                                                       |
|  SMPL Motion Sequence (24 joints)                             |
|       |                                                       |
|       v                                                       |
|  Motion Retargeting (SMPL -> TonyPi 18 servos)                |
|       |                                                       |
|       v                                                       |
|  Servo Commands -> HiwonderSDK -> Robot Dances!               |
+---------------------------------------------------------------+
```

### Fallback Approach: Rule-Based Beat Sync
If FACT pipeline fails or is too slow, use simpler approach:

```python
# Pseudo-code for fallback
while music_playing:
    audio_features = extract_features(mic_input)

    if audio_features.onset_detected:
        trigger_random_dance_move()

    if audio_features.beat_detected:
        sync_movement_to_beat()
```

Uses your existing aubio experience from the Arduino-Firmata project.

### Motion Retargeting: SMPL -> TonyPi

**TonyPi Joint Mapping** (GUESSES ONLY - must confirm Saturday):

Servo ID to joint mapping is **NOT published online**. The mapping below is a typical humanoid layout guess. We need to either:
1. Ask teammate (if they have TonyPi PC software)
2. Discover manually by testing each servo

```python
# PLACEHOLDER - WILL BE WRONG! Update after testing.
TONYPI_JOINTS = {
    # Right Arm (guesses)
    'r_shoulder_pitch': 1,
    'r_shoulder_roll': 2,
    'r_elbow': 3,
    # Left Arm (guesses)
    'l_shoulder_pitch': 4,
    'l_shoulder_roll': 5,
    'l_elbow': 6,
    # Head (guesses)
    'head_pan': 7,
    'head_tilt': 8,
    # Legs (guesses)
    'r_hip': 9, 'r_knee': 10, 'r_ankle': 11,
    'l_hip': 12, 'l_knee': 13, 'l_ankle': 14,
    # Torso (guess)
    'waist': 15,
    # Hands (CONFIRMED from docs)
    'r_hand': 17,  # CONFIRMED
    'l_hand': 18,  # CONFIRMED
}
```

**Safety**: Limit leg motion to 30% of full range to prevent falling.

---

## 7. Key Files & Resources

### Repos to Use
| Resource | URL | Purpose |
|----------|-----|---------|
| MINT | github.com/google-research/mint | FACT model (chosen) |
| AIST++ API | github.com/google/aistplusplus_api | Dataset loading |
| TonyPi SDK | github.com/Hiwonder/TonyPi | Robot control |
| GMR | github.com/YanjieZe/GMR | Motion retargeting |
| MDLT | github.com/meowatthemoon/MDLT | Alternative (no pretrained) |

### Documentation
- TonyPi: docs.hiwonder.com/projects/TonyPi/en/latest/
- AIST++: google.github.io/aistplusplus_dataset/

### Papers
1. AI Choreographer (FACT): arxiv.org/abs/2101.08779
2. MDLT: arxiv.org/abs/2403.15569
3. Robot Dance (DFKI): doi.org/10.1109/IROS47612.2022.9981462
4. Frontiers Survey 2025: doi.org/10.3389/fcomp.2025.1575667

---

## 8. Implementation Checklist (24-Hour Version)

### Friday - Setup & Pipeline (No Robot Needed)
- [ ] Clone MINT repo: `git clone https://github.com/liruilong940607/mint --recursive`
- [ ] Download pre-trained checkpoints from Google Drive
- [ ] Install dependencies (TensorFlow, librosa, etc.)
- [ ] Run FACT inference on a test audio file
- [ ] Verify output format (SMPL joint rotations)
- [ ] Write SMPL -> TonyPi retargeting script
- [ ] Clone TonyPi SDK: `git clone https://github.com/Hiwonder/TonyPi`
- [ ] Write servo command generator (angles -> PWM)
- [ ] Create mock/simulator test for servo sequences
- [ ] Prepare fallback beat-sync code with aubio

### Saturday - Hardware Integration
- [ ] Connect to TonyPi via SSH/serial
- [ ] Test individual servo control
- [ ] Run full pipeline with pre-recorded song
- [ ] Tune servo speed limits (prevent jerky motion)
- [ ] Adjust joint angle scaling (SMPL -> servo ranges)
- [ ] Test with multiple songs
- [ ] Add real-time mic input (if time permits)
- [ ] Final polish for rave

### First Commands to Run
```bash
# 1. Clone repos
git clone https://github.com/liruilong940607/mint --recursive
git clone https://github.com/Hiwonder/TonyPi

# 2. Download checkpoints (open in browser)
# https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm

# 3. Install MINT dependencies
cd mint
pip install -r requirements.txt

# 4. Test inference
python evaluator.py --config_path ./configs/fact_v5_deeper_t10_cm12.config --model_dir ./checkpoints
```

---

## 9. Risk Assessment (24-Hour Timeline)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FACT checkpoints don't work | Medium | High | Have fallback beat-sync code ready |
| SMPL->TonyPi retargeting wrong | High | Medium | Start with arms only, add legs later |
| Robot falls over | Medium | High | Limit leg motion to 30%, keep torso still |
| TonyPi SDK issues | Low | High | Team member may have examples |
| **Servo mapping unknown** | **High** | **High** | **Ask teammate for PC software screenshot, or test each servo manually** |
| Inference too slow for real-time | Medium | Low | Pre-compute dances for playlist |
| Servo overheating | Low | Medium | Add 2-second pauses between moves |
| Dependencies conflict | Medium | Medium | Use fresh conda environment |

### Critical Path Items
1. **FACT inference working** - if this fails, immediately switch to fallback
2. **Basic servo control** - test ASAP Saturday morning
3. **Not falling over** - conservative motion limits essential

---

## Sources

- [Hiwonder TonyPi Pro](https://www.hiwonder.com/products/tonypi-pro)
- [TonyPi GitHub SDK](https://github.com/Hiwonder/TonyPi)
- [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/)
- [AI Choreographer](https://google.github.io/aichoreographer/)
- [Google MINT](https://github.com/google-research/mint)
- [MDLT Paper](https://arxiv.org/abs/2403.15569)
- [Robot Dance Survey 2025](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1575667/full)
- [GMR Motion Retargeting](https://github.com/YanjieZe/GMR)
