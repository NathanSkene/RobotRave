# FACT Choreography Pipeline

Generate robot dance choreography from audio using the FACT model.

## Full Pipeline Overview

```
Audio (.mp3)
    ↓ [Lambda Server - FACT model]
Motion (.npy)
    ↓ [Laptop - retargeting]
Dance JSON (servo values)
    ↓ [Laptop - feature matching]
Choreography JSON (action triggers)
    ↓ [Robot - playback]
Robot dances + Music plays
```

---

## Prerequisites

**Lambda Server (132.145.180.105):**
- FACT model and checkpoints installed
- `generate_dance.py` script

**Laptop:**
```bash
pip install sounddevice soundfile scipy numpy
```

**Robot:**
```bash
pip install scipy
```

**Passwordless SSH to robot (one-time):**
```bash
ssh-copy-id pi@raspberrypi.local
# Password: raspberrypi
```

---

## Step 1: Generate Dance Motion (Lambda Server)

SSH into Lambda and run FACT model on your audio:

```bash
ssh ubuntu@132.145.180.105

cd ~
python generate_dance.py \
    --audio mHO2_house_120bpm.mp3 \
    --output dance_house.npy \
    --duration 30
```

**Outputs:**
- `dance_house.npy` - Raw SMPL motion [frames, 225]
- `dance_house_angles.npy` - Euler angles [frames, 24, 3]

**Copy angles file to laptop:**
```bash
# Run from laptop
scp ubuntu@132.145.180.105:~/dance_house_angles.npy /Users/nathanskene/Projects/RobotRave/
```

---

## Step 2: Retarget to Robot Servos (Laptop)

Convert SMPL angles to TonyPi servo values:

```bash
cd /Users/nathanskene/Projects/RobotRave

python3 retarget_to_tonypi.py \
    --input dance_house_angles.npy \
    --output dance_house_tonypi.json
```

**Output:** `dance_house_tonypi.json` (servo pulse values 0-1000)

---

## Step 3: Build Action Library (One-Time)

Extract motion features from all .d6a action files:

```bash
python3 build_library.py \
    --input ./action_groups \
    --output library.json \
    --summary
```

**Output:** `library.json` (117 actions with pre-computed features)

*Only needs to be run once, unless you add new .d6a actions.*

---

## Step 4: Generate Choreography (Laptop)

Match FACT dance segments to library actions:

```bash
python3 generate_choreography.py \
    --dance dance_house_tonypi.json \
    --library library.json \
    --output choreo.json \
    --verbose
```

**Input:** `dance_house_tonypi.json` (from Step 2)
**Output:** `choreo.json` (action triggers with timestamps)

**Options:**
- `--method adaptive` (default) - finds natural motion boundaries
- `--method fixed_window --window-ms 1000` - fixed 1-second segments
- `--stats` - print statistics

---

## Step 5: Copy Files to Robot

```bash
scp choreo.json play_choreography.py read_d6a.py pi@raspberrypi.local:~/
```

---

## Step 6: Play on Robot

**Option A: Synced playback (music on laptop, robot dances)**

```bash
python3 play_synced.py
```

Options:
- `--audio mHO2.mp3` - specify audio file
- `--countdown 5` - longer countdown
- `--no-audio` - robot only
- `--test-ssh` - test connection

**Option B: Everything on robot**

```bash
ssh pi@raspberrypi.local

python3 play_choreography.py \
    --input choreo.json \
    --action-dir /home/pi/TonyPi/ActionGroups \
    --audio mHO2.mp3 \
    --verbose
```

**Option C: Preview only (no movement)**

```bash
python3 play_choreography.py --input choreo.json --preview --action-dir ./action_groups
```

---

## Quick Reference: Full Pipeline Commands

```bash
# === LAMBDA SERVER ===
ssh ubuntu@132.145.180.105
python generate_dance.py --audio mHO2_house_120bpm.mp3 --output dance.npy --duration 30
exit

# === LAPTOP ===
cd /Users/nathanskene/Projects/RobotRave

# Get the angles file from Lambda
scp ubuntu@132.145.180.105:~/dance_angles.npy .

# Retarget to robot servos
python3 retarget_to_tonypi.py --input dance_angles.npy --output dance_tonypi.json

# Generate choreography (library.json already exists)
python3 generate_choreography.py --dance dance_tonypi.json --library library.json --output choreo.json

# Copy to robot
scp choreo.json play_choreography.py read_d6a.py pi@raspberrypi.local:~/

# Play with synced music
python3 play_synced.py --audio mHO2.mp3
```

---

## File Reference

| File | Location | Purpose |
|------|----------|---------|
| `mHO2_house_120bpm.mp3` | Lambda | Input audio for FACT |
| `dance_angles.npy` | Lambda → Laptop | SMPL euler angles output |
| `dance_tonypi.json` | Laptop | Retargeted servo values |
| `library.json` | Laptop | Pre-computed action features |
| `choreo.json` | Laptop → Robot | Generated choreography |
| `mHO2.mp3` | Laptop + Robot | Music for playback |

**Scripts:**

| Script | Location | Purpose |
|--------|----------|---------|
| `generate_dance.py` | Lambda | Run FACT model |
| `retarget_to_tonypi.py` | Laptop | Convert angles to servos |
| `build_library.py` | Laptop | Build action library |
| `generate_choreography.py` | Laptop | Match dance to actions |
| `play_synced.py` | Laptop | Synced laptop+robot playback |
| `play_choreography.py` | Robot | Execute choreography |
| `motion_features.py` | Laptop | Feature extraction |
| `match_motion.py` | Laptop | Similarity scoring |
| `read_d6a.py` | Both | Read .d6a action files |

---

## Troubleshooting

**Lambda SSH fails:**
```bash
ssh ubuntu@132.145.180.105
```

**Robot SSH fails:**
```bash
ssh-copy-id pi@raspberrypi.local
# Password: raspberrypi
```

**No audio on laptop:**
```bash
pip install sounddevice soundfile
```

**Action not found:**
- Robot path: `/home/pi/TonyPi/ActionGroups`
- Local path: `./action_groups`

**Test feature extraction:**
```bash
python3 motion_features.py ./action_groups/wave.d6a
```

---

## Server Details

**Lambda Labs Server:**
- IP: 132.145.180.105
- User: ubuntu
- FACT config: `~/mint/configs/fact_v5_deeper_t10_cm12.config`

**TonyPi Robot:**
- Host: raspberrypi.local
- User: pi
- Password: raspberrypi
- Action files: `/home/pi/TonyPi/ActionGroups`
