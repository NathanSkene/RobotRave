# RobotRave Project Notes

## Overview

This project uses the FACT (Full Attention Cross-modal Transformer) model to generate dance motion from audio, then maps it to pre-scripted robot actions for playback on a TonyPi humanoid robot.

## Current Approach: Feature-Based Choreography

**We convert FACT-generated dance to sequences of pre-scripted .d6a actions** rather than sending raw servo values directly. This approach:

1. Extracts high-level motion features (activity, periodicity, symmetry, energy) from FACT output
2. Matches segments to the most similar actions from a library of 117 pre-scripted .d6a files
3. Generates a choreography timeline of action triggers
4. Plays actions in sequence synced with music

**See [CHOREOGRAPHY_README.md](CHOREOGRAPHY_README.md) for full pipeline documentation.**

### Quick Start

```bash
# Generate choreography from FACT dance
python3 generate_choreography.py --dance dance_house_tonypi.json --library library.json --output choreo.json

# Copy to robot
scp choreo.json play_choreography.py read_d6a.py pi@raspberrypi.local:~/

# Play with synced music (music on laptop, robot dances)
python3 play_synced.py
```

---

## Robot (TonyPi)
- **SSH**: `ssh pi@raspberrypi.local`
- **Password**: raspberrypi
- **Action files**: `/home/pi/TonyPi/ActionGroups`

## Lambda Labs Server
- **Server IP**: 132.145.180.105
- **SSH**: `ssh ubuntu@132.145.180.105`
- **FACT config**: `~/mint/configs/fact_v5_deeper_t10_cm12.config`

---

## Full Pipeline

```
Audio (.mp3)
    ↓ [Lambda Server - FACT model]
Motion (.npy)
    ↓ [Laptop - retarget_to_tonypi.py]
Dance JSON (servo values)
    ↓ [Laptop - generate_choreography.py]  ← CURRENT APPROACH
Choreography JSON (action triggers)
    ↓ [Robot - play_choreography.py]
Robot dances + Music plays
```

### Step 1: Generate Dance Motion (Lambda)
```bash
ssh ubuntu@132.145.180.105
python generate_dance.py --audio mHO2_house_120bpm.mp3 --output dance.npy --duration 30
```
**Output**: `dance_angles.npy` - SMPL euler angles [frames, 24, 3]

### Step 2: Retarget to Servos (Laptop)
```bash
scp ubuntu@132.145.180.105:~/dance_angles.npy .
python3 retarget_to_tonypi.py --input dance_angles.npy --output dance_tonypi.json
```
**Output**: `dance_tonypi.json` - servo pulse values

### Step 3: Generate Choreography (Laptop)
```bash
python3 generate_choreography.py --dance dance_tonypi.json --library library.json --output choreo.json
```
**Output**: `choreo.json` - action triggers with timestamps

### Step 4: Play on Robot
```bash
scp choreo.json play_choreography.py read_d6a.py pi@raspberrypi.local:~/
python3 play_synced.py  # Music on laptop, robot dances
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `generate_choreography.py` | Match FACT dance to library actions |
| `play_choreography.py` | Execute choreography on robot |
| `play_synced.py` | Synced playback (laptop audio + robot) |
| `build_library.py` | Pre-compute features for .d6a actions |
| `motion_features.py` | Extract motion features |
| `retarget_to_tonypi.py` | Convert SMPL angles to servo values |

---

## Lambda Labs Setup (DON'T pip install tensorflow!)

Lambda Stack has TensorFlow pre-configured for GPU. Installing tensorflow via pip **overwrites** the working system version.

```bash
# Only install missing packages
pip install --upgrade ml_dtypes einops tensorflow-graphics websockets soundfile scipy
```

### base_models.py Patch
```bash
sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' mint/mint/core/base_models.py
```

---

## FACT Model Architecture

### How `infer_auto_regressive()` Works

```python
def infer_auto_regressive(self, inputs, steps=1200):
    # inputs["audio_input"] must be shape [batch, steps + 240, 35]
    # inputs["motion_input"] is shape [batch, 120, 225] (seed motion)

    for i in range(steps):
        audio_input = inputs["audio_input"][:, i: i + 240]
        output = self.call({"motion_input": motion_input, "audio_input": audio_input})
        output = output[:, 0:1, :]
        motion_input = tf.concat([motion_input[:, 1:, :], output], axis=1)
```

**Critical**: Pass FULL audio sequence upfront. Model slides 240-frame window.

---

## Examples

The `examples/` folder contains:
- `dance_house_tonypi.json` - Sample FACT dance output (retargeted)
- `choreo.json` - Generated choreography
- `mHO2.mp3` - House music track used for generation
