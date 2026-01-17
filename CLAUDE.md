# RobotRave Project Notes

## Robot (TonyPi)
- **SSH**: `ssh pi@raspberrypi.local`
- **Password**: raspberrypi

## Lambda Labs Server
- **Server IP**: 132.145.180.105
- **Port**: 8765
- **WebSocket URL**: ws://132.145.180.105:8765
- **SSH**: `ssh ubuntu@132.145.180.105`

## Key Learnings

### Lambda Labs Setup (DON'T pip install tensorflow!)
Lambda Stack has TensorFlow pre-configured for GPU. Installing tensorflow via pip **overwrites**
the working system version and breaks GPU access.

**Correct setup:**
```bash
# Only install missing packages
pip install --upgrade ml_dtypes einops tensorflow-graphics websockets soundfile scipy
```

### Config Paths
Use absolute paths when running the server:
```bash
python ~/fact_server.py \
  --config ~/mint/configs/fact_v5_deeper_t10_cm12.config \
  --checkpoint ~/mint/checkpoints \
  --port 8765
```

### base_models.py Patch
Newer TensorFlow requires `name=` parameter in add_weight():
```bash
sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' mint/mint/core/base_models.py
```

## FACT Model Architecture

### How `infer_auto_regressive()` Works

```python
def infer_auto_regressive(self, inputs, steps=1200):
    # inputs["audio_input"] must be shape [batch, steps + 240, 35]
    # inputs["motion_input"] is shape [batch, 120, 225] (seed motion)

    for i in range(steps):
        # AUDIO: sliding window starting at position i
        audio_input = inputs["audio_input"][:, i: i + 240]

        # Forward pass: 120 motion + 240 audio = 360 frames
        output = self.call({"motion_input": motion_input, "audio_input": audio_input})
        output = output[:, 0:1, :]  # keep only first generated frame

        # MOTION: shift buffer left, append new frame (keeps last 120)
        motion_input = tf.concat([motion_input[:, 1:, :], output], axis=1)
```

**Critical requirements:**
1. **Audio**: Pass FULL sequence upfront. Model slides 240-frame window from position 0 to position `steps`
2. **Motion**: Only holds last 120 frames. Rolling buffer, not full history
3. **Each step**: Processes 360 total frames (120 motion + 240 audio)
4. **Memory per step**: ~5-10 MB for attention matrices. Should fit easily on any GPU

**DO NOT chunk by restarting audio at position 0** - this breaks the sliding window!

### Why OOM Happens
Not from single forward pass (only 360 frames). Caused by:
- TensorFlow eager mode accumulating tensors across thousands of iterations
- Python list `outputs.append()` holding tensor references
- No garbage collection between iterations

### Solution
1. Enable TensorFlow memory growth before any TF operations
2. Use single `infer_auto_regressive` call with full audio sequence
3. Add garbage collection after generation completes

## Dance Generation Pipeline

### Overview
```
Audio (mp3) → FACT Model (Lambda) → SMPL Angles → Retargeting → Servo JSON → Robot
```

### Step-by-Step

#### 1. Generate Dance Motion (Lambda Server)
```bash
ssh ubuntu@132.145.180.105
python generate_dance.py --audio mHO2_house_120bpm.mp3 --output dance_house.npy --duration 30
```
- **Input**: Audio file (mp3)
- **Output**:
  - `dance_house.npy` - Raw motion [1797, 225] (6 padding + 3 translation + 24×9 rotation matrices)
  - `dance_house_angles.npy` - Euler angles [1797, 24, 3] (24 SMPL joints × XYZ)

#### 2. Retarget to Robot (Local)
```bash
python retarget_to_tonypi.py --input dance_house_angles.npy --output dance_house_tonypi.json
```
- **Input**: SMPL euler angles
- **Output**: JSON with servo pulse values (0-1000 for bus servos, 1500-centered for PWM head servos)
- Applies smoothing, scales angles to servo ranges, clamps to safe limits

#### 3. Play on Robot
```bash
# Option A: Direct playback (on robot)
python play_dance.py --input dance_house_tonypi.json --audio song.mp3

# Option B: Stream via WebSocket
# On robot: python servo_receiver.py --port 8766
# On laptop: python fact_client.py --robot ws://192.168.149.1:8766
```

### JSON Format (Final Output)
```json
{
  "fps": 60,
  "n_frames": 1797,
  "frames": [
    {"time_ms": 16, "servos": {"1": 880, "2": 508, "pwm1": 1500, ...}},
    ...
  ]
}
```
- **No further processing** - servo values sent directly to `board.bus_servo_set_position()`
- Bus servos (body): IDs 1-16, pulse range 0-1000
- PWM servos (head): "pwm1", "pwm2", pulse range ~1000-2000

### Successful Test
- Audio: `mHO2_house_120bpm.mp3` (AIST++ House track, 120 BPM)
- Generated: 1797 frames (~30 seconds at 60fps)
- Output files: `dance_house_angles.npy`, `dance_house_tonypi.json`
