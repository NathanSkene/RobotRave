# Old Approaches & Notes

This document contains notes on approaches we tried before settling on the feature-based choreography system.

## Direct Servo Playback (Deprecated)

Our original approach was to translate FACT's SMPL output directly to servo commands. This had stability issues because:
- FACT uses full 0-1000 servo range; robot movements are safer in restricted ranges
- Small timing errors compound into jerky motion
- No guarantee the generated poses are physically stable for the robot

### How We Translated 24 Joints → 16 Servos

SMPL describes a human body with 24 joints, but TonyPi only has 16 servos (+ 2 PWM for head).

**What we kept:**
| SMPL Joint | TonyPi Servo | Why |
|------------|--------------|-----|
| head | pwm1, pwm2 | Direct match |
| r_shoulder | r_shoulder_pitch (16) | Forward/back rotation |
| r_elbow | r_elbow_pitch (11) | Bend angle |
| l_shoulder | l_shoulder_pitch (8) | Forward/back rotation |
| l_elbow | l_elbow_pitch (6) | Bend angle |
| r_hip | r_hip_pitch/roll (12,13) | 2 DOF |
| r_knee | r_knee_pitch (14) | Direct match |
| r_ankle | r_ankle_pitch/roll (10,9) | 2 DOF |
| (same for left leg) | ... | ... |

**What we lost:**
- Wrist rotation (no wrist servos)
- Spine/torso twist (no torso servo)
- Finger movements (no hand servos)

**The conversion process:**
```python
# SMPL gives us: right_shoulder rotation = [0.5, -0.3, 0.2] radians (x, y, z)

# 1. Extract the relevant axis (x = pitch = forward/back)
angle = smpl_right_shoulder[0]  # 0.5 radians

# 2. Convert radians to servo pulse (0-1000 scale, 500 = center)
pulse = 500 + (angle * 191)  # 191 pulse units per radian

# 3. Clamp to safe range
pulse = clamp(pulse, 200, 800)

# 4. Send to robot
servo.set_position(16, pulse)  # Servo 16 = right shoulder pitch
```

**Safety scaling:**
- Arms and head: 100% of SMPL motion
- Legs: 30% of SMPL motion (prevent falls)

---

## Beat-Sync Dancing (Fallback)

Rule-based beat-synchronized dancing using `aubio` for real-time beat/onset detection:

```bash
# Dance to an audio file
python beat_sync_dance.py --audio your_song.wav

# Live microphone input
python beat_sync_dance.py --live
```

This doesn't use FACT at all - it just triggers pre-defined moves on beat events.

---

## Smart Style-Adaptive Dancing

Detects music genre and adapts dance style:

```bash
python smart_dance.py --live
python smart_dance.py --audio song.wav
python smart_dance.py --demo --simulate
```

---

## Real-Time WebSocket Streaming

We tried streaming FACT output in real-time via WebSocket:

```bash
# On cloud GPU server
python fact_server.py --port 8765

# On robot
python fact_client.py --server ws://<server-ip>:8765
```

This had latency issues and was replaced by the batch choreography approach.

---

## Legacy Python API

```python
from tony_pro import SERVO, MOVES, TonyProController

# Access servo IDs (NOTE: these may be outdated)
SERVO.HEAD_PITCH  # pwm1
SERVO.R_SHOULDER  # 16
SERVO.L_ELBOW     # 6

# Use pre-defined moves
controller = TonyProController(simulate=False)
controller.execute_move('arms_up')
controller.go_neutral()
```

---

## Why We Moved to Choreography

The feature-based choreography approach solved these problems:
1. **Stability**: Pre-scripted .d6a actions are tested and safe
2. **Domain gap**: Features are normalized, not raw servo values
3. **Semantic matching**: "both have oscillating arms" instead of "servo 8 values match"
4. **Speed independent**: Features capture motion type, not exact timing

See [CHOREOGRAPHY_README.md](CHOREOGRAPHY_README.md) for the current approach.
