# RobotRave

Make a Tony Pro humanoid robot dance autonomously to music at a robot rave.

## The Approach

We use Google's **FACT model** (Full-Attention Cross-modal Transformer) from their [AI Choreographer](https://google.github.io/aichoreographer/) research to generate human dance movements from music. The model was trained on the AIST++ dataset containing 1,408 dance sequences across 10 genres.

```
Music Audio → FACT Model → SMPL Human Motion → Retarget → Tony Pro Servos → Robot Dances!
```

### Why FACT?
We evaluated three approaches:
- **MDLT** (Music to Dance Language Translation) - No pre-trained weights, requires ~140 hours training
- **DFKI Robot Dance** - Code not publicly available
- **FACT** (Google) - Pre-trained checkpoints available, state-of-the-art quality

FACT outputs motion in SMPL format (24 joints), which we retarget to Tony Pro's 16 servos.

### Fallback: Beat-Sync Dancing
If FACT inference is too slow or fails, we have rule-based beat-synchronized dancing using `aubio` for real-time beat/onset detection with pre-defined dance moves.

---

## Tony Pro Servo Mapping

After discovery, the correct 16-servo mapping:

| ID | Joint | Group |
|----|-------|-------|
| 1 | right_hip_roll | right_leg |
| 2 | right_hip_yaw | right_leg |
| 3 | right_knee_pitch | right_leg |
| 4 | right_hip_pitch | right_leg |
| 5 | right_elbow_pitch | right_arm |
| 6 | right_ankle_pitch | right_leg |
| 7 | right_shoulder_pitch | right_arm |
| 8 | head_pitch | head |
| 9 | left_hip_roll | left_leg |
| 10 | left_hip_yaw | left_leg |
| 11 | left_knee_pitch | left_leg |
| 12 | left_hip_pitch | left_leg |
| 13 | left_elbow_pitch | left_arm |
| 14 | left_ankle_pitch | left_leg |
| 15 | left_shoulder_pitch | left_arm |
| 16 | head_yaw | head |

**Summary:**
- Head: 2 DOF (pitch + yaw)
- Each arm: 2 DOF (shoulder pitch + elbow pitch)
- Each leg: 5 DOF (hip roll/yaw/pitch + knee + ankle)

---

## Project Structure

```
RobotRave/
├── tony_pro.py              # Central servo config & controller module
├── tony_pro_config.json     # Servo mapping in JSON format
├── beat_sync_dance.py       # Fallback: beat-triggered dance moves
├── smart_dance.py           # Style-adaptive dancing (chill/pop/edm/hiphop)
├── play_dance.py            # Play pre-generated dance sequences
├── retarget_to_tonypi.py    # SMPL → servo command conversion
├── generate_dance.py        # Run FACT model on audio files
├── fact_server.py           # Cloud GPU server for FACT inference
├── fact_client.py           # Robot client for cloud inference
├── discover_servos.py       # Interactive servo discovery tool
├── RESEARCH.md              # Detailed research notes & approach analysis
├── CLOUD_SETUP.md           # GPU cloud deployment guide
└── (submodules - not tracked)
    ├── mint/                # Google FACT model
    ├── TonyPi/              # Hiwonder SDK
    └── MDLT/                # Alternative approach (unused)
```

---

## Saturday Checklist

### 1. Initial Connection (~15 min)
```bash
# SSH into the robot's Raspberry Pi
ssh pi@<robot-ip-address>

# Copy project files to robot
scp -r *.py tony_pro_config.json pi@<robot-ip>:~/RobotRave/
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
