# RobotRave Project Notes

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
