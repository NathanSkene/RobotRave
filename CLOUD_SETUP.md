# FACT Server Cloud Deployment

## Quick Start: Lambda Labs (Recommended)

Lambda Labs is the easiest option because their "Lambda Stack" comes with TensorFlow pre-configured for GPU use. **Don't fight it - use it!**

### 1. Rent GPU Instance
Go to [Lambda Labs](https://lambdalabs.com/) and rent:
- **1x H100 PCIe** (~$2/hour) - Best performance
- **1x A100** (~$1.10/hour) - Good balance

### 2. SSH into Instance
```bash
ssh ubuntu@<instance-ip>
```

### 3. Run Setup Script (Easy Way)
```bash
# One-liner setup for Lambda Labs
curl -sL https://raw.githubusercontent.com/YOUR_REPO/RobotRave/main/lambda_setup.sh | bash
```

### 3b. Manual Setup (If You Need to Understand)
```bash
# IMPORTANT: Do NOT create a virtual environment on Lambda Labs!
# The system Python has TensorFlow pre-configured for GPU.

# Install ONLY the missing packages (NOT tensorflow!)
pip install --upgrade ml_dtypes einops tensorflow-graphics websockets soundfile scipy

# Clone MINT repository
cd ~
git clone --depth 1 https://github.com/google-research/mint.git

# Apply Keras compatibility fix (add_weight needs name= parameter)
sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' mint/mint/core/base_models.py
sed -i 's/self.add_weight(\s*"cls_token"/self.add_weight(name="cls_token"/g' mint/mint/core/base_models.py

# Compile protocol buffers
pip install grpcio-tools
cd mint && python -m grpc_tools.protoc -I. --python_out=. ./mint/protos/*.proto && cd ~

# Download checkpoints (from Google Drive)
# https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm
# Upload to ~/mint/checkpoints/

# Verify GPU is detected
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Start server (use ABSOLUTE paths!)
python ~/fact_server.py \
  --config ~/mint/configs/fact_v5_deeper_t10_cm12.config \
  --checkpoint ~/mint/checkpoints \
  --port 8765
```

### 4. Open Firewall
Lambda Labs: Go to dashboard → Firewall → Allow port 8765

### 5. Connect from Robot/Mac
```bash
# Test connection
python fact_client.py --server ws://<instance-ip>:8765 --test

# Run live!
python fact_client.py --server ws://<instance-ip>:8765 --simulate
```

---

## What Can Go Wrong (Troubleshooting)

### "ModuleNotFoundError: No module named 'ml_dtypes'"
```bash
pip install --upgrade ml_dtypes
```

### "TypeError: add_weight() got multiple values for argument 'name'"
The MINT base_models.py uses old Keras syntax. Apply the fix:
```bash
sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' mint/mint/core/base_models.py
```

### "No GPU detected" / TensorFlow not seeing GPU
**On Lambda Labs, this usually means you pip installed tensorflow and overwrote the working system version!**

Solution:
```bash
# Remove the broken tensorflow
pip uninstall tensorflow tensorflow-gpu -y

# Reinstall from Lambda's system packages
sudo apt install --reinstall python3-tensorflow 2>/dev/null || true

# Or just start a fresh instance and DON'T pip install tensorflow
```

### "FileNotFoundError: mint/configs/..."
The config path is relative. Either:
1. Run from the directory containing `mint/`: `cd ~ && python fact_server.py`
2. Use absolute paths: `--config ~/mint/configs/...`

### "protobuf version conflict"
```bash
pip install --upgrade protobuf
```

### "numpy/scipy version issues"
```bash
pip install --upgrade numpy scipy
```

### "Connection Refused" from client
- Check firewall allows port 8765 (Lambda dashboard → Firewall)
- Verify server is running: `ps aux | grep fact_server`
- Check the server logs for errors

### Slow Inference (< 5 FPS)
- Verify GPU is detected: `nvidia-smi`
- Check TensorFlow sees GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- If empty list, see "No GPU detected" above

---

## Alternative: AWS (More Setup Required)

AWS requires more configuration but you may have credits to use.

### 1. Launch GPU Instance
```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.2xlarge \
  --key-name your-key-name \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx
```

**Recommended instance types:**
| Instance | GPU | VRAM | Cost/hr | With $5000 credits |
|----------|-----|------|---------|-------------------|
| g5.xlarge | 1x A10G | 24GB | $1.00 | 5000 hours |
| g5.2xlarge | 1x A10G | 24GB | $1.21 | 4100 hours |
| p4d.24xlarge | 8x A100 | 320GB | $32.77 | 152 hours |

### 2. Configure Security Group
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 8765 \
  --cidr 0.0.0.0/0
```

### 3. SSH and Setup
```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Install dependencies (AWS Deep Learning AMI may need these)
sudo apt update && sudo apt install -y python3-pip git
pip install tensorflow[and-cuda] numpy scipy websockets soundfile einops tensorflow-graphics

# Clone MINT and setup (same as Lambda Labs steps above)
git clone https://github.com/google-research/mint.git
# ... rest of setup ...
```

### 4. Connect from Robot
```bash
python fact_client.py --server ws://<aws-public-ip>:8765
```

---

## Alternative: Google Colab (Free but Limited)

Colab has free GPUs but:
- No incoming WebSocket connections allowed
- Need to use ngrok tunnel (adds latency)
- GPU may disconnect after ~30 min

### Colab Notebook Setup
```python
# Install dependencies
!pip install websockets pyngrok

# Clone and setup
!git clone https://github.com/google-research/mint.git
# ... setup code ...

# Start ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8765, "tcp")
print(f"Connect to: {public_url}")

# Run server
!python fact_server.py --port 8765
```

---

## Alternative: RunPod (~$1.50/hr for A100)

1. Go to [RunPod](https://runpod.io/)
2. Deploy "PyTorch" template with A100
3. SSH in and follow Lambda Labs steps above
4. Use the provided public IP

---

## Expected Performance

| GPU | Provider | Cost/hr | Est. FPS | Real-time? |
|-----|----------|---------|----------|------------|
| CPU (i7) | - | - | 0.5 | ❌ |
| T4 | Colab/AWS | $0.52 | ~5 | ❌ |
| **A10G** | **AWS g5** | **$1.00** | **~15-25** | **✅ Probably** |
| L4 | AWS g6 | $0.80 | ~20 | ✅ Probably |
| A100 | Lambda/AWS | $1.10-32 | ~50+ | ✅ Yes |
| H100 | Lambda | ~$2.00 | ~80+ | ✅ Yes |

**Recommendation: Lambda Labs H100 or A100** - Best value with simplest setup!

*Note: These are estimates. We generate 10 frames per inference call for efficiency.*

---

## Optimization Tips

1. **Batch Processing**: Generate 10-20 frames per inference call
2. **Model Quantization**: Use TensorRT or XLA for faster inference
3. **Reduce Motion Buffer**: Smaller seed = faster inference

---

## Cost Estimate for Rave

- 4-hour rave × $2/hour (H100) = **$8 total**
- That's cheaper than a pizza!
