# FACT Server Cloud Deployment

## Quick Start (AWS - You Have Credits!)

### 1. Launch GPU Instance
```bash
# Using AWS CLI (or use Console)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.2xlarge \
  --key-name your-key-name \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fact-dance-server}]'
```

**Recommended instance types:**
| Instance | GPU | VRAM | Cost/hr | With $5000 credits |
|----------|-----|------|---------|-------------------|
| g5.xlarge | 1x A10G | 24GB | $1.00 | 5000 hours |
| g5.2xlarge | 1x A10G | 24GB | $1.21 | 4100 hours |
| p4d.24xlarge | 8x A100 | 320GB | $32.77 | 152 hours |

### 2. Configure Security Group
Allow inbound traffic on port 8765:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 8765 \
  --cidr 0.0.0.0/0
```

### 3. SSH and Setup (Easy Way)
```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>

# One-liner setup (downloads and runs deploy script)
curl -sL https://raw.githubusercontent.com/YOUR_REPO/RobotRave/main/deploy_server.sh | bash
```

### 3b. SSH and Setup (Manual Way)
```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv venv
source venv/bin/activate

pip install tensorflow[and-cuda] numpy scipy websockets soundfile

# Clone repo and MINT
git clone https://github.com/google-research/mint.git
# Download checkpoints to mint/checkpoints/

# Copy fact_server.py and retarget_to_tonypi.py from your local machine
# scp fact_server.py retarget_to_tonypi.py ubuntu@<ip>:~/

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Start server
python fact_server.py --port 8765
```

### 4. Connect from Robot
```bash
python fact_client.py --server ws://<aws-public-ip>:8765
```

---

## Alternative: Lambda Labs (Simpler, No Credits Needed)

### 1. Rent GPU Instance
Go to [Lambda Labs](https://lambdalabs.com/) and rent:
- **1x H100 PCIe** (~$2/hour) - Best price/performance
- Or **1x A100** (~$1.10/hour) - Still good

### 2. SSH into Instance
```bash
ssh ubuntu@<instance-ip>
```

### 3. Set Up Server
```bash
# Clone the repo (or scp your files)
git clone https://github.com/YOUR_REPO/RobotRave.git
cd RobotRave

# Set up Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow numpy scipy websockets soundfile

# Clone MINT and download checkpoints
git clone https://github.com/google-research/mint.git
# Download checkpoints from Google Drive to mint/checkpoints/

# Start server
python fact_server.py --port 8765
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

## Alternative: AWS (More Complex)

```bash
# Launch p4d.24xlarge (8x A100) or g5.xlarge (A10G)
aws ec2 run-instances \
  --image-id ami-0xxx \
  --instance-type p4d.24xlarge \
  --key-name your-key

# SSH and setup (same as above)
```

---

## Expected Performance

| GPU | AWS Instance | Cost/hr | Est. FPS | Real-time? |
|-----|--------------|---------|----------|------------|
| CPU (i7) | - | - | 0.5 | ❌ |
| T4 | g4dn.xlarge | $0.52 | ~5 | ❌ |
| **A10G** | **g5.xlarge** | **$1.00** | **~15-25** | **✅ Probably** |
| L4 | g6.xlarge | $0.80 | ~20 | ✅ Probably |
| A100 | p4d.24xlarge | $32.77 | ~50+ | ✅ Yes |
| H100 | p5.48xlarge | $98.32 | ~80+ | ✅ Yes |

**Recommendation: g5.xlarge or g5.2xlarge** - Best value with your credits!

*Note: These are estimates. We generate 10 frames per inference call for efficiency.*

---

## Optimization Tips

1. **Batch Processing**: Generate 10-20 frames per inference call
2. **Model Quantization**: Use TensorRT or XLA for faster inference
3. **Reduce Motion Buffer**: Smaller seed = faster inference

---

## Troubleshooting

### "Connection Refused"
- Check firewall allows port 8765
- Verify server is running: `ps aux | grep fact_server`

### Slow Inference
- Verify GPU is detected: `nvidia-smi`
- Check TensorFlow sees GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### High Latency
- Use server closer to your location
- Reduce audio chunk size in client
- Use wired internet if possible

---

## Cost Estimate for Rave

- 4-hour rave × $2/hour (H100) = **$8 total**
- That's cheaper than a pizza!
