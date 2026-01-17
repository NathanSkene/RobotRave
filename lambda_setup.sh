#!/bin/bash
#
# Lambda Labs Quick Setup for FACT Dance Server
#
# Usage: curl -sL https://raw.githubusercontent.com/YOUR_REPO/RobotRave/main/lambda_setup.sh | bash
#
# IMPORTANT: Lambda Labs has TensorFlow pre-installed with GPU support.
# Do NOT pip install tensorflow - it will break GPU access!
#

set -e

echo "========================================"
echo "FACT Dance Server - Lambda Labs Setup"
echo "========================================"
echo ""
echo "Using Lambda's pre-installed TensorFlow (GPU-enabled)"
echo ""

cd ~

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Install ONLY the missing packages (NOT tensorflow!)
echo ""
echo "Installing missing packages (keeping system TensorFlow)..."
pip install --upgrade ml_dtypes einops tensorflow-graphics websockets soundfile scipy grpcio-tools

# Clone MINT
if [ ! -d "mint" ]; then
    echo ""
    echo "Cloning MINT repository..."
    git clone --depth 1 https://github.com/google-research/mint.git
else
    echo ""
    echo "MINT already cloned"
fi

# Compile protos
echo ""
echo "Compiling protocol buffers..."
cd ~/mint
python -m grpc_tools.protoc -I. --python_out=. ./mint/protos/*.proto 2>/dev/null || true
cd ~

# Fix Keras compatibility (add_weight needs name= parameter in newer TF)
echo ""
echo "Applying Keras compatibility fix..."
sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' ~/mint/mint/core/base_models.py
sed -i 's/self.add_weight(\s*"cls_token"/self.add_weight(name="cls_token"/g' ~/mint/mint/core/base_models.py

# Verify TensorFlow sees GPU
echo ""
echo "Verifying TensorFlow GPU access..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'SUCCESS: {len(gpus)} GPU(s) detected')
    for g in gpus:
        print(f'  - {g}')
else:
    print('ERROR: No GPUs detected!')
    print('If you pip installed tensorflow, that broke GPU access.')
    print('Start a fresh instance and run this script again.')
    exit(1)
"

# Check for checkpoints
echo ""
if [ ! -d "mint/checkpoints" ] || [ -z "$(ls -A mint/checkpoints 2>/dev/null)" ]; then
    echo "CHECKPOINTS NEEDED!"
    echo ""
    echo "Download from:"
    echo "  https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm"
    echo ""
    echo "Upload to server:"
    echo "  scp checkpoints.tar.gz ubuntu@\$(curl -s ifconfig.me):~/"
    echo ""
    echo "Then extract:"
    echo "  tar -xzf checkpoints.tar.gz -C ~/mint/"
    echo ""
else
    echo "Checkpoints found!"
fi

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start server:"
echo "  python ~/fact_server.py --port 8765"
echo ""
echo "From your Mac/robot:"
echo "  python fact_client.py --server ws://${PUBLIC_IP}:8765"
echo ""
echo "Public IP: ${PUBLIC_IP}"
echo ""
echo "IMPORTANT: Open firewall in Lambda dashboard"
echo "  Dashboard -> Firewall -> Allow port 8765"
echo ""
