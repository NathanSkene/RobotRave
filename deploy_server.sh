#!/bin/bash
# Deploy FACT Dance Server to cloud GPU
# Works on AWS, Lambda Labs, RunPod, or any Ubuntu GPU instance
#
# Usage:
#   # Copy this script to your cloud instance, then run:
#   chmod +x deploy_server.sh
#   ./deploy_server.sh
#
# Or run remotely:
#   ssh user@gpu-server 'bash -s' < deploy_server.sh

set -e

echo "========================================"
echo "🤖 FACT Dance Server Deployment"
echo "========================================"

# Check if running on GPU instance
echo ""
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo "✅ GPU detected!"
else
    echo "⚠️  No nvidia-smi found. GPU may not be available."
    echo "   Continuing anyway (will be slow on CPU)..."
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git wget

# Create project directory
echo ""
echo "Setting up project directory..."
mkdir -p ~/RobotRave
cd ~/RobotRave

# Create virtual environment
echo ""
echo "Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo ""
echo "Installing Python packages..."
pip install --upgrade pip -q
pip install tensorflow[and-cuda] numpy scipy websockets soundfile protobuf grpcio-tools -q

# Check TensorFlow GPU
echo ""
echo "Verifying TensorFlow GPU support..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ TensorFlow sees {len(gpus)} GPU(s): {gpus}')
else:
    print('⚠️  No GPU detected by TensorFlow')
"

# Clone MINT if not present
if [ ! -d "mint" ]; then
    echo ""
    echo "Cloning MINT repository..."
    git clone https://github.com/google-research/mint.git
fi

# Compile protos
echo ""
echo "Compiling protocol buffers..."
cd mint
python -m grpc_tools.protoc -I. --python_out=. ./mint/protos/*.proto 2>/dev/null || true
cd ..

# Download checkpoints if not present
if [ ! -d "mint/checkpoints" ] || [ -z "$(ls -A mint/checkpoints 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  Checkpoints not found!"
    echo "   Please download from Google Drive:"
    echo "   https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm"
    echo ""
    echo "   Then copy to: ~/RobotRave/mint/checkpoints/"
    echo ""
    read -p "Press Enter once checkpoints are in place..."
fi

# Create the server script if not present
if [ ! -f "fact_server.py" ]; then
    echo ""
    echo "⚠️  fact_server.py not found!"
    echo "   Please copy fact_server.py and retarget_to_tonypi.py to ~/RobotRave/"
    read -p "Press Enter once files are in place..."
fi

# Fix Keras compatibility if needed
echo ""
echo "Applying Keras compatibility fix..."
BASEFILE="mint/mint/core/base_models.py"
if [ -f "$BASEFILE" ]; then
    sed -i 's/self.add_weight(\s*"position_embedding"/self.add_weight(name="position_embedding"/g' "$BASEFILE" 2>/dev/null || true
fi

# Get public IP
echo ""
echo "Getting public IP..."
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

echo ""
echo "========================================"
echo "✅ Deployment Complete!"
echo "========================================"
echo ""
echo "To start the server:"
echo "  cd ~/RobotRave"
echo "  source venv/bin/activate"
echo "  python fact_server.py --port 8765"
echo ""
echo "Connect from robot/client:"
echo "  python fact_client.py --server ws://${PUBLIC_IP}:8765"
echo ""
echo "Make sure port 8765 is open in your firewall/security group!"
echo ""

# Optionally start server
read -p "Start server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting server..."
    python fact_server.py --port 8765
fi
