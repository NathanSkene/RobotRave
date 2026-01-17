#!/bin/bash
# RobotRave Setup Script
# Run this after cloning the repo to download all required dependencies

set -e  # Exit on error

echo "==================================="
echo "RobotRave Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "README.md" ] || ! grep -q "RobotRave" README.md 2>/dev/null; then
    echo -e "${RED}Error: Run this script from the RobotRave directory${NC}"
    exit 1
fi

echo ""
echo "Step 1: Cloning MINT (Google FACT model)..."
echo "-------------------------------------------"
if [ -d "mint" ] && [ -f "mint/README.md" ]; then
    echo -e "${GREEN}MINT already cloned${NC}"
else
    echo "Cloning https://github.com/google-research/mint..."
    git clone --recursive https://github.com/google-research/mint.git
    echo -e "${GREEN}Done${NC}"
fi

echo ""
echo "Step 2: Installing gdown (Google Drive downloader)..."
echo "------------------------------------------------------"
if command -v gdown &> /dev/null; then
    echo -e "${GREEN}gdown already installed${NC}"
else
    pip install gdown
    echo -e "${GREEN}Done${NC}"
fi

echo ""
echo "Step 3: Downloading FACT model checkpoints..."
echo "----------------------------------------------"
CHECKPOINT_DIR="mint/checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/ckpt-214501.index"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo -e "${GREEN}Checkpoints already downloaded${NC}"
else
    mkdir -p "$CHECKPOINT_DIR"
    echo "Downloading from Google Drive..."
    echo "Source: https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm"

    # Download the entire folder
    gdown --folder "https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm" -O "$CHECKPOINT_DIR"

    if [ -f "$CHECKPOINT_FILE" ]; then
        echo -e "${GREEN}Done${NC}"
    else
        echo -e "${RED}Warning: Download may have failed. Check $CHECKPOINT_DIR${NC}"
        echo "You can manually download from:"
        echo "https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm"
    fi
fi

echo ""
echo "Step 4: Installing Python dependencies..."
echo "------------------------------------------"
echo "For robot (Raspberry Pi):"
echo "  pip install numpy aubio sounddevice soundfile"
echo ""
echo "For development machine:"
echo "  pip install numpy aubio sounddevice soundfile scipy"
echo ""
echo "For FACT model (GPU server):"
echo "  pip install tensorflow numpy scipy websockets soundfile"
echo ""

read -p "Install development dependencies now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install numpy aubio sounddevice soundfile scipy
    echo -e "${GREEN}Done${NC}"
fi

echo ""
echo "==================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "==================================="
echo ""
echo "Checkpoint files are in: mint/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Connect to robot WiFi: HW-67B69FFB"
echo "  2. SSH: ssh pi@192.168.149.1 (password: raspberrypi)"
echo "  3. Test: python beat_sync_dance.py --test --simulate"
echo ""
echo "See README.md for full instructions."
