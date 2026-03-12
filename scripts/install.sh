#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "DySSNet Installation Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓${NC} Python version: $python_version"
else
    echo -e "${RED}✗${NC} Python $python_version is too old. Please use Python >= 3.8"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓${NC} CUDA version: $cuda_version"
else
    echo -e "${YELLOW}⚠${NC}  Warning: CUDA not found. Mamba-SSM requires CUDA for compilation."
    echo "   Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠${NC}  Warning: Not running in a conda environment."
    echo "   It's recommended to create a conda environment first:"
    echo "   conda create -n dyssnet python=3.10"
    echo "   conda activate dyssnet"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "========================================="
echo "Installing PyTorch..."
echo "========================================="
read -p "Select CUDA version (118/121/cpu): " cuda_choice

case $cuda_choice in
    118)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    121)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        echo "Installing PyTorch (CPU only)..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
        echo -e "${YELLOW}⚠${NC}  Note: Mamba-SSM may not work without CUDA"
        ;;
    *)
        echo -e "${RED}✗${NC} Invalid choice. Exiting."
        exit 1
        ;;
esac

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo -e "${RED}✗${NC} PyTorch installation failed"
    exit 1
}
echo -e "${GREEN}✓${NC} PyTorch installed successfully"

# Install Mamba dependencies
echo ""
echo "========================================="
echo "Installing Mamba SSM dependencies..."
echo "========================================="

echo "Installing packaging..."
pip install packaging

echo "Installing causal-conv1d..."
pip install causal-conv1d>=1.1.0 || {
    echo -e "${YELLOW}⚠${NC}  causal-conv1d installation failed. Trying alternative method..."
    pip install causal-conv1d --no-build-isolation
}

echo "Installing mamba-ssm (this may take 5-10 minutes)..."
pip install mamba-ssm>=1.1.0 || {
    echo -e "${RED}✗${NC} mamba-ssm installation failed."
    echo "Please check:"
    echo "  1. CUDA toolkit is properly installed"
    echo "  2. gcc/g++ compiler is available"
    echo "  3. PyTorch CUDA version matches your CUDA toolkit"
    exit 1
}

echo -e "${GREEN}✓${NC} Mamba dependencies installed successfully"

# Install other requirements
echo ""
echo "========================================="
echo "Installing other dependencies..."
echo "========================================="

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Requirements installed successfully"
else
    echo -e "${YELLOW}⚠${NC}  requirements.txt not found. Installing manually..."
    pip install einops>=0.7.0 timm>=0.9.0 numpy>=1.24.0
fi

# Verify installation
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="

python -c "
import torch
import einops
import timm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('✓ All core packages imported successfully')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - CUDA available: {torch.cuda.is_available()}')
" || {
    echo -e "${RED}✗${NC} Installation verification failed"
    exit 1
}

echo ""
echo "========================================="
echo -e "${GREEN}✅ Installation complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Download pretrained weights:"
echo "     bash scripts/download_weights.sh"
echo ""
echo "  2. Run example inference:"
echo "     python examples/inference.py"
echo ""
echo "  3. Train your model:"
echo "     python train.py --config configs/default.yaml"
echo ""
