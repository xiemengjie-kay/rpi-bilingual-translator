#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Define variables
ENV_NAME="cs329e"
PYTHON_VERSION="3.10"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_PATH="$HOME/miniconda3"

# Step 1: Download Miniconda installer
echo "Downloading Miniconda installer..."
wget "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

# Step 2: Install Miniconda silently
echo "Installing Miniconda..."
bash "$MINICONDA_INSTALLER" -b -u -p "$INSTALL_PATH"

# Step 3: Initialize Conda for the current shell
echo "Initializing Conda..."
eval "$("$INSTALL_PATH/bin/conda" shell.bash hook)"
conda init

# Step 4: Create a new Conda environment with specified Python version
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Step 5: Activate the new environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Step 6: Install dependencies using Conda
conda install -y -c conda-forge \
    transformers=4.41.0 \
    optimum=1.20.0 \
    onnxruntime=1.20.1 \
    onnx=1.17.0 \
    seaborn \
    sacrebleu \
    sentencepiece \
    libsndfile \
    soundfile=0.13.1

# Step 7: Install dependencies using Pip
pip install \
    ctranslate2==4.6.0 \
    faster-whisper==1.1.1 \
    av==14.3.0 \
    soxr==0.5.0.post1

# Step 7: Install additional dependencies using pip
# pip install requests beautifulsoup4

# Step 8: Clean up installer
echo "Cleaning up..."
rm "$MINICONDA_INSTALLER"

echo "✅ Setup complete. Environment '$ENV_NAME' is ready to use."
