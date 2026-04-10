#!/bin/bash
set -e

# Network acceleration
source /etc/network_turbo 2>/dev/null || true

CONDA=/root/miniconda3
source $CONDA/etc/profile.d/conda.sh

# Step 1: System dependencies
echo '=== [1/7] System dependencies ==='
apt-get update -y --fix-missing && apt-get install -y ffmpeg unzip sox libsox-dev git-lfs && git lfs install

# Step 2: Create conda env
echo '=== [2/7] Create conda env ==='
if conda env list | grep -q cosyvoice; then
    echo 'env cosyvoice already exists, skipping'
else
    conda create -y -n cosyvoice python=3.10
fi
conda activate cosyvoice

# Step 3: Install pynini
echo '=== [3/7] Install pynini ==='
conda install -y -c conda-forge pynini==2.1.5

# Step 4: Install Python dependencies
echo '=== [4/7] Install Python deps ==='
pip install 'setuptools<75' wheel --no-cache-dir
pip install openai-whisper==20231117 --no-build-isolation --no-cache-dir
pip install -r /root/CosyVoice/requirements.txt --no-cache-dir

# Step 5: Install vLLM + overrides
echo '=== [5/7] Install vLLM ==='
pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4 python-dotenv 'ruamel.yaml<0.18' --no-cache-dir

# Step 6: Download models
echo '=== [6/7] Download models ==='
if [ -f /root/models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml ]; then
    echo 'Model already downloaded, skipping'
else
    python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/root/models/Fun-CosyVoice3-0.5B')
"
fi

# Step 7: Download and install ttsfrd
echo '=== [7/7] Install ttsfrd ==='
if python -c 'import ttsfrd' 2>/dev/null; then
    echo 'ttsfrd already installed, skipping'
else
    python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/root/models/CosyVoice-ttsfrd')
"
    cd /root/models/CosyVoice-ttsfrd
    unzip -o resource.zip -d .
    pip install ttsfrd_dependency-0.1-py3-none-any.whl
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
fi

echo '=== SETUP COMPLETE ==='
