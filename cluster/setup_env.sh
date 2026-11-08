#!/bin/bash
# Run once on the Sonic login node, from ~/scratch/voynich-letters, after cluster/sync.sh. Builds the training environment
# in scratch, then fetches, patches and compiles the letter model and downloads its detector (scripts/setup_model.sh).
# Sonic's GPU drivers run CUDA 12.7 at most (4 Nov 2026), so torch is the CUDA 12.6 build (torchvision 0.29.1, the newest
# built for 12.6), and the extension is compiled with Sonic's CUDA 12.6 module for the V100, A100, L40S and H100 (the
# login node has no GPU; compiling needs none). The tests at the end stop the setup unless all of this holds.
set -euo pipefail
mkdir -p logs
module load anaconda3/2025.12-1 cuda/12.6.2-gcc-11.5.0-3bilwrd
# download caches in scratch: home's quota is small
export CONDA_PKGS_DIRS=$HOME/scratch/conda-pkgs PIP_CACHE_DIR=$HOME/scratch/pip-cache
[ -d env ] || conda create -y -p "$PWD/env" -c conda-forge --override-channels python=3.11.18 pip
env/bin/pip install torch==2.14.0+cu126 torchvision==0.29.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
env/bin/pip install -r cluster/requirements-train.txt
env/bin/pip check
env/bin/python -c "import sys, torch; sys.exit(None if torch.version.cuda == '12.6' else
    f'torch {torch.__version__} is built for CUDA {torch.version.cuda}, not 12.6: Sonic cannot run it')"
nvcc --version | grep -q "release 12.6" || { echo "nvcc is not CUDA 12.6"; exit 1; }
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")") PY=$PWD/env/bin/python TORCH_CUDA_ARCH_LIST="7.0;8.0;8.9;9.0" \
    bash scripts/setup_model.sh
