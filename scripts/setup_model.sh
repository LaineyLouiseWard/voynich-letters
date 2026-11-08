#!/bin/bash
# Fetch the letter model's code at config/model.yaml's commit, apply scripts/morphology4metrology.patch, compile its
# deformable attention CUDA extension, and download the pretrained detector (sha256 tested). Run from the repo root:
#   PY=<env python> TORCH_CUDA_ARCH_LIST=<GPU generations> bash scripts/setup_model.sh
# Laptop (RTX 5060): TORCH_CUDA_ARCH_LIST=12.0 (make model). Sonic (V100, A100, L40S, H100): "7.0;8.0;8.9;9.0"
# (cluster/setup_env.sh). The env needs nvcc of torch's CUDA version (conda cuda-nvcc); compiling needs no GPU.
# The patch: two fixes needed to run the code (a non-breaking space in datasets/dataset.py; value.type() calls removed
# from current PyTorch, in the CUDA extension), compiling without a GPU present (FORCE_CUDA), and three speed changes
# to reconstruction.py's per-batch bookkeeping that leave the model and its training unchanged.
set -euo pipefail
: "${PY:?set PY to the env python}"
: "${TORCH_CUDA_ARCH_LIST:?set TORCH_CUDA_ARCH_LIST}"
ENV=$(dirname "$(dirname "$PY")")
read -r CODE COMMIT PREDICT CKPT_URL CKPT_SHA M CKPT < <("$PY" -c "import yaml
m = yaml.safe_load(open('config/model.yaml')); p = yaml.safe_load(open('config/paths.yaml'))
print(m['code'], m['commit'], m['predict_commit'], m['checkpoint']['url'], m['checkpoint']['sha256'], p['model_code'],
      p['checkpoint'])")
PATCH=$PWD/scripts/morphology4metrology.patch

[ -d "$M/.git" ] || git clone -q "$CODE" "$M"
git -C "$M" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$M" checkout -q "$COMMIT"
if git -C "$M" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    git -C "$M" apply "$PATCH"
    echo "patch applied"
fi
# the prediction script from config/model.yaml's predict_commit (see the comment there)
git -C "$M" show "$PREDICT:predict.py" > "$M/predict.py"

# CUDA_HOME: Sonic's CUDA module if set (cluster/setup_env.sh), else the env's own conda nvcc (laptop).
export CUDA_HOME=${CUDA_HOME:-$ENV}
export PATH=$ENV/bin:$PATH FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$("$PY" -c "import glob, nvidia, os
print(':'.join(glob.glob(os.path.join(nvidia.__path__[0], '*', 'include'))))")
(cd "$M/models/dino/ops" && "$PY" -m pip install -q --no-build-isolation .)
(cd "$M" && "$PY" -c "import torch, MultiScaleDeformableAttention; print('extension built for torch', torch.__version__)")

mkdir -p "$(dirname "$CKPT")"
if [ ! -f "$CKPT" ]; then
    curl -sL --fail -o "$CKPT.part" "$CKPT_URL"
    mv "$CKPT.part" "$CKPT"
fi
echo "$CKPT_SHA  $CKPT" | sha256sum -c -
