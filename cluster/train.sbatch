#!/bin/bash
# The full training: step 0 then step 1 with config/model.yaml's settings. Submit only after cluster/test.sbatch, with
# --time set from its measured time per pass (L40S: step 0 93 s, step 1 66 s, so about 2.5 hours in all):
#   sbatch --time=04:00:00 --export=ALL,RUN=<name> cluster/train.sbatch
# Output: logs/letter-model-train-<job>.out and data/runs/<RUN>/.
#SBATCH --job-name=letter-model-train
#SBATCH --partition=gpu
#SBATCH --constraint=l40s|h100          # step 0 peaked at 39.7 GB on an L40S (test job 1080334); Sonic's A100s may be 40 GB
#SBATCH --account=shared_acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8               # the training code's 4 data-loading workers and the main process
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${RUN:?set RUN, the run name}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmented GPU memory; results unchanged
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
env/bin/python scripts/letter_model.py train "$RUN" 0
env/bin/python scripts/letter_model.py train "$RUN" 1
