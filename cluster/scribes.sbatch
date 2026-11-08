#!/bin/bash
# Step 2 per scribe (scripts/letter_model.py finetune): RUN's step-1 letter pictures re-tuned on each Davis scribe's
# training lines in turn, into data/runs/<RUN>/<NAME>/<scribe>/. Test job first, one pass per scribe:
#   sbatch --time=00:40:00 --export=ALL,RUN=reference,NAME=scribes-test,EXTRA="--max_e 0" cluster/scribes.sbatch
# then the full run, --time from the test:
#   sbatch --time=HH:MM:SS --export=ALL,RUN=reference,NAME=scribes cluster/scribes.sbatch
# Output: logs/letter-model-scribes-<job>.out, and the GPU memory every 2 s in logs/letter-model-scribes-<job>.gpu.
#SBATCH --job-name=letter-model-scribes
#SBATCH --partition=gpu
#SBATCH --constraint=l40s|h100          # as the training job: step 1 needed more than 40 GB at 8 lines a batch
#SBATCH --account=shared_acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8               # the training code's 4 data-loading workers and the main process
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${RUN:?set RUN, the run whose step 1 is re-tuned}"
: "${NAME:?set NAME, the output folder under data/runs/RUN}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmented GPU memory; results unchanged
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p logs
gpu_log=logs/letter-model-scribes-${SLURM_JOB_ID:-local}.gpu
(while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$gpu_log"; sleep 2; done) &
trap 'kill %1' EXIT
start=$(date +%s)
env/bin/python scripts/letter_model.py finetune "$RUN" "$NAME" ${EXTRA:-}
echo "all scribes in $(( $(date +%s) - start )) s"
