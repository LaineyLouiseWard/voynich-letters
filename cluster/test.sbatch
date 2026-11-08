#!/bin/bash
# Test job before the full training: one pass of step 0 and one of step 1 over the training lines with the full run's
# settings (config/model.yaml, the authors' batch sizes), to measure the time per pass and the peak GPU memory on
# Sonic's card. Output: logs/letter-model-test-<job>.out and data/runs/test-<job>/.
# The laptop can run this file too, for a dry run: PY=<env python> EXTRA0="--batch_size 4" EXTRA1="--batch_size 2"
# bash cluster/test.sbatch (smaller batches fit its 8 GB).
#SBATCH --job-name=letter-model-test
#SBATCH --partition=gpu
#SBATCH --constraint=l40s|a100|h100
#SBATCH --account=shared_acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8               # the training code's 4 data-loading workers and the main process
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-env/bin/python}
JOB=${SLURM_JOB_ID:-laptop-$(date +%H%M%S)}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
gpu_log=logs/letter-model-test-$JOB.gpu
mkdir -p logs
(while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$gpu_log"; sleep 2; done) &
trap 'kill %1' EXIT
for step in 0 1; do
    extra=EXTRA$step
    start=$(date +%s)
    "$PY" scripts/letter_model.py train "test-$JOB" "$step" --max_e 0 ${!extra:-}
    echo "step $step: one pass in $(( $(date +%s) - start )) s, model loading included"
done
echo "peak GPU MiB: $(cut -d, -f1 "$gpu_log" | sort -n | tail -1)"
