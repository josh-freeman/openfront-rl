#!/bin/bash
#SBATCH --job-name=openfront-rl
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=03:00:00
#SBATCH --requeue

set -euo pipefail

REPO_DIR=/orcd/home/002/joshinou/openfront-rl
cd "$REPO_DIR"
mkdir -p logs

# Load modules
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

# Activate venv
source "$REPO_DIR/.venv/bin/activate"

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Auto-resume: if checkpoints exist, add --resume flag
RESUME_FLAG=""
if [ -f checkpoints/state.json ]; then
    echo "Found existing checkpoint, will resume training"
    RESUME_FLAG="--resume"
fi

cd rl
python -u train.py \
    --maps plains,big_plains,world,giantworldmap,ocean_and_land,half_land_half_ocean \
    --opponents 2 \
    --difficulty Easy \
    --num-envs 8 \
    --rollout-steps 512 \
    --num-updates 4000 \
    --lr 2e-4 \
    --vf-coef 0.25 \
    --anneal-lr \
    --curriculum \
    --save-interval 50 \
    --log-interval 5 \
    $RESUME_FLAG

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
