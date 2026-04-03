#!/bin/bash
#SBATCH --job-name=ofrl-v9c
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail

REPO_DIR=/orcd/home/002/joshinou/openfront-rl
cd "$REPO_DIR"
mkdir -p logs

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64
source "$REPO_DIR/.venv/bin/activate"

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "v9c: 32 envs, LR=5e-4, one-shot reward, win-gated curriculum"

cd rl
python -u train.py \
    --opponents 2 \
    --difficulty Easy \
    --num-envs 32 \
    --rollout-steps 512 \
    --num-updates 4000 \
    --lr 5e-4 \
    --vf-coef 0.25 \
    --curriculum \
    --hidden-sizes 512,512,256 \
    --save-interval 10 \
    --log-interval 5 \
    --save-dir ./checkpoints_v9c

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
