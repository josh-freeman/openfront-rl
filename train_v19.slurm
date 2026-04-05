#!/bin/bash
#SBATCH --job-name=ofrl-v19
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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
echo "v19: potential-based reward shaping (alpha=1.25)"

git checkout v19-potential-shaping

cd rl
python -u train.py \
    --opponents 2 \
    --difficulty Easy \
    --num-envs 16 \
    --rollout-steps 1024 \
    --num-updates 4000 \
    --lr 1.5e-4 \
    --vf-coef 0.5 \
    --curriculum \
    --win-threshold 0.80 \
    --hidden-sizes 512,512,256 \
    --save-interval 10 \
    --log-interval 5 \
    --potential-alpha 1 \
    --save-dir ./checkpoints_v19

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
