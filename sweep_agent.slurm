#!/bin/bash
#SBATCH --job-name=ofrl-sweep
#SBATCH --output=logs/sweep_%j.log
#SBATCH --error=logs/sweep_%j.log
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

echo "=== Sweep agent $SLURM_JOB_ID starting at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Sweep ID: $SWEEP_ID"

git checkout sweep

cd rl

# Clean up any previous sweep checkpoints so trials don't collide
rm -rf checkpoints_sweep

# Run 1 sweep trial per job (--count 1)
wandb agent --count 1 "hallu-probes-ethz/openfront-rl/$SWEEP_ID"

echo "=== Sweep agent $SLURM_JOB_ID finished at $(date) ==="
