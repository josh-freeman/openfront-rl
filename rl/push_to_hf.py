"""
Push trained OpenFront RL model to HuggingFace Hub.

Usage:
  python push_to_hf.py --repo mischievers/openfront-rl-agent
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def create_model_card(checkpoint_dir: Path) -> str:
    """Generate a model card from training state and logs."""
    state = {}
    state_path = checkpoint_dir / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    log_path = checkpoint_dir / "training_log.json"
    final_metrics = {}
    if log_path.exists():
        with open(log_path) as f:
            logs = json.load(f)
        if logs:
            final_metrics = logs[-1]

    # Pull config from state.json (saved by train.py)
    config = state.get("config", {})
    maps = config.get("maps", "N/A")
    if isinstance(maps, list):
        maps = ", ".join(maps)
    opponents = config.get("opponents", "N/A")
    difficulty = config.get("difficulty", "N/A")
    obs_dim = config.get("obs_dim", "N/A")
    max_neighbors = config.get("max_neighbors", "N/A")
    num_envs = config.get("num_envs", "N/A")
    lr = config.get("lr", "N/A")
    rollout_steps = config.get("rollout_steps", "N/A")

    return f"""---
license: mit
tags:
  - reinforcement-learning
  - ppo
  - openfront
  - game-ai
---

# OpenFront RL Agent

PPO-trained agent for [OpenFront.io](https://openfront.io), a multiplayer territory control game.

## Training Details

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Architecture:** Actor-Critic with shared backbone (256→256→128)
- **Observation dim:** {obs_dim}
- **Max neighbors:** {max_neighbors}
- **Maps:** {maps} (random per episode)
- **Opponents:** {opponents} {difficulty} bots
- **Parallel envs:** {num_envs}
- **Learning rate:** {lr}
- **Rollout steps:** {rollout_steps}
- **Updates trained:** {state.get('update', 'N/A')}
- **Global steps:** {state.get('global_step', 'N/A')}
- **Best mean reward:** {state.get('best_reward', 'N/A')}

## Final Training Metrics

- **Mean reward:** {final_metrics.get('mean_reward', 'N/A')}
- **Mean episode length:** {final_metrics.get('mean_length', 'N/A')}
- **Loss:** {final_metrics.get('loss', 'N/A')}

## Usage

```python
from train import ActorCritic
import torch

model = ActorCritic(obs_dim={obs_dim}, max_neighbors={max_neighbors})
model.load_state_dict(torch.load("best_model.pt", weights_only=True))
model.eval()
```

## Repository

Trained from [josh-freeman/openfront-rl](https://github.com/josh-freeman/openfront-rl).
"""


def get_remote_best_reward(api: HfApi, repo: str) -> float:
    """Fetch the best_reward from the remote HF repo's training_log.json."""
    try:
        import tempfile
        path = api.hf_hub_download(repo_id=repo, filename="training_log.json",
                                    cache_dir=tempfile.mkdtemp())
        with open(path) as f:
            logs = json.load(f)
        if logs:
            return float(logs[-1].get("mean_reward", -float("inf")))
    except Exception:
        pass
    return -float("inf")


def push(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(args.repo, exist_ok=True)

    # Upload best model
    best_model = checkpoint_dir / "best_model.pt"
    if not best_model.exists():
        print(f"ERROR: {best_model} not found. Train first.")
        return

    # Compare against current HF model before overwriting
    local_state_path = checkpoint_dir / "state.json"
    local_reward = -float("inf")
    if local_state_path.exists():
        with open(local_state_path) as f:
            local_reward = float(json.load(f).get("best_reward", -float("inf")))

    remote_reward = get_remote_best_reward(api, args.repo)
    print(f"Local best reward: {local_reward:.2f}, Remote best reward: {remote_reward:.2f}")

    if remote_reward > local_reward and not args.force:
        print(f"SKIPPING push: remote model is better ({remote_reward:.2f} > {local_reward:.2f}). Use --force to override.")
        return

    print(f"Uploading {best_model} to {args.repo}...")
    api.upload_file(
        path_or_fileobj=str(best_model),
        path_in_repo="best_model.pt",
        repo_id=args.repo,
    )

    # Upload training log
    log_path = checkpoint_dir / "training_log.json"
    if log_path.exists():
        print(f"Uploading {log_path}...")
        api.upload_file(
            path_or_fileobj=str(log_path),
            path_in_repo="training_log.json",
            repo_id=args.repo,
        )

    # Upload model card
    card = create_model_card(checkpoint_dir)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
    )

    print(f"Done! Model at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push trained model to HuggingFace")
    parser.add_argument("--repo", default="mischievers/openfront-rl-agent")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--force", action="store_true", help="Push even if remote model has higher reward")
    args = parser.parse_args()
    push(args)
