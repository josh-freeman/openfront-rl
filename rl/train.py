"""
PPO Training for OpenFront RL Agent

Uses PyTorch for the policy/value networks and trains against
built-in bot opponents in the headless OpenFront engine.

GPU-ready: automatically uses CUDA if available.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
import argparse
import json
import random
from pathlib import Path

import wandb

AVAILABLE_MAPS = ["plains", "big_plains", "world", "giantworldmap", "ocean_and_land", "half_land_half_ocean"]

from env import OpenFrontEnv, NUM_ACTIONS


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for MultiDiscrete action space."""

    def __init__(self, obs_dim: int, max_neighbors: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Action heads (one per MultiDiscrete dimension)
        self.action_type_head = nn.Linear(128, NUM_ACTIONS)
        self.target_head = nn.Linear(128, max_neighbors)
        self.troop_head = nn.Linear(128, 5)

        # Value head
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "action_type": self.action_type_head(features),
            "target": self.target_head(features),
            "troop": self.troop_head(features),
            "value": self.value_head(features).squeeze(-1),
        }

    def get_action_and_value(self, x, action=None):
        out = self.forward(x)

        # Create categorical distributions for each action dimension
        dist_type = torch.distributions.Categorical(logits=out["action_type"])
        dist_target = torch.distributions.Categorical(logits=out["target"])
        dist_troop = torch.distributions.Categorical(logits=out["troop"])

        if action is None:
            a_type = dist_type.sample()
            a_target = dist_target.sample()
            a_troop = dist_troop.sample()
            action = torch.stack([a_type, a_target, a_troop], dim=-1)
        else:
            a_type = action[..., 0].long()
            a_target = action[..., 1].long()
            a_troop = action[..., 2].long()

        log_prob = (
            dist_type.log_prob(a_type)
            + dist_target.log_prob(a_target)
            + dist_troop.log_prob(a_troop)
        )
        entropy = (
            dist_type.entropy() + dist_target.entropy() + dist_troop.entropy()
        )

        return action, log_prob, entropy, out["value"]


class RolloutBuffer:
    """Stores trajectories for PPO updates."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self, device):
        return {
            "obs": torch.FloatTensor(np.array(self.obs)).to(device),
            "actions": torch.FloatTensor(np.array(self.actions)).to(device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)).to(device),
            "rewards": torch.FloatTensor(np.array(self.rewards)).to(device),
            "dones": torch.FloatTensor(np.array(self.dones)).to(device),
            "values": torch.FloatTensor(np.array(self.values)).to(device),
        }

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    """Compute Generalized Advantage Estimation.

    last_value: bootstrap value V(s_T) for truncated episodes. Should be 0 for
    true terminal states, or the critic's estimate for the final obs if truncated.
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def find_latest_checkpoint(save_dir: Path):
    """Find the checkpoint with the highest episode number."""
    ckpts = list(save_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        return None
    # Extract episode numbers and find max
    def ep_num(p):
        return int(p.stem.split("_")[1])
    return max(ckpts, key=ep_num)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = OpenFrontEnv(
        map_name=args.map,
        num_opponents=args.opponents,
        difficulty=args.difficulty,
        ticks_per_step=args.ticks_per_step,
        max_steps=args.max_steps,
    )

    obs_dim = env.observation_space.shape[0]
    model = ActorCritic(obs_dim, env.max_neighbors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_reward = -float("inf")
    log_entries = []
    start_episode = 0
    global_step = 0

    # Resume from checkpoint if requested
    if args.resume:
        state_path = save_dir / "state.json"
        ckpt = find_latest_checkpoint(save_dir)
        if ckpt is not None:
            print(f"Resuming from checkpoint: {ckpt}")
            checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = checkpoint["episode"]
            global_step = checkpoint["global_step"]

            # Restore best_reward from state.json
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                best_reward = state.get("best_reward", -float("inf"))

            # Restore training log
            log_path = save_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log_entries = json.load(f)

            print(f"Resumed at episode={start_episode}, global_step={global_step}, best_reward={best_reward:.2f}")
        else:
            print("No checkpoint found, starting fresh.")

    maps = args.maps.split(",")

    # Initialize wandb
    wandb.init(
        project="openfront-rl",
        config=vars(args),
        resume="allow",
    )
    wandb.config.update({"obs_dim": obs_dim, "maps": maps}, allow_val_change=True)

    print(f"Training PPO on maps={maps}, opponents={args.opponents}")
    print(f"obs_dim={obs_dim}, device={device}")
    print(f"Saving checkpoints to {save_dir}")

    for episode in range(start_episode, args.num_episodes):
        # Randomize map each episode
        env.map_name = random.choice(maps)
        obs, info = env.reset()
        buffer = RolloutBuffer()
        ep_reward = 0
        ep_steps = 0

        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(action_np)

            buffer.add(
                obs,
                action_np,
                log_prob.item(),
                reward,
                float(done or truncated),
                value.item(),
            )

            ep_reward += reward
            ep_steps += 1
            global_step += 1
            obs = next_obs

            if done or truncated:
                break

        # Bootstrap value for truncated episodes
        last_value = 0.0
        if truncated and not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, _, v = model.get_action_and_value(obs_t)
                last_value = v.item()

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        # PPO update
        data = buffer.get(device)
        with torch.no_grad():
            values_np = data["values"].cpu().numpy()
        advantages, returns = compute_gae(
            data["rewards"].cpu().numpy(),
            values_np,
            data["dones"].cpu().numpy(),
            gamma=args.gamma,
            lam=args.gae_lambda,
            last_value=last_value,
        )
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # PPO epochs with minibatch updates
        batch_size = len(advantages)
        for _ in range(args.ppo_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, args.minibatch_size):
                end = min(start + args.minibatch_size, batch_size)
                mb_idx = indices[start:end]

                mb_obs = data["obs"][mb_idx]
                mb_actions = data["actions"][mb_idx]
                mb_log_probs = data["log_probs"][mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                _, new_log_probs, entropy, new_values = model.get_action_and_value(
                    mb_obs, mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_log_probs)
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)

                policy_loss = -torch.min(
                    ratio * mb_advantages, clipped * mb_advantages
                ).mean()
                value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if (episode + 1) % args.log_interval == 0:
            mean_r = np.mean(episode_rewards)
            mean_l = np.mean(episode_lengths)
            entry = {
                "episode": episode + 1,
                "global_step": global_step,
                "mean_reward": float(mean_r),
                "mean_length": float(mean_l),
                "ep_reward": float(ep_reward),
                "loss": float(loss.item()),
            }
            log_entries.append(entry)
            wandb.log({
                "episode": episode + 1,
                "global_step": global_step,
                "reward/mean": float(mean_r),
                "reward/episode": float(ep_reward),
                "episode_length": float(mean_l),
                "loss/total": float(loss.item()),
                "loss/policy": float(policy_loss.item()),
                "loss/value": float(value_loss.item()),
                "loss/entropy": float(entropy_loss.item()),
            }, step=global_step)
            print(
                f"[ep {episode+1}/{args.num_episodes}] "
                f"reward={mean_r:.2f} len={mean_l:.0f} loss={loss.item():.4f} "
                f"steps={global_step}"
            )

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            ckpt_path = save_dir / f"checkpoint_{episode+1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": episode + 1,
                    "global_step": global_step,
                },
                ckpt_path,
            )

            # Save best — only overwrite if actually better
            mean_r = np.mean(episode_rewards) if episode_rewards else -float("inf")
            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  New best model saved (reward={best_reward:.2f})")

            # Save state.json for resume
            with open(save_dir / "state.json", "w") as f:
                json.dump({
                    "episode": episode + 1,
                    "global_step": global_step,
                    "best_reward": float(best_reward),
                }, f, indent=2)

            # Save log
            with open(save_dir / "training_log.json", "w") as f:
                json.dump(log_entries, f, indent=2)

    # Final save
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    env.close()
    wandb.finish()
    print(f"Training complete. Best reward: {best_reward:.2f}")

    # Auto-push to HuggingFace
    if (save_dir / "best_model.pt").exists():
        try:
            from push_to_hf import push as hf_push
            hf_args = argparse.Namespace(repo=args.hf_repo, checkpoint_dir=str(save_dir))
            hf_push(hf_args)
        except Exception as e:
            print(f"HuggingFace push failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenFront RL agent with PPO")
    parser.add_argument("--map", default="plains", help="Map for env init")
    parser.add_argument("--maps", default=",".join(AVAILABLE_MAPS), help="Comma-separated maps to randomly sample each episode")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--hf-repo", default="JoshuaFreeman/openfront-rl-agent", help="HuggingFace repo for auto-push")
    args = parser.parse_args()
    train(args)
