"""
PPO Training for OpenFront RL Agent

Uses PyTorch for the policy/value networks and trains against
built-in bot opponents in the headless OpenFront engine.

GPU-ready: automatically uses CUDA if available.
Supports vectorized environments for parallel rollout collection.
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

try:
    import wandb
except ImportError:
    wandb = None

# Maps organized by complexity tier — curriculum adds more maps as training progresses
MAP_TIER_1 = [
    # Simple, balanced maps for learning basics
    "plains", "big_plains", "world", "giantworldmap", "ocean_and_land", "half_land_half_ocean",
]
MAP_TIER_2 = MAP_TIER_1 + [
    # Mid-complexity: real geography, moderate water/chokepoints
    "europe", "europeclassic", "northamerica", "africa", "asia", "australia",
    "southamerica", "mediterranean", "britannia", "britanniaclassic",
    "eastasia", "oceania", "pangaea", "mena",
]
MAP_TIER_3 = MAP_TIER_2 + [
    # Harder: islands, straits, unusual layouts
    "aegean", "alps", "amazonriver", "amazonriverwide", "arctic",
    "baikal", "beringstrait", "betweentwoseas", "blacksea",
    "bosphorusstraits", "deglaciatedantarctica",
    "falklandislands", "faroeislands", "fourislands",
    "gatewaytotheatlantic", "gulfofstlawrence", "halkidiki", "hawaii",
    "iceland", "italia", "japan", "lemnos", "lisbon", "manicouagan",
    "niledelta", "passage", "sanfrancisco",
    "straitofgibraltar", "straitofhormuz", "surrounded",
    "thebox", "theboxplus", "tourney1", "tourney2", "tourney3", "tourney4",
    "tradersdream", "twolakes", "worldrotated", "yenisei",
    "achiran", "mars", "milkyway", "montreal", "newyorkcity", "pluto",
    "reglaciatedantarctica",
]
AVAILABLE_MAPS = MAP_TIER_3  # Full list for --maps default

from env import NUM_ACTIONS


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for MultiDiscrete action space."""

    def __init__(self, obs_dim: int, max_neighbors: int, hidden_sizes: list[int] = None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Action heads (one per MultiDiscrete dimension)
        self.action_type_head = nn.Linear(in_dim, NUM_ACTIONS)
        self.target_head = nn.Linear(in_dim, max_neighbors)
        self.troop_head = nn.Linear(in_dim, 5)

        # Value head
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "action_type": self.action_type_head(features),
            "target": self.target_head(features),
            "troop": self.troop_head(features),
            "value": self.value_head(features).squeeze(-1),
        }

    def get_action_and_value(self, x, action=None, action_mask=None):
        out = self.forward(x)

        # Apply action mask: set logits of invalid actions to -inf
        action_type_logits = out["action_type"]
        if action_mask is not None:
            # action_mask: (batch, NUM_ACTIONS) with 1=valid, 0=invalid
            action_type_logits = action_type_logits + (1 - action_mask) * (-1e8)

        # Create categorical distributions for each action dimension
        dist_type = torch.distributions.Categorical(logits=action_type_logits)
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


def compute_gae_vec(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Compute GAE for vectorized rollouts.

    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        last_values: (num_envs,) bootstrap values for last step
    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(num_envs)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def find_latest_checkpoint(save_dir: Path):
    """Find the checkpoint with the highest episode number."""
    ckpts = list(save_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        return None
    def ep_num(p):
        return int(p.stem.split("_")[1])
    return max(ckpts, key=ep_num)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    maps = args.maps.split(",")
    # When using curriculum, start with tier 1 maps regardless of --maps
    if args.curriculum:
        maps = MAP_TIER_1
    max_neighbors = 16
    obs_dim = 16 + max_neighbors * 4  # 80

    # Create vectorized environment
    from vec_env import VecOpenFrontEnv
    envs = VecOpenFrontEnv(
        num_envs=args.num_envs,
        maps=maps,
        num_opponents=args.opponents,
        difficulty=args.difficulty,
        ticks_per_step=args.ticks_per_step,
        max_steps=args.max_steps,
        max_neighbors=max_neighbors,
    )

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    model = ActorCritic(obs_dim, max_neighbors, hidden_sizes=hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_reward = -float("inf")
    log_entries = []
    global_step = 0
    start_update = 0
    num_episodes = 0

    # Resume from checkpoint if requested
    if args.resume:
        state_path = save_dir / "state.json"
        ckpt = find_latest_checkpoint(save_dir)
        if ckpt is not None:
            print(f"Resuming from checkpoint: {ckpt}")
            checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_update = checkpoint.get("update", 0)
            global_step = checkpoint.get("global_step", 0)
            num_episodes = checkpoint.get("num_episodes", 0)

            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                best_reward = state.get("best_reward", -float("inf"))

            log_path = save_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log_entries = json.load(f)

            print(f"Resumed at update={start_update}, global_step={global_step}, best_reward={best_reward:.2f}")
        else:
            print("No checkpoint found, starting fresh.")

    # Initialize wandb
    if wandb is not None:
        wandb.init(
            project="openfront-rl",
            config=vars(args),
            resume="allow",
        )
        wandb.config.update({"obs_dim": obs_dim, "maps": maps}, allow_val_change=True)

    print(f"Training PPO on maps={maps}, opponents={args.opponents}")
    print(f"num_envs={args.num_envs}, rollout_steps={args.rollout_steps}")
    print(f"batch_size={args.num_envs * args.rollout_steps}, minibatch_size={args.minibatch_size}")
    print(f"obs_dim={obs_dim}, device={device}")
    print(f"Saving checkpoints to {save_dir}")

    # Storage for rollouts: (num_steps, num_envs, ...)
    obs_buf = np.zeros((args.rollout_steps, args.num_envs, obs_dim), dtype=np.float32)
    masks_buf = np.ones((args.rollout_steps, args.num_envs, NUM_ACTIONS), dtype=np.float32)
    actions_buf = np.zeros((args.rollout_steps, args.num_envs, 3), dtype=np.float32)
    logprobs_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    dones_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    values_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)

    # Track per-env episode stats
    env_ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    env_ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    # Initialize
    obs, action_masks = envs.reset_all()  # (num_envs, obs_dim), (num_envs, NUM_ACTIONS)

    # Curriculum phase boundaries (fraction of num_updates)
    CURRICULUM_BOUNDS = [0.05, 0.30, 0.50, 1.0]

    for update in range(start_update, args.num_updates):
        t_start = time.time()

        # Curriculum learning: ramp difficulty, opponents, AND map pool over training
        if args.curriculum:
            progress = update / args.num_updates
            if progress < CURRICULUM_BOUNDS[0]:
                new_diff, new_opp, new_maps = "Easy", 2, MAP_TIER_1
            elif progress < CURRICULUM_BOUNDS[1]:
                new_diff, new_opp, new_maps = "Medium", 5, MAP_TIER_2
            elif progress < CURRICULUM_BOUNDS[2]:
                new_diff, new_opp, new_maps = "Hard", 8, MAP_TIER_3
            else:
                new_diff, new_opp, new_maps = "Hard", 12, MAP_TIER_3
            if envs.difficulty != new_diff or envs.num_opponents != new_opp:
                print(f"  Curriculum: switching to {new_diff} with {new_opp} opponents, {len(new_maps)} maps (progress={progress:.0%})")
                envs.difficulty = new_diff
                envs.num_opponents = new_opp
            if len(envs.maps) != len(new_maps):
                envs.maps = new_maps

        # Global LR annealing
        if args.anneal_lr:
            frac = 1.0 - update / args.num_updates
            lr_now = frac * args.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        # Collect rollout
        for step in range(args.rollout_steps):
            obs_buf[step] = obs
            masks_buf[step] = action_masks

            obs_t = torch.FloatTensor(obs).to(device)
            masks_t = torch.FloatTensor(action_masks).to(device)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t, action_mask=masks_t)

            actions_np = action.cpu().numpy()  # (num_envs, 3)
            logprobs_buf[step] = log_prob.cpu().numpy()
            values_buf[step] = value.cpu().numpy()
            actions_buf[step] = actions_np

            next_obs, next_masks, rewards, dones, truncateds, infos = envs.step(actions_np)
            rewards_buf[step] = rewards
            dones_buf[step] = dones | truncateds

            # Track episode stats and auto-reset finished envs
            env_ep_rewards += rewards
            env_ep_lengths += 1
            for i in range(args.num_envs):
                if dones[i] or truncateds[i]:
                    episode_rewards.append(float(env_ep_rewards[i]))
                    episode_lengths.append(int(env_ep_lengths[i]))
                    num_episodes += 1
                    env_ep_rewards[i] = 0
                    env_ep_lengths[i] = 0
                    next_obs[i], next_masks[i] = envs.reset_single(i)

            global_step += args.num_envs
            obs = next_obs
            action_masks = next_masks

        # Bootstrap values for last step
        with torch.no_grad():
            last_values = model.get_action_and_value(torch.FloatTensor(obs).to(device))[3]
            last_values = last_values.cpu().numpy()

        # Compute GAE
        advantages, returns = compute_gae_vec(
            rewards_buf, values_buf, dones_buf, last_values,
            gamma=args.gamma, lam=args.gae_lambda,
        )

        # Flatten (num_steps, num_envs, ...) -> (batch_size, ...)
        batch_size = args.rollout_steps * args.num_envs
        b_obs = torch.FloatTensor(obs_buf.reshape(batch_size, -1)).to(device)
        b_masks = torch.FloatTensor(masks_buf.reshape(batch_size, -1)).to(device)
        b_actions = torch.FloatTensor(actions_buf.reshape(batch_size, -1)).to(device)
        b_logprobs = torch.FloatTensor(logprobs_buf.reshape(batch_size)).to(device)
        b_advantages = torch.FloatTensor(advantages.reshape(batch_size)).to(device)
        b_returns = torch.FloatTensor(returns.reshape(batch_size)).to(device)
        b_values = torch.FloatTensor(values_buf.reshape(batch_size)).to(device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO epochs with minibatch updates
        for _ in range(args.ppo_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, args.minibatch_size):
                end = min(start + args.minibatch_size, batch_size)
                mb_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = model.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx], action_mask=b_masks[mb_idx]
                )

                ratio = torch.exp(new_log_probs - b_logprobs[mb_idx])
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)

                policy_loss = -torch.min(
                    ratio * b_advantages[mb_idx], clipped * b_advantages[mb_idx]
                ).mean()

                # Clipped value loss to prevent large value updates
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_values - b_values[mb_idx], -args.clip_eps, args.clip_eps
                )
                v_loss_unclipped = (new_values - b_returns[mb_idx]).pow(2)
                v_loss_clipped = (v_clipped - b_returns[mb_idx]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        t_elapsed = time.time() - t_start
        sps = (args.rollout_steps * args.num_envs) / t_elapsed

        if (update + 1) % args.log_interval == 0 and len(episode_rewards) > 0:
            mean_r = np.mean(episode_rewards)
            mean_l = np.mean(episode_lengths)
            entry = {
                "update": update + 1,
                "global_step": global_step,
                "num_episodes": num_episodes,
                "mean_reward": float(mean_r),
                "mean_length": float(mean_l),
                "loss": float(loss.item()),
                "sps": float(sps),
            }
            log_entries.append(entry)
            current_lr = optimizer.param_groups[0]["lr"]
            if wandb is not None:
                wandb.log({
                    "update": update + 1,
                    "global_step": global_step,
                    "num_episodes": num_episodes,
                    "reward/mean": float(mean_r),
                    "episode_length": float(mean_l),
                    "loss/total": float(loss.item()),
                    "loss/policy": float(policy_loss.item()),
                    "loss/value": float(value_loss.item()),
                    "loss/entropy": float(entropy_loss.item()),
                    "perf/sps": float(sps),
                    "lr": current_lr,
                }, step=global_step)
            print(
                f"[update {update+1}/{args.num_updates}] "
                f"episodes={num_episodes} reward={mean_r:.2f} len={mean_l:.0f} "
                f"loss={loss.item():.4f} sps={sps:.0f}"
            )

        # Save checkpoint
        if (update + 1) % args.save_interval == 0:
            ckpt_path = save_dir / f"checkpoint_{update+1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update + 1,
                    "global_step": global_step,
                    "num_episodes": num_episodes,
                },
                ckpt_path,
            )

            mean_r = np.mean(episode_rewards) if episode_rewards else -float("inf")
            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  New best model saved (reward={best_reward:.2f})")

            with open(save_dir / "state.json", "w") as f:
                json.dump({
                    "update": update + 1,
                    "global_step": global_step,
                    "num_episodes": num_episodes,
                    "best_reward": float(best_reward),
                    "config": {
                        "maps": maps,
                        "opponents": args.opponents,
                        "difficulty": args.difficulty,
                        "obs_dim": obs_dim,
                        "max_neighbors": max_neighbors,
                        "num_envs": args.num_envs,
                        "lr": args.lr,
                        "rollout_steps": args.rollout_steps,
                        "hidden_sizes": hidden_sizes,
                    },
                }, f, indent=2)

            with open(save_dir / "training_log.json", "w") as f:
                json.dump(log_entries, f, indent=2)

    # Final save
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    envs.close()
    if wandb is not None:
        wandb.finish()
    print(f"Training complete. Best reward: {best_reward:.2f}")

    # Auto-push to HuggingFace
    if (save_dir / "best_model.pt").exists():
        try:
            from push_to_hf import push as hf_push
            hf_args = argparse.Namespace(repo=args.hf_repo, checkpoint_dir=str(save_dir), force=False)
            hf_push(hf_args)
        except Exception as e:
            print(f"HuggingFace push failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenFront RL agent with PPO")
    parser.add_argument("--maps", default=",".join(AVAILABLE_MAPS), help="Comma-separated maps to randomly sample each episode")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Steps per env per rollout")
    parser.add_argument("--num-updates", type=int, default=10000, help="Number of PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true", help="Linear LR annealing to zero")
    parser.add_argument("--curriculum", action="store_true", help="Curriculum learning: ramp difficulty/opponents over training")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--hidden-sizes", default="256,256,128", help="Comma-separated backbone layer sizes")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--hf-repo", default="mischievers/openfront-rl-agent", help="HuggingFace repo for auto-push")
    args = parser.parse_args()
    train(args)
