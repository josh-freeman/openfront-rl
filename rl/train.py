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

from env import (
    NUM_ACTIONS, ACTION_ATTACK, ACTION_BOAT_ATTACK,
    ACTION_LAUNCH_ATOM, ACTION_LAUNCH_HBOMB, ACTION_LAUNCH_MIRV,
    ACTION_MOVE_WARSHIP,
)


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

    def get_action_and_value(self, x, action=None, action_mask=None,
                            land_target_mask=None, sea_target_mask=None):
        out = self.forward(x)

        # Apply action mask: set logits of invalid actions to -inf
        action_type_logits = out["action_type"]
        if action_mask is not None:
            # action_mask: (batch, NUM_ACTIONS) with 1=valid, 0=invalid
            action_type_logits = action_type_logits + (1 - action_mask) * (-1e8)

        # Create categorical distribution for action type
        dist_type = torch.distributions.Categorical(logits=action_type_logits)

        if action is None:
            a_type = dist_type.sample()
        else:
            a_type = action[..., 0].long()

        # Apply conditional target mask based on sampled action type
        target_logits = out["target"]
        if land_target_mask is not None and sea_target_mask is not None:
            is_land_attack = (a_type == ACTION_ATTACK)
            is_sea_action = (
                (a_type == ACTION_BOAT_ATTACK) |
                (a_type == ACTION_LAUNCH_ATOM) |
                (a_type == ACTION_LAUNCH_HBOMB) |
                (a_type == ACTION_LAUNCH_MIRV) |
                (a_type == ACTION_MOVE_WARSHIP)
            )
            # Pick the right mask per sample
            tmask = torch.where(
                is_land_attack.unsqueeze(-1),
                land_target_mask,
                torch.where(
                    is_sea_action.unsqueeze(-1),
                    sea_target_mask,
                    torch.ones_like(land_target_mask),  # other actions: all valid
                ),
            )
            target_logits = target_logits + (1 - tmask) * (-1e8)

        dist_target = torch.distributions.Categorical(logits=target_logits)
        dist_troop = torch.distributions.Categorical(logits=out["troop"])

        if action is None:
            a_target = dist_target.sample()
            a_troop = dist_troop.sample()
            action = torch.stack([a_type, a_target, a_troop], dim=-1)
        else:
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


def compute_gae_vec(rewards, values, terminals, truncateds, last_values,
                    truncated_values=None, gamma=0.99, lam=0.95):
    """Compute GAE for vectorized rollouts with proper truncation handling.

    Terminal episodes (death) zero out bootstrapping.
    Truncated episodes (max_steps) bootstrap from the value estimate,
    because the episode was artificially cut short.

    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        terminals: (num_steps, num_envs) - true episode ends (death/win)
        truncateds: (num_steps, num_envs) - forced episode ends (max_steps)
        last_values: (num_envs,) bootstrap values for last step
        truncated_values: (num_steps, num_envs) - value estimates at truncation points
    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(num_envs)

    # Only terminal deaths should zero out bootstrapping
    # Truncations should bootstrap from the value function
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]

        # At truncation, bootstrap from value estimate of the truncated state
        if truncated_values is not None:
            next_values = np.where(truncateds[t], truncated_values[t], next_values)

        # Only terminals zero out the future (not truncations)
        not_terminal = 1.0 - terminals[t]
        delta = rewards[t] + gamma * next_values * not_terminal - values[t]
        last_gae = delta + gamma * lam * not_terminal * last_gae
        # Reset GAE at truncation boundaries too (new episode starts)
        last_gae = np.where(truncateds[t], delta, last_gae)
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

    # Load weights only (fresh optimizer) from a specific checkpoint
    if args.load_weights:
        wpath = Path(args.load_weights)
        print(f"Loading weights from: {wpath}")
        checkpoint = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        print(f"Weights loaded (fresh optimizer, starting from update 0)")

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
    land_masks_buf = np.ones((args.rollout_steps, args.num_envs, max_neighbors), dtype=np.float32)
    sea_masks_buf = np.ones((args.rollout_steps, args.num_envs, max_neighbors), dtype=np.float32)
    actions_buf = np.zeros((args.rollout_steps, args.num_envs, 3), dtype=np.float32)
    logprobs_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    terminals_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    truncateds_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    truncated_values_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
    values_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)

    # Track per-env episode stats
    env_ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    env_ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    # Initialize
    obs, action_masks, land_target_masks, sea_target_masks = envs.reset_all()

    # Curriculum: gradual ramp of (difficulty, opponents, maps)
    # Each step is a small increment to avoid distributional shock
    CURRICULUM_STAGES = [
        # (progress_threshold, difficulty, opponents, maps, max_steps)
        (0.05, "Easy",   2,  MAP_TIER_1, 10000),
        (0.10, "Easy",   3,  MAP_TIER_1, 15000),
        (0.15, "Medium", 3,  MAP_TIER_2, 20000),
        (0.20, "Medium", 4,  MAP_TIER_2, 25000),
        (0.30, "Medium", 5,  MAP_TIER_2, 30000),
        (0.38, "Hard",   5,  MAP_TIER_3, 40000),
        (0.45, "Hard",   6,  MAP_TIER_3, 50000),
        (0.50, "Hard",   8,  MAP_TIER_3, 60000),
        (0.65, "Hard",  10,  MAP_TIER_3, 80000),
        (1.00, "Hard",  12,  MAP_TIER_3, 100000),
    ]
    # LR warmdown: after a curriculum transition, temporarily reduce LR
    # then ramp back up over WARMDOWN_UPDATES
    WARMDOWN_UPDATES = 50
    warmdown_counter = 0
    WARMDOWN_FACTOR = 0.3  # drop LR to 30% at transition, ramp back to 100%

    for update in range(start_update, args.num_updates):
        t_start = time.time()

        # Curriculum learning: gradual ramp
        if args.curriculum:
            progress = update / args.num_updates
            # Find current stage
            for thresh, diff, opp, maps, msteps in CURRICULUM_STAGES:
                if progress < thresh:
                    new_diff, new_opp, new_maps, new_max_steps = diff, opp, maps, msteps
                    break
            else:
                s = CURRICULUM_STAGES[-1]
                new_diff, new_opp, new_maps, new_max_steps = s[1], s[2], s[3], s[4]

            if envs.difficulty != new_diff or envs.num_opponents != new_opp:
                print(f"  Curriculum: switching to {new_diff} with {new_opp} opponents, {len(new_maps)} maps, max_steps={new_max_steps} (progress={progress:.0%})")
                envs.difficulty = new_diff
                envs.num_opponents = new_opp
                warmdown_counter = WARMDOWN_UPDATES  # trigger LR warmdown
            if len(envs.maps) != len(new_maps):
                envs.maps = new_maps
            envs.max_steps = new_max_steps

        # LR schedule: apply warmdown after curriculum transitions
        lr_now = args.lr
        if args.anneal_lr:
            frac = 1.0 - update / args.num_updates
            lr_now = frac * args.lr
        if warmdown_counter > 0:
            # Linearly ramp from WARMDOWN_FACTOR back to 1.0
            warmdown_frac = WARMDOWN_FACTOR + (1.0 - WARMDOWN_FACTOR) * (1.0 - warmdown_counter / WARMDOWN_UPDATES)
            lr_now *= warmdown_frac
            warmdown_counter -= 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # Collect rollout
        for step in range(args.rollout_steps):
            obs_buf[step] = obs
            masks_buf[step] = action_masks
            land_masks_buf[step] = land_target_masks
            sea_masks_buf[step] = sea_target_masks

            obs_t = torch.FloatTensor(obs).to(device)
            masks_t = torch.FloatTensor(action_masks).to(device)
            land_t = torch.FloatTensor(land_target_masks).to(device)
            sea_t = torch.FloatTensor(sea_target_masks).to(device)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(
                    obs_t, action_mask=masks_t,
                    land_target_mask=land_t, sea_target_mask=sea_t)

            actions_np = action.cpu().numpy()  # (num_envs, 3)
            logprobs_buf[step] = log_prob.cpu().numpy()
            values_buf[step] = value.cpu().numpy()
            actions_buf[step] = actions_np

            next_obs, next_masks, next_land, next_sea, rewards, dones, truncateds, infos = envs.step(actions_np)
            rewards_buf[step] = rewards
            terminals_buf[step] = dones.astype(np.float32)
            truncateds_buf[step] = truncateds.astype(np.float32)

            # For truncated envs, estimate the value of the terminal state
            # before resetting — this is the bootstrap target
            if np.any(truncateds):
                trunc_obs = torch.FloatTensor(next_obs[truncateds]).to(device)
                with torch.no_grad():
                    trunc_vals = model.get_action_and_value(trunc_obs)[3]
                truncated_values_buf[step, truncateds] = trunc_vals.cpu().numpy()

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
                    next_obs[i], next_masks[i], next_land[i], next_sea[i] = envs.reset_single(i)

            global_step += args.num_envs
            obs = next_obs
            action_masks = next_masks
            land_target_masks = next_land
            sea_target_masks = next_sea

        # Bootstrap values for last step
        with torch.no_grad():
            last_values = model.get_action_and_value(torch.FloatTensor(obs).to(device))[3]
            last_values = last_values.cpu().numpy()

        # Compute GAE with proper truncation bootstrapping
        advantages, returns = compute_gae_vec(
            rewards_buf, values_buf, terminals_buf, truncateds_buf,
            last_values, truncated_values_buf,
            gamma=args.gamma, lam=args.gae_lambda,
        )

        # Flatten (num_steps, num_envs, ...) -> (batch_size, ...)
        batch_size = args.rollout_steps * args.num_envs
        b_obs = torch.FloatTensor(obs_buf.reshape(batch_size, -1)).to(device)
        b_masks = torch.FloatTensor(masks_buf.reshape(batch_size, -1)).to(device)
        b_land_masks = torch.FloatTensor(land_masks_buf.reshape(batch_size, -1)).to(device)
        b_sea_masks = torch.FloatTensor(sea_masks_buf.reshape(batch_size, -1)).to(device)
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
                    b_obs[mb_idx], b_actions[mb_idx], action_mask=b_masks[mb_idx],
                    land_target_mask=b_land_masks[mb_idx],
                    sea_target_mask=b_sea_masks[mb_idx],
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
    parser.add_argument("--max-steps", type=int, default=100000)
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
    parser.add_argument("--load-weights", default="", help="Load model weights only (fresh optimizer) from a .pt file")
    parser.add_argument("--hf-repo", default="mischievers/openfront-rl-agent", help="HuggingFace repo for auto-push")
    args = parser.parse_args()
    train(args)
