"""
Play OpenFront with a trained RL model.

Can be used in two modes:
1. Headless (for evaluation): plays against bots in the headless engine
2. Live (for deployment): exports actions as a JSON policy server
   that the Puppeteer bot can query

Usage:
  # Headless evaluation
  python play.py --model checkpoints/best_model.pt --mode headless

  # Policy server for live play (the Puppeteer bot queries this)
  python play.py --model checkpoints/best_model.pt --mode server --port 8765
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

from env import OpenFrontEnv, NUM_ACTIONS
from train import ActorCritic


def load_model(model_path: str, obs_dim: int, max_neighbors: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    # Infer hidden sizes from backbone weights
    hidden_sizes = []
    i = 0
    while f"backbone.{i}.weight" in state_dict:
        hidden_sizes.append(state_dict[f"backbone.{i}.weight"].shape[0])
        i += 2  # skip ReLU layers (no params, but Sequential indexes them)
    if not hidden_sizes:
        hidden_sizes = [256, 256, 128]
    model = ActorCritic(obs_dim, max_neighbors, hidden_sizes=hidden_sizes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def play_headless(args):
    """Play multiple games headlessly and report aggregate stats."""
    env = OpenFrontEnv(
        map_name=args.map,
        num_opponents=args.opponents,
        difficulty=args.difficulty,
        ticks_per_step=args.ticks_per_step,
    )
    obs_dim = env.observation_space.shape[0]
    model, device = load_model(args.model, obs_dim)

    wins = 0
    survival_steps = []
    territory_pcts = []

    for game in range(args.num_games):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_territory = 0

        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            # Use action mask from info if available
            mask_t = None
            land_t = None
            sea_t = None
            if "action_mask" in info:
                mask_arr = np.array(info["action_mask"][:17], dtype=np.float32)
                mask_arr[0] = 1.0
                mask_t = torch.FloatTensor(mask_arr).unsqueeze(0).to(device)
            if "land_target_mask" in info:
                land_t = torch.FloatTensor(info["land_target_mask"]).unsqueeze(0).to(device)
            if "sea_target_mask" in info:
                sea_t = torch.FloatTensor(info["sea_target_mask"]).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, value = model.get_action_and_value(
                    obs_t, action_mask=mask_t,
                    land_target_mask=land_t, sea_target_mask=sea_t)
            action_np = action.squeeze(0).cpu().numpy()

            if steps % 20 == 0 and game == 0:
                act_type = int(action_np[0])
                act_names = ["NOOP","ATK","BOAT","RET","CITY","FAC","DEF","PORT","SAM","SILO","WSHIP","ATOM","HBOMB","MIRV","MVWSH","UPG","DEL"]
                name = act_names[act_type] if act_type < len(act_names) else str(act_type)
                troops = obs[1] * 100000
                gold = obs[2] * 1000000
                pct = obs[3] * 100
                print(f"    [{steps:4d}] {name:5s} tgt={int(action_np[1])} | troops={troops:.0f} gold={gold:.0f} terr={pct:.2f}% val={value.item():.1f}")

            obs, reward, done, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1
            territory = info.get("territoryPct", obs[3])
            if territory > max_territory:
                max_territory = territory

            if done or truncated:
                break

        won = info.get("weWon", False)
        if won:
            wins += 1
        survival_steps.append(steps)
        territory_pcts.append(max_territory)
        print(f"  Game {game+1}/{args.num_games}: steps={steps}, reward={total_reward:.2f}, won={won}, territory={max_territory:.1%}")

    print(f"\n{'='*50}")
    print(f"Evaluation: {args.num_games} games on map={args.map}, opponents={args.opponents}")
    print(f"  Win rate:          {wins}/{args.num_games} ({wins/args.num_games:.1%})")
    print(f"  Avg survival:      {np.mean(survival_steps):.0f} steps")
    print(f"  Avg max territory: {np.mean(territory_pcts):.1%}")
    print(f"{'='*50}")
    env.close()


class PolicyHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves model predictions for the Puppeteer bot."""

    model = None
    device = None

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        obs_dict = json.loads(body)

        # Convert observation dict to vector (same as env.py)
        obs_vec = np.zeros(80, dtype=np.float32)  # 16 + 16*4
        total = max(obs_dict.get("totalMapTiles", 1), 1)
        obs_vec[0] = obs_dict.get("myTiles", 0) / total
        obs_vec[1] = obs_dict.get("myTroops", 0) / 100000
        obs_vec[2] = obs_dict.get("myGold", 0) / 1000000
        obs_vec[3] = obs_dict.get("territoryPct", 0)
        obs_vec[4] = obs_dict.get("incomingAttacks", 0) / 10
        obs_vec[5] = obs_dict.get("outgoingAttacks", 0) / 10
        obs_vec[6] = len(obs_dict.get("units", [])) / 20
        neighbors = obs_dict.get("neighbors", [])
        obs_vec[7] = len(neighbors) / 16
        obs_vec[8] = float(obs_dict.get("hasSilo", False))
        obs_vec[9] = float(obs_dict.get("hasPort", False))
        obs_vec[10] = float(obs_dict.get("hasSAM", False))
        obs_vec[11] = obs_dict.get("numWarships", 0) / 5
        obs_vec[12] = obs_dict.get("numNukes", 0) / 5
        obs_vec[13] = obs_dict.get("tick", 0) / 100000

        # Build feedback
        build_result = obs_dict.get("lastBuildResult", "none")
        if build_result == "success":
            obs_vec[14] = 1.0
        elif build_result == "none":
            obs_vec[14] = 0.0
        else:
            obs_vec[14] = -1.0
        obs_vec[15] = float(obs_dict.get("lastActionSucceeded", False))

        for i, n in enumerate(neighbors[:16]):
            base = 16 + i * 4
            obs_vec[base] = n.get("tiles", 0) / total
            obs_vec[base + 1] = n.get("troops", 0) / 100000
            obs_vec[base + 2] = n.get("relation", 0) / 3
            obs_vec[base + 3] = float(n.get("isLandNeighbor", True))

        # Extract action mask if provided
        action_mask = obs_dict.get("actionMask", None)
        mask_t = None
        if action_mask is not None:
            mask_arr = np.array(action_mask[:17], dtype=np.float32)
            mask_arr[0] = 1.0  # NOOP always valid
            mask_t = torch.FloatTensor(mask_arr).unsqueeze(0).to(self.device)

        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)

        # Build target masks from neighbor data (matches env_server.ts logic)
        land_mask = np.zeros(16, dtype=np.float32)
        sea_mask = np.zeros(16, dtype=np.float32)
        for i, n in enumerate(neighbors[:16]):
            not_allied = not n.get("isAllied", False)
            if n.get("isLandNeighbor", False) and not_allied:
                land_mask[i] = 1.0
            if not_allied:
                sea_mask[i] = 1.0
        land_t = torch.FloatTensor(land_mask).unsqueeze(0).to(self.device)
        sea_t = torch.FloatTensor(sea_mask).unsqueeze(0).to(self.device)

        # Debug: log raw dict values AND obs vector
        with torch.no_grad():
            out = self.model.forward(obs_t)
            raw_logits = out["action_type"].squeeze(0).cpu().numpy()
            with open("/tmp/policy_debug.log", "a") as f:
                # Log raw values from bot
                f.write(f"RAW: tiles={obs_dict.get('myTiles')} troops={obs_dict.get('myTroops')} gold={obs_dict.get('myGold')} pct={obs_dict.get('territoryPct')} total={obs_dict.get('totalMapTiles')} tick={obs_dict.get('tick')} in={obs_dict.get('incomingAttacks')} out={obs_dict.get('outgoingAttacks')} units={obs_dict.get('units')} nNeigh={len(neighbors)}\n")
                # Log normalized obs vector
                player = obs_vec[:16]
                f.write(f"VEC: [{', '.join(f'{v:.5f}' for v in player)}]\n")
                # Log first 3 neighbors
                for ni in range(min(3, len(neighbors))):
                    base = 16 + ni * 4
                    n = neighbors[ni]
                    f.write(f"  n{ni}: raw=[tiles={n.get('tiles')} troops={n.get('troops')} rel={n.get('relation')} alive={n.get('alive')} land={n.get('isLandNeighbor')}] vec=[{obs_vec[base]:.5f}, {obs_vec[base+1]:.4f}, {obs_vec[base+2]:.2f}, {obs_vec[base+3]:.0f}]\n")
                if mask_t is not None:
                    mask_np = mask_t.squeeze(0).cpu().numpy()
                    masked_logits = raw_logits + (1 - mask_np) * (-1e8)
                    top3 = sorted(enumerate(masked_logits), key=lambda x: -x[1])[:3]
                    f.write(f"mask={mask_np.astype(int).tolist()} top3={[(i, f'{v:.1f}') for i,v in top3]}\n")
                f.write(f"land_mask={land_mask.astype(int).tolist()} sea_mask={sea_mask.astype(int).tolist()}\n")
            action, _, _, _ = self.model.get_action_and_value(
                obs_t, action_mask=mask_t,
                land_target_mask=land_t, sea_target_mask=sea_t)

        action_np = action.squeeze(0).cpu().numpy().tolist()

        # Decode action to game-friendly format
        action_type = int(action_np[0])
        target_idx = int(action_np[1])
        troop_bucket = int(action_np[2])
        with open("/tmp/policy_debug.log", "a") as f:
            f.write(f"→ type={action_type} target={target_idx} bucket={troop_bucket}\n\n")

        response = {
            "actionType": action_type,
            "targetIdx": target_idx,
            "troopFraction": (troop_bucket + 1) * 0.2,
            "raw": action_np,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def serve_policy(args):
    """Run HTTP policy server for live Puppeteer bot play."""
    model, device = load_model(args.model, 80)
    PolicyHandler.model = model
    PolicyHandler.device = device

    server = HTTPServer(("0.0.0.0", args.port), PolicyHandler)
    print(f"Policy server running on http://0.0.0.0:{args.port}")
    print("The Puppeteer bot can POST observations and get actions back.")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["headless", "server"], default="headless")
    parser.add_argument("--map", default="plains")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-games", type=int, default=10, help="Number of games for headless evaluation")
    args = parser.parse_args()

    if args.mode == "headless":
        play_headless(args)
    else:
        serve_policy(args)
