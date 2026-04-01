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


def load_model(model_path: str, obs_dim: int, max_neighbors: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, max_neighbors).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, device


def play_headless(args):
    """Play a full game headlessly and print results."""
    env = OpenFrontEnv(
        map_name=args.map,
        num_opponents=args.opponents,
        difficulty=args.difficulty,
        ticks_per_step=args.ticks_per_step,
    )
    obs_dim = env.observation_space.shape[0]
    model, device = load_model(args.model, obs_dim)

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, value = model.get_action_and_value(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, done, truncated, info = env.step(action_np)
        total_reward += reward
        steps += 1

        if steps % 100 == 0:
            print(f"  Step {steps}: reward={total_reward:.2f}, info={info}")

        if done or truncated:
            break

    print(f"\nGame over after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Winner: {info.get('winner', 'none')}")
    print(f"We won: {info.get('weWon', False)}")
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
        obs_vec = np.zeros(32, dtype=np.float32)  # 8 + 8*3
        obs_vec[0] = obs_dict.get("myTiles", 0) / max(obs_dict.get("totalMapTiles", 1), 1)
        obs_vec[1] = obs_dict.get("myTroops", 0) / 100000
        obs_vec[2] = obs_dict.get("myGold", 0) / 1000000
        obs_vec[3] = obs_dict.get("territoryPct", 0)
        obs_vec[4] = obs_dict.get("incomingAttacks", 0) / 10
        obs_vec[5] = obs_dict.get("outgoingAttacks", 0) / 10
        obs_vec[6] = len(obs_dict.get("units", [])) / 20
        neighbors = obs_dict.get("neighbors", [])
        obs_vec[7] = len(neighbors) / 8
        for i, n in enumerate(neighbors[:8]):
            base = 8 + i * 3
            obs_vec[base] = n.get("tiles", 0) / max(obs_dict.get("totalMapTiles", 1), 1)
            obs_vec[base + 1] = n.get("troops", 0) / 100000
            obs_vec[base + 2] = n.get("relation", 0) / 3

        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_t)

        action_np = action.squeeze(0).cpu().numpy().tolist()

        # Decode action to game-friendly format
        action_type = int(action_np[0])
        target_idx = int(action_np[1])
        troop_bucket = int(action_np[2])

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
    model, device = load_model(args.model, 32)
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
    args = parser.parse_args()

    if args.mode == "headless":
        play_headless(args)
    else:
        serve_policy(args)
