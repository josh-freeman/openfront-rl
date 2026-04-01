"""
OpenFront RL Gymnasium Environment

Wraps the TypeScript headless game server via stdin/stdout JSON protocol.
"""

import json
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Any

# Action types
ACTION_NOOP = 0
ACTION_ATTACK = 1
ACTION_BUILD_CITY = 2
ACTION_BUILD_FACTORY = 3
ACTION_BUILD_DEFENSE = 4
ACTION_BUILD_PORT = 5
ACTION_BUILD_SAM = 6
ACTION_BUILD_SILO = 7
NUM_ACTIONS = 8

BUILD_TYPES = {
    ACTION_BUILD_CITY: "city",
    ACTION_BUILD_FACTORY: "factory",
    ACTION_BUILD_DEFENSE: "defense_post",
    ACTION_BUILD_PORT: "port",
    ACTION_BUILD_SAM: "sam_launcher",
    ACTION_BUILD_SILO: "missile_silo",
}


class OpenFrontEnv(gym.Env):
    """
    OpenFront RL environment.

    Observation space: Dict with player stats, neighbor info, etc.
    Action space: Discrete(8) x Discrete(max_neighbors) x Continuous(troop_fraction)

    For simplicity, we use a flattened observation vector and MultiDiscrete actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "plains",
        num_opponents: int = 3,
        difficulty: str = "Medium",
        ticks_per_step: int = 10,
        max_steps: int = 3000,
        max_neighbors: int = 8,
    ):
        super().__init__()

        self.map_name = map_name
        self.num_opponents = num_opponents
        self.difficulty = difficulty
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.max_neighbors = max_neighbors
        self.step_count = 0

        # Observation: fixed-size vector
        # [myTiles, myTroops, myGold, territoryPct, incomingAttacks, outgoingAttacks,
        #  numUnits, numNeighbors,
        #  neighbor_0_tiles, neighbor_0_troops, neighbor_0_relation, ...,
        #  neighbor_N_tiles, neighbor_N_troops, neighbor_N_relation]
        obs_size = 8 + max_neighbors * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: [action_type, target_neighbor_idx, troop_fraction_bucket]
        # action_type: 0=noop, 1=attack, 2-7=build types
        # target_neighbor: 0..max_neighbors-1 (for attacks)
        # troop_fraction: 0..4 -> [0.2, 0.4, 0.6, 0.8, 1.0]
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS, max_neighbors, 5]
        )

        self._proc = None
        self._neighbors_cache = []

    def _start_server(self):
        if self._proc is not None:
            self._proc.kill()

        rl_dir = Path(__file__).parent
        repo_dir = rl_dir.parent

        self._proc = subprocess.Popen(
            ["npx", "tsx", str(rl_dir / "env_server.ts")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(repo_dir),
            text=True,
            bufsize=1,
        )

        # Wait for "ready" message
        line = self._proc.stdout.readline()
        msg = json.loads(line)
        assert msg.get("status") == "ready", f"Server not ready: {msg}"

    def _send(self, msg: dict) -> dict:
        assert self._proc is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Server closed connection")
        return json.loads(line)

    def _obs_to_vec(self, obs: dict) -> np.ndarray:
        """Convert observation dict to fixed-size float vector."""
        vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        vec[0] = obs.get("myTiles", 0) / max(obs.get("totalMapTiles", 1), 1)
        vec[1] = obs.get("myTroops", 0) / 100000  # normalize
        vec[2] = obs.get("myGold", 0) / 1000000
        vec[3] = obs.get("territoryPct", 0)
        vec[4] = obs.get("incomingAttacks", 0) / 10
        vec[5] = obs.get("outgoingAttacks", 0) / 10
        vec[6] = len(obs.get("units", [])) / 20
        vec[7] = len(obs.get("neighbors", [])) / self.max_neighbors

        # Neighbor features
        neighbors = obs.get("neighbors", [])
        self._neighbors_cache = neighbors
        for i, n in enumerate(neighbors[: self.max_neighbors]):
            base = 8 + i * 3
            total = max(obs.get("totalMapTiles", 1), 1)
            vec[base + 0] = n.get("tiles", 0) / total
            vec[base + 1] = n.get("troops", 0) / 100000
            vec[base + 2] = n.get("relation", 0) / 3  # 0=hostile..3=friendly

        return vec

    def _decode_action(self, action: np.ndarray) -> dict:
        """Convert MultiDiscrete action to game action dict."""
        action_type = int(action[0])
        target_idx = int(action[1])
        troop_bucket = int(action[2])
        troop_fraction = (troop_bucket + 1) * 0.2  # 0.2, 0.4, 0.6, 0.8, 1.0

        if action_type == ACTION_NOOP:
            return {"type": "noop"}
        elif action_type == ACTION_ATTACK:
            if target_idx < len(self._neighbors_cache):
                target = self._neighbors_cache[target_idx]
                return {
                    "type": "attack",
                    "targetPlayerId": target["id"],
                    "troopFraction": troop_fraction,
                }
            return {"type": "noop"}
        elif action_type in BUILD_TYPES:
            return {
                "type": "build",
                "unitType": BUILD_TYPES[action_type],
            }
        return {"type": "noop"}

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_count = 0

        if self._proc is None:
            self._start_server()

        resp = self._send({
            "cmd": "reset",
            "config": {
                "map": self.map_name,
                "numOpponents": self.num_opponents,
                "difficulty": self.difficulty,
            },
        })

        obs = self._obs_to_vec(resp["obs"])
        info = resp.get("info", {})
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        game_action = self._decode_action(action)

        resp = self._send({
            "cmd": "step",
            "action": game_action,
            "ticksPerStep": self.ticks_per_step,
        })

        obs = self._obs_to_vec(resp["obs"])
        reward = float(resp.get("reward", 0))
        done = bool(resp.get("done", False))
        truncated = self.step_count >= self.max_steps
        info = resp.get("info", {})

        return obs, reward, done, truncated, info

    def close(self):
        if self._proc:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            self._proc.kill()
            self._proc = None
