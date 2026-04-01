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
ACTION_BOAT_ATTACK = 2
ACTION_RETREAT = 3
ACTION_BUILD_CITY = 4
ACTION_BUILD_FACTORY = 5
ACTION_BUILD_DEFENSE = 6
ACTION_BUILD_PORT = 7
ACTION_BUILD_SAM = 8
ACTION_BUILD_SILO = 9
ACTION_BUILD_WARSHIP = 10
ACTION_LAUNCH_ATOM = 11
ACTION_LAUNCH_HBOMB = 12
ACTION_LAUNCH_MIRV = 13
ACTION_MOVE_WARSHIP = 14
ACTION_UPGRADE = 15
ACTION_DELETE_UNIT = 16
NUM_ACTIONS = 17

BUILD_TYPES = {
    ACTION_BUILD_CITY: "city",
    ACTION_BUILD_FACTORY: "factory",
    ACTION_BUILD_DEFENSE: "defense_post",
    ACTION_BUILD_PORT: "port",
    ACTION_BUILD_SAM: "sam_launcher",
    ACTION_BUILD_SILO: "missile_silo",
    ACTION_BUILD_WARSHIP: "warship",
}

NUKE_TYPES = {
    ACTION_LAUNCH_ATOM: "atom_bomb",
    ACTION_LAUNCH_HBOMB: "hydrogen_bomb",
    ACTION_LAUNCH_MIRV: "mirv",
}


class OpenFrontEnv(gym.Env):
    """
    OpenFront RL environment.

    Observation: fixed-size vector with player stats, unit counts, neighbor info.
    Action: MultiDiscrete [action_type, target_idx, troop_fraction_bucket]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "plains",
        num_opponents: int = 3,
        difficulty: str = "Medium",
        ticks_per_step: int = 10,
        max_steps: int = 10000,
        max_neighbors: int = 16,
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
        # [8 player stats] + [max_neighbors * 3 neighbor features]
        # Player stats:
        #   0: territoryPct (myTiles / totalMapTiles)
        #   1: myTroops (normalized)
        #   2: myGold (normalized)
        #   3: territoryPct (raw from server)
        #   4: incomingAttacks (normalized)
        #   5: outgoingAttacks (normalized)
        #   6: numUnits (normalized)
        #   7: numNeighbors (normalized)
        #   8: hasSilo (0/1)
        #   9: hasPort (0/1)
        #   10: hasSAM (0/1)
        #   11: numWarships (normalized)
        #   12: numNukes (normalized)
        #   13: tickProgress (normalized by max ticks)
        #   14: lastBuildSuccess (1=success, 0=none, -1=fail)
        #   15: lastActionSucceeded (0/1)
        #   16-22: canAfford flags (city, defense, factory, port, silo, SAM, warship)
        # Neighbor features (per neighbor):
        #   0: tiles (normalized)
        #   1: troops (normalized)
        #   2: relation (normalized)
        #   3: isAlive (0/1)
        obs_size = 23 + max_neighbors * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

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
        total = max(obs.get("totalMapTiles", 1), 1)

        # Player stats
        vec[0] = obs.get("myTiles", 0) / total
        vec[1] = obs.get("myTroops", 0) / 100000
        vec[2] = obs.get("myGold", 0) / 1000000
        vec[3] = obs.get("territoryPct", 0)
        vec[4] = obs.get("incomingAttacks", 0) / 10
        vec[5] = obs.get("outgoingAttacks", 0) / 10
        vec[6] = len(obs.get("units", [])) / 20
        neighbors = obs.get("neighbors", [])
        vec[7] = len(neighbors) / self.max_neighbors
        vec[8] = float(obs.get("hasSilo", False))
        vec[9] = float(obs.get("hasPort", False))
        vec[10] = float(obs.get("hasSAM", False))
        vec[11] = obs.get("numWarships", 0) / 5
        vec[12] = obs.get("numNukes", 0) / 5
        vec[13] = obs.get("tick", 0) / 100000

        # Build feedback
        build_result = obs.get("lastBuildResult", "none")
        if build_result == "success":
            vec[14] = 1.0
        elif build_result == "none":
            vec[14] = 0.0
        else:
            vec[14] = -1.0  # any failure
        vec[15] = float(obs.get("lastActionSucceeded", False))

        # Affordability flags — what can we actually build right now?
        vec[16] = float(obs.get("canAffordCity", False))
        vec[17] = float(obs.get("canAffordDefense", False))
        vec[18] = float(obs.get("canAffordFactory", False))
        vec[19] = float(obs.get("canAffordPort", False))
        vec[20] = float(obs.get("canAffordSilo", False))
        vec[21] = float(obs.get("canAffordSAM", False))
        vec[22] = float(obs.get("canAffordWarship", False))

        # Neighbor features
        self._neighbors_cache = neighbors
        for i, n in enumerate(neighbors[: self.max_neighbors]):
            base = 23 + i * 4
            vec[base] = n.get("tiles", 0) / total
            vec[base + 1] = n.get("troops", 0) / 100000
            vec[base + 2] = n.get("relation", 0) / 3
            vec[base + 3] = float(n.get("alive", True))

        return vec

    def _decode_action(self, action: np.ndarray) -> dict:
        """Convert MultiDiscrete action to game action dict."""
        action_type = int(action[0])
        target_idx = int(action[1])
        troop_bucket = int(action[2])
        troop_fraction = (troop_bucket + 1) * 0.2

        if action_type == ACTION_NOOP:
            return {"type": "noop"}
        elif action_type in (ACTION_ATTACK, ACTION_BOAT_ATTACK):
            if target_idx < len(self._neighbors_cache):
                target = self._neighbors_cache[target_idx]
                return {
                    "type": "boat_attack" if action_type == ACTION_BOAT_ATTACK else "attack",
                    "targetPlayerId": target["id"],
                    "troopFraction": troop_fraction,
                }
            return {"type": "noop"}
        elif action_type == ACTION_RETREAT:
            return {"type": "retreat"}
        elif action_type in BUILD_TYPES:
            return {"type": "build", "unitType": BUILD_TYPES[action_type]}
        elif action_type in NUKE_TYPES:
            if target_idx < len(self._neighbors_cache):
                target = self._neighbors_cache[target_idx]
                return {
                    "type": "launch_nuke",
                    "nukeType": NUKE_TYPES[action_type],
                    "targetPlayerId": target["id"],
                }
            return {"type": "noop"}
        elif action_type == ACTION_MOVE_WARSHIP:
            if target_idx < len(self._neighbors_cache):
                target = self._neighbors_cache[target_idx]
                return {
                    "type": "move_warship",
                    "targetPlayerId": target["id"],
                }
            return {"type": "noop"}
        elif action_type == ACTION_UPGRADE:
            return {"type": "upgrade"}
        elif action_type == ACTION_DELETE_UNIT:
            return {"type": "delete_unit"}
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
