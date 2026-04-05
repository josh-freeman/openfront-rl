"""
Vectorized OpenFront Environment

Runs N game server processes in parallel for faster rollout collection.
Each env has its own TypeScript server subprocess.
"""

import json
import subprocess
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from env import (
    NUM_ACTIONS, BUILD_TYPES, NUKE_TYPES,
    ACTION_NOOP, ACTION_ATTACK, ACTION_BOAT_ATTACK, ACTION_RETREAT,
    ACTION_MOVE_WARSHIP, ACTION_UPGRADE, ACTION_DELETE_UNIT,
)


class VecOpenFrontEnv:
    """Vectorized environment running N game servers in parallel."""

    def __init__(
        self,
        num_envs: int = 4,
        maps: list[str] = None,
        num_opponents: int = 3,
        difficulty: str = "Medium",
        ticks_per_step: int = 10,
        max_steps: int = 10000,
        max_neighbors: int = 16,
    ):
        self.num_envs = num_envs
        self.maps = maps or ["plains"]
        self.num_opponents = num_opponents
        self.difficulty = difficulty
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.max_neighbors = max_neighbors

        # 16 player stats + max_neighbors * 5 neighbor features
        obs_size = 16 + max_neighbors * 5
        self.obs_dim = obs_size

        self._procs: list[Optional[subprocess.Popen]] = [None] * num_envs
        self._neighbors_caches: list[list] = [[] for _ in range(num_envs)]
        self._step_counts = np.zeros(num_envs, dtype=np.int32)

        rl_dir = Path(__file__).parent
        self._repo_dir = str(rl_dir.parent)
        self._server_script = str(rl_dir / "env_server.ts")

        self._pool = ThreadPoolExecutor(max_workers=num_envs)
        self._start_all()

    def _start_server(self, idx: int):
        if self._procs[idx] is not None:
            try:
                self._procs[idx].kill()
            except Exception:
                pass

        proc = subprocess.Popen(
            ["npx", "tsx", self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._repo_dir,
            text=True,
            bufsize=1,
        )
        line = proc.stdout.readline()
        msg = json.loads(line)
        assert msg.get("status") == "ready", f"Server {idx} not ready: {msg}"
        self._procs[idx] = proc

    def _start_all(self):
        for i in range(self.num_envs):
            self._start_server(i)

    def _send(self, idx: int, msg: dict) -> dict:
        proc = self._procs[idx]
        assert proc is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Server {idx} closed connection")
        return json.loads(line)

    def _obs_to_vec(self, obs: dict, idx: int) -> np.ndarray:
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        total = max(obs.get("totalMapTiles", 1), 1)

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
            vec[14] = -1.0
        vec[15] = float(obs.get("lastActionSucceeded", False))

        self._neighbors_caches[idx] = neighbors
        for i, n in enumerate(neighbors[: self.max_neighbors]):
            base = 16 + i * 5
            vec[base] = n.get("tiles", 0) / total
            vec[base + 1] = n.get("troops", 0) / 100000
            vec[base + 2] = n.get("relation", 0) / 3
            vec[base + 3] = float(n.get("isLandNeighbor", True))
            vec[base + 4] = n.get("distance", 1.0)
        return vec

    def _extract_action_mask(self, obs: dict) -> np.ndarray:
        """Extract 17-element action mask from observation dict."""
        mask = obs.get("actionMask", [True] * NUM_ACTIONS)
        arr = np.array(mask[:NUM_ACTIONS], dtype=np.float32)
        if len(arr) < NUM_ACTIONS:
            arr = np.concatenate([arr, np.ones(NUM_ACTIONS - len(arr), dtype=np.float32)])
        arr[0] = 1.0  # NOOP always valid
        return arr

    def _extract_target_masks(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-target masks for land and sea actions."""
        land = obs.get("landTargetMask", [1] * self.max_neighbors)
        sea = obs.get("seaTargetMask", [1] * self.max_neighbors)
        land_arr = np.zeros(self.max_neighbors, dtype=np.float32)
        sea_arr = np.zeros(self.max_neighbors, dtype=np.float32)
        for i in range(min(len(land), self.max_neighbors)):
            land_arr[i] = float(land[i])
        for i in range(min(len(sea), self.max_neighbors)):
            sea_arr[i] = float(sea[i])
        return land_arr, sea_arr

    def _decode_action(self, action: np.ndarray, idx: int) -> dict:
        action_type = int(action[0])
        target_idx = int(action[1])
        troop_bucket = int(action[2])
        troop_fraction = (troop_bucket + 1) * 0.2
        neighbors = self._neighbors_caches[idx]

        if action_type == ACTION_NOOP:
            return {"type": "noop"}
        elif action_type in (ACTION_ATTACK, ACTION_BOAT_ATTACK):
            if target_idx < len(neighbors):
                return {
                    "type": "boat_attack" if action_type == ACTION_BOAT_ATTACK else "attack",
                    "targetPlayerId": neighbors[target_idx]["id"],
                    "troopFraction": troop_fraction,
                }
            return {"type": "noop"}
        elif action_type == ACTION_RETREAT:
            return {"type": "retreat"}
        elif action_type in BUILD_TYPES:
            return {"type": "build", "unitType": BUILD_TYPES[action_type]}
        elif action_type in NUKE_TYPES:
            if target_idx < len(neighbors):
                return {
                    "type": "launch_nuke",
                    "nukeType": NUKE_TYPES[action_type],
                    "targetPlayerId": neighbors[target_idx]["id"],
                }
            return {"type": "noop"}
        elif action_type == ACTION_MOVE_WARSHIP:
            if target_idx < len(neighbors):
                return {
                    "type": "move_warship",
                    "targetPlayerId": neighbors[target_idx]["id"],
                }
            return {"type": "noop"}
        elif action_type == ACTION_UPGRADE:
            return {"type": "upgrade"}
        elif action_type == ACTION_DELETE_UNIT:
            return {"type": "delete_unit"}
        return {"type": "noop"}

    def reset_single(self, idx: int):
        """Returns (obs_vec, action_mask, land_target_mask, sea_target_mask)."""
        self._step_counts[idx] = 0
        map_name = random.choice(self.maps)
        try:
            resp = self._send(idx, {
                "cmd": "reset",
                "config": {
                    "map": map_name,
                    "numOpponents": self.num_opponents,
                    "difficulty": self.difficulty,
                },
            })
        except (RuntimeError, BrokenPipeError):
            self._start_server(idx)
            resp = self._send(idx, {
                "cmd": "reset",
                "config": {
                    "map": map_name,
                    "numOpponents": self.num_opponents,
                    "difficulty": self.difficulty,
                },
            })
        land_mask, sea_mask = self._extract_target_masks(resp["obs"])
        return (self._obs_to_vec(resp["obs"], idx),
                self._extract_action_mask(resp["obs"]),
                land_mask, sea_mask)

    def reset_all(self):
        """Returns (obs, masks, land_target_masks, sea_target_masks) each shape (num_envs, ...)."""
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        masks = np.ones((self.num_envs, NUM_ACTIONS), dtype=np.float32)
        land_masks = np.ones((self.num_envs, self.max_neighbors), dtype=np.float32)
        sea_masks = np.ones((self.num_envs, self.max_neighbors), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i], masks[i], land_masks[i], sea_masks[i] = self.reset_single(i)
        return obs, masks, land_masks, sea_masks

    def _step_single(self, i: int, action: np.ndarray):
        """Step a single env. Returns (i, resp) or (i, None) on error."""
        self._step_counts[i] += 1
        game_action = self._decode_action(action, i)
        try:
            resp = self._send(i, {
                "cmd": "step",
                "action": game_action,
                "ticksPerStep": self.ticks_per_step,
            })
            return (i, resp)
        except (RuntimeError, BrokenPipeError):
            return (i, None)

    def step(self, actions: np.ndarray):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        masks = np.ones((self.num_envs, NUM_ACTIONS), dtype=np.float32)
        land_masks = np.ones((self.num_envs, self.max_neighbors), dtype=np.float32)
        sea_masks = np.ones((self.num_envs, self.max_neighbors), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        infos = [{}] * self.num_envs

        # Step all envs in parallel (I/O-bound: waiting on subprocess responses)
        futures = [self._pool.submit(self._step_single, i, actions[i])
                   for i in range(self.num_envs)]

        for future in futures:
            i, resp = future.result()

            if resp is None or "obs" not in resp:
                if resp is not None:
                    print(f"[env {i}] bad response (no 'obs' key), resetting: {resp}")
                dones[i] = True
                obs[i], masks[i], land_masks[i], sea_masks[i] = self.reset_single(i)
                continue

            obs[i] = self._obs_to_vec(resp["obs"], i)
            masks[i] = self._extract_action_mask(resp["obs"])
            land_masks[i], sea_masks[i] = self._extract_target_masks(resp["obs"])
            rewards[i] = float(resp.get("reward", 0))
            dones[i] = bool(resp.get("done", False))
            truncateds[i] = self._step_counts[i] >= self.max_steps and not dones[i]
            infos[i] = resp.get("info", {})

        return obs, masks, land_masks, sea_masks, rewards, dones, truncateds, infos

    def close(self):
        self._pool.shutdown(wait=False)
        for i, proc in enumerate(self._procs):
            if proc is not None:
                try:
                    self._send(i, {"cmd": "close"})
                except Exception:
                    pass
                proc.kill()
                self._procs[i] = None
