"""
Vectorized OpenFront Environment (Multi-Agent Self-Play)

Runs N game server processes in parallel. Each process hosts K RL agents
(shared-policy self-play) + M Nations (strong AI) + B Bots (tribes). Slots
are flattened to a total of `num_envs * K` env slots for PPO batching;
all K slots of a game share a single server process and a single game-done /
truncation signal (whole-game resets, not per-slot).
"""

import json
import subprocess
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from env import (
    NUM_ACTIONS, BUILD_TYPES, NUKE_TYPES,
    ACTION_NOOP, ACTION_ATTACK, ACTION_BOAT_ATTACK, ACTION_RETREAT,
    ACTION_MOVE_WARSHIP, ACTION_UPGRADE, ACTION_DELETE_UNIT,
)


class VecOpenFrontEnv:
    """Vectorized multi-agent self-play environment.

    Layout:
        num_envs game processes × K agents each = num_envs * K total slots.
        Slot at flat index `e * K + k` is agent k inside game process e.

    Game-level invariants:
        - All K slots of a game share the same `dones` flag (emitted at game
          end: engine winner, all RL dead, or Python-side max_steps).
        - Resets are per-game (reset_env(e)), not per-slot.
        - A dead agent produces zombie frames (zero obs, NOOP-only mask,
          reward=0) until its game resolves.
    """

    def __init__(
        self,
        num_envs: int = 4,
        num_agents_per_env: int = 4,
        maps: list[str] = None,
        num_nations: int = 0,
        num_bots: int = 8,
        difficulty: str = "Easy",
        ticks_per_step: int = 10,
        max_steps: int = 10000,
        max_neighbors: int = 16,
        potential_alpha: float = 0.0,
    ):
        self.num_envs = num_envs
        self.K = num_agents_per_env
        self.num_slots = num_envs * num_agents_per_env
        self.maps = maps or ["plains"]
        self.num_nations = num_nations
        self.num_bots = num_bots
        self.difficulty = difficulty
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.max_neighbors = max_neighbors
        self.potential_alpha = potential_alpha

        obs_size = 16 + max_neighbors * 5
        self.obs_dim = obs_size

        self._procs: list[Optional[subprocess.Popen]] = [None] * num_envs
        # Per-slot neighbor caches for action decoding (each slot has its own
        # ordering of neighbors since observations are per-agent).
        self._neighbors_caches: list[list] = [[] for _ in range(self.num_slots)]
        self._step_counts = np.zeros(num_envs, dtype=np.int32)

        rl_dir = Path(__file__).parent
        self._repo_dir = str(rl_dir.parent)
        self._server_script = str(rl_dir / "env_server.ts")
        # Use tsx binary directly to avoid npx resolution overhead
        self._tsx_bin = str(rl_dir.parent / "node_modules" / ".bin" / "tsx")

        self._pool = ThreadPoolExecutor(max_workers=num_envs)
        self._start_all()

    # ---------- Server lifecycle ----------

    def _start_server(self, env_idx: int):
        if self._procs[env_idx] is not None:
            try:
                self._procs[env_idx].kill()
            except Exception:
                pass

        t0 = time.time()
        proc = subprocess.Popen(
            [self._tsx_bin, self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._repo_dir,
            text=True,
            bufsize=1,
        )
        line = proc.stdout.readline()
        msg = json.loads(line)
        assert msg.get("status") == "ready", f"Server {env_idx} not ready: {msg}"
        self._procs[env_idx] = proc
        print(f"[init] server {env_idx} ready in {time.time()-t0:.1f}s", flush=True)

    def _start_all(self):
        t0 = time.time()
        list(self._pool.map(self._start_server, range(self.num_envs)))
        print(f"[init] all {self.num_envs} servers ready in {time.time()-t0:.1f}s", flush=True)

    def _send(self, env_idx: int, msg: dict) -> dict:
        proc = self._procs[env_idx]
        assert proc is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Server {env_idx} closed connection")
        return json.loads(line)

    # ---------- Observation / mask conversion ----------

    def _obs_to_vec(self, obs: dict, slot_idx: int) -> np.ndarray:
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

        build_result = obs.get("lastBuildResult", "none")
        if build_result == "success":
            vec[14] = 1.0
        elif build_result == "none":
            vec[14] = 0.0
        else:
            vec[14] = -1.0
        vec[15] = float(obs.get("lastActionSucceeded", False))

        self._neighbors_caches[slot_idx] = neighbors
        for i, n in enumerate(neighbors[: self.max_neighbors]):
            base = 16 + i * 5
            vec[base] = n.get("tiles", 0) / total
            vec[base + 1] = n.get("troops", 0) / 100000
            vec[base + 2] = n.get("relation", 0) / 3
            vec[base + 3] = float(n.get("isLandNeighbor", True))
            vec[base + 4] = n.get("distance", 1.0)
        return vec

    def _extract_action_mask(self, obs: dict) -> np.ndarray:
        mask = obs.get("actionMask", [True] * NUM_ACTIONS)
        arr = np.array(mask[:NUM_ACTIONS], dtype=np.float32)
        if len(arr) < NUM_ACTIONS:
            arr = np.concatenate([arr, np.ones(NUM_ACTIONS - len(arr), dtype=np.float32)])
        arr[0] = 1.0  # NOOP always valid
        return arr

    def _extract_target_masks(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        land = obs.get("landTargetMask", [1] * self.max_neighbors)
        sea = obs.get("seaTargetMask", [1] * self.max_neighbors)
        land_arr = np.zeros(self.max_neighbors, dtype=np.float32)
        sea_arr = np.zeros(self.max_neighbors, dtype=np.float32)
        for i in range(min(len(land), self.max_neighbors)):
            land_arr[i] = float(land[i])
        for i in range(min(len(sea), self.max_neighbors)):
            sea_arr[i] = float(sea[i])
        return land_arr, sea_arr

    def _decode_action(self, action: np.ndarray, slot_idx: int) -> dict:
        action_type = int(action[0])
        target_idx = int(action[1])
        troop_bucket = int(action[2])
        troop_fraction = (troop_bucket + 1) * 0.2
        neighbors = self._neighbors_caches[slot_idx]

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

    # ---------- Reset ----------

    def _reset_config(self, map_name: str) -> dict:
        return {
            "map": map_name,
            "numAgents": self.K,
            "numNations": self.num_nations,
            "numBots": self.num_bots,
            "difficulty": self.difficulty,
            "potentialAlpha": self.potential_alpha,
        }

    def reset_env(self, env_idx: int, _time_it: bool = False):
        """Reset an entire game (all K agent slots).

        Returns a tuple of four (K, ...) arrays: obs, action_mask, land_target_mask,
        sea_target_mask. Callers are responsible for placing them at slot range
        `[env_idx*K : (env_idx+1)*K]` in flat buffers.
        """
        self._step_counts[env_idx] = 0
        map_name = random.choice(self.maps)
        t0 = time.time()
        try:
            resp = self._send(env_idx, {"cmd": "reset", "config": self._reset_config(map_name)})
        except (RuntimeError, BrokenPipeError):
            self._start_server(env_idx)
            resp = self._send(env_idx, {"cmd": "reset", "config": self._reset_config(map_name)})
        if _time_it:
            print(f"[init] env {env_idx} reset ({map_name}) in {time.time()-t0:.1f}s", flush=True)

        obs_arr = resp["obs"]  # list of K obs dicts
        obs_k = np.zeros((self.K, self.obs_dim), dtype=np.float32)
        mask_k = np.ones((self.K, NUM_ACTIONS), dtype=np.float32)
        land_k = np.ones((self.K, self.max_neighbors), dtype=np.float32)
        sea_k = np.ones((self.K, self.max_neighbors), dtype=np.float32)
        for k in range(self.K):
            o = obs_arr[k]
            flat = env_idx * self.K + k
            obs_k[k] = self._obs_to_vec(o, flat)
            mask_k[k] = self._extract_action_mask(o)
            land_k[k], sea_k[k] = self._extract_target_masks(o)
        return obs_k, mask_k, land_k, sea_k

    def reset_all(self, _time_it: bool = False):
        """Reset all envs in parallel. Returns (num_slots, ...) arrays."""
        obs = np.zeros((self.num_slots, self.obs_dim), dtype=np.float32)
        masks = np.ones((self.num_slots, NUM_ACTIONS), dtype=np.float32)
        land_masks = np.ones((self.num_slots, self.max_neighbors), dtype=np.float32)
        sea_masks = np.ones((self.num_slots, self.max_neighbors), dtype=np.float32)
        t0 = time.time()
        futures = {e: self._pool.submit(self.reset_env, e, _time_it) for e in range(self.num_envs)}
        for e, fut in futures.items():
            o_k, m_k, l_k, s_k = fut.result()
            start = e * self.K
            end = start + self.K
            obs[start:end] = o_k
            masks[start:end] = m_k
            land_masks[start:end] = l_k
            sea_masks[start:end] = s_k
        if _time_it:
            print(f"[init] all {self.num_envs} envs reset in {time.time()-t0:.1f}s", flush=True)
        return obs, masks, land_masks, sea_masks

    # ---------- Step ----------

    def _step_env(self, env_idx: int, actions_K: list[dict]):
        """Step a single game with K per-agent actions."""
        self._step_counts[env_idx] += 1
        try:
            resp = self._send(env_idx, {
                "cmd": "step",
                "actions": actions_K,
                "ticksPerStep": self.ticks_per_step,
            })
            return (env_idx, resp)
        except (RuntimeError, BrokenPipeError):
            return (env_idx, None)

    def step(self, actions: np.ndarray):
        """Step all games in parallel.

        Args:
            actions: (num_slots, 3) int array.

        Returns:
            obs: (num_slots, obs_dim)
            masks: (num_slots, NUM_ACTIONS)
            land_masks: (num_slots, max_neighbors)
            sea_masks: (num_slots, max_neighbors)
            rewards: (num_slots,)
            dones: (num_slots,) bool — True for ALL K slots of a game on game
                end (engine winner or all RL dead).
            truncateds: (num_slots,) bool — True for ALL K slots of a game
                when Python-side max_steps is hit before natural termination.
            infos: list of per-slot dicts. For slots in envs whose game
                finished or was truncated this step, info includes
                `anyAgentBeatAI` (game-level milestone flag).
        """
        obs = np.zeros((self.num_slots, self.obs_dim), dtype=np.float32)
        masks = np.ones((self.num_slots, NUM_ACTIONS), dtype=np.float32)
        land_masks = np.ones((self.num_slots, self.max_neighbors), dtype=np.float32)
        sea_masks = np.ones((self.num_slots, self.max_neighbors), dtype=np.float32)
        rewards = np.zeros(self.num_slots, dtype=np.float32)
        dones = np.zeros(self.num_slots, dtype=bool)
        truncateds = np.zeros(self.num_slots, dtype=bool)
        infos: list = [{} for _ in range(self.num_slots)]

        # Decode per-env action lists using per-slot neighbor caches
        per_env_actions: list[list[dict]] = []
        for e in range(self.num_envs):
            k_actions = []
            for k in range(self.K):
                flat = e * self.K + k
                k_actions.append(self._decode_action(actions[flat], flat))
            per_env_actions.append(k_actions)

        # Dispatch all games in parallel (I/O bound on subprocess pipes)
        futures = [self._pool.submit(self._step_env, e, per_env_actions[e])
                   for e in range(self.num_envs)]

        for future in futures:
            e, resp = future.result()
            start = e * self.K
            end = start + self.K

            if resp is None or "obs" not in resp:
                if resp is not None:
                    print(f"[env {e}] bad response (no 'obs' key), resetting: {resp}")
                dones[start:end] = True
                o_k, m_k, l_k, s_k = self.reset_env(e)
                obs[start:end] = o_k
                masks[start:end] = m_k
                land_masks[start:end] = l_k
                sea_masks[start:end] = s_k
                continue

            obs_arr = resp["obs"]
            rewards_arr = resp["rewards"]
            dones_arr = resp["dones"]
            game_done = bool(resp.get("gameDone", False))
            game_info = resp.get("gameInfo", {})

            # Python-side time limit: if max_steps exceeded without natural
            # termination, mark all K slots of this env as truncated.
            truncated_this_env = (
                self._step_counts[e] >= self.max_steps and not game_done
            )

            for k in range(self.K):
                flat = start + k
                o = obs_arr[k]
                obs[flat] = self._obs_to_vec(o, flat)
                masks[flat] = self._extract_action_mask(o)
                land_masks[flat], sea_masks[flat] = self._extract_target_masks(o)
                rewards[flat] = float(rewards_arr[k])
                dones[flat] = bool(dones_arr[k])
                truncateds[flat] = truncated_this_env
                infos[flat] = {
                    "env_idx": e,
                    "slot": k,
                    "gameDone": game_done,
                    "anyAgentBeatAI": bool(game_info.get("anyAgentBeatAI", False)),
                    "numAgentsAliveAtEnd": game_info.get("numAgentsAliveAtEnd", 0),
                    "tickCount": game_info.get("tickCount", 0),
                    "winner": game_info.get("winner"),
                }

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
