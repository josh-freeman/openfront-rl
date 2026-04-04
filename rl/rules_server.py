"""
Rules-based policy server — drop-in replacement for `python play.py --mode server`.

Strategy: accumulate troops to half the theoretical troop cap, then attack
the weakest land neighbour with 25% of current troops. Repeat.

Usage:
  python rules_server.py --port 8765
"""

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

ACTION_NOOP = 0
ACTION_ATTACK = 1


def max_troops(my_tiles: int) -> float:
    """Troop cap formula from DefaultConfig.ts (human player, no city bonus)."""
    return 2 * (max(my_tiles, 1) ** 0.6 * 1000 + 50_000)


def rules_agent(obs: dict) -> dict:
    my_tiles = obs.get("myTiles", 0)
    my_troops = obs.get("myTroops", 0)
    action_mask = obs.get("actionMask", [True] + [False] * 16)
    neighbors = obs.get("neighbors", [])

    threshold = max_troops(my_tiles) / 2

    if my_troops >= threshold and len(action_mask) > ACTION_ATTACK and action_mask[ACTION_ATTACK]:
        land_neighbors = [
            (i, n)
            for i, n in enumerate(neighbors)
            if n.get("isLandNeighbor", False) and not n.get("isAllied", False)
        ]
        if land_neighbors:
            target_idx, _ = min(land_neighbors, key=lambda x: x[1].get("troops", 0))
            return {"actionType": ACTION_ATTACK, "targetIdx": target_idx, "troopFraction": 0.25}

    return {"actionType": ACTION_NOOP, "targetIdx": 0, "troopFraction": 0.25}


class RulesHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        obs = json.loads(self.rfile.read(length))
        action = rules_agent(obs)
        body = json.dumps(action).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        my_tiles = getattr(self, "_last_tiles", 0)
        pass  # suppress per-request noise; summary printed in do_POST if desired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Rules server running on http://0.0.0.0:{args.port}")
    print("Strategy: wait for troops >= maxTroops/2, then attack weakest land neighbour at 25%")
    HTTPServer(("0.0.0.0", args.port), RulesHandler).serve_forever()


if __name__ == "__main__":
    main()
