/**
 * OpenFront RL Environment Server (Multi-Agent Self-Play)
 *
 * Runs the headless OpenFront game and communicates with a Python
 * training script via JSON lines over stdin/stdout.
 *
 * Single game process hosts K RL agents (shared policy on the Python
 * side) + N Nations (PlayerType.Nation, strong AI) + B Bots (PlayerType.Bot,
 * tribes). All K RL agents are registered as PlayerType.Human so the engine's
 * WinCheckExecution considers them for winner determination.
 *
 * Protocol:
 *   Python sends: { "cmd": "reset", "config": { map, numAgents, numNations,
 *                                                numBots, difficulty,
 *                                                potentialAlpha } }
 *   Server sends: { "obs": [obs_0, ..., obs_{K-1}],
 *                   "rewards": [r_0, ..., r_{K-1}],
 *                   "dones":   [d_0, ..., d_{K-1}],
 *                   "gameDone": bool,
 *                   "gameInfo": { anyAgentBeatAI, numAgentsAliveAtEnd, ... } }
 *
 *   Python sends: { "cmd": "step", "actions": [a_0, ..., a_{K-1}],
 *                   "ticksPerStep": 10 }
 *   Server sends: same shape as reset response.
 *
 * Reward semantics (per agent):
 *   + 1/numOpponentsAtStart per newly-dead opponent (shared kill credit)
 *   - 1 on death (once, then zombie frames with reward=0)
 *   + 1 when game.getWinner() == this agent
 *   + potentialAlpha * (phi_t - phi_{t-1}) — per-agent territory delta
 *
 * Done semantics:
 *   Per-slot done=True is emitted ONLY at game end (engine winner, all RL
 *   dead, or max_steps). Dead agents produce zombie frames (zero obs, NOOP
 *   mask, reward=0) until game end.
 */

import fs from "fs";
import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";
import { DefaultConfig } from "../src/core/configuration/DefaultConfig";
import { AttackExecution } from "../src/core/execution/AttackExecution";
import { ConstructionExecution } from "../src/core/execution/ConstructionExecution";
import { DeleteUnitExecution } from "../src/core/execution/DeleteUnitExecution";
import { MoveWarshipExecution } from "../src/core/execution/MoveWarshipExecution";
import { NationExecution } from "../src/core/execution/NationExecution";
import { NukeExecution } from "../src/core/execution/NukeExecution";
import { RetreatExecution } from "../src/core/execution/RetreatExecution";
import { SpawnExecution } from "../src/core/execution/SpawnExecution";
import { TransportShipExecution } from "../src/core/execution/TransportShipExecution";
import { UpgradeStructureExecution } from "../src/core/execution/UpgradeStructureExecution";
import {
  Cell,
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Nation,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../src/core/game/Game";
import { createGame } from "../src/core/game/GameImpl";
import { TileRef } from "../src/core/game/GameMap";
import {
  genTerrainFromBin,
  Nation as ManifestNation,
  MapManifest,
} from "../src/core/game/TerrainMapLoader";
import { UserSettings } from "../src/core/game/UserSettings";
import { GameConfig } from "../src/core/Schemas";

const __rldir = path.dirname(fileURLToPath(import.meta.url));

// ---------- Minimal ServerConfig stub (same pattern as TestServerConfig) ----------

class RLServerConfig {
  turnstileSiteKey() {
    return "";
  }
  turnstileSecretKey() {
    return "";
  }
  apiKey() {
    return "";
  }
  allowedFlares() {
    return undefined;
  }
  stripePublishableKey() {
    return "";
  }
  domain() {
    return "";
  }
  subdomain() {
    return "";
  }
  jwtAudience() {
    return "";
  }
  jwtIssuer() {
    return "";
  }
  async jwkPublicKey() {
    return {} as any;
  }
  otelEnabled() {
    return false;
  }
  otelEndpoint() {
    return "";
  }
  otelAuthHeader() {
    return "";
  }
  turnIntervalMs() {
    return 100;
  }
  gameCreationRate() {
    return 0;
  }
  async lobbyMaxPlayers() {
    return 100;
  }
  numWorkers() {
    return 1;
  }
  workerIndex() {
    return 0;
  }
  workerPath() {
    return "";
  }
  workerPort() {
    return 0;
  }
  workerPortByIndex() {
    return 0;
  }
  env() {
    return "dev" as any;
  }
  adminToken() {
    return "";
  }
  adminHeader() {
    return "";
  }
  gitCommit() {
    return "rl";
  }
  getRandomPublicGameModifiers() {
    return {
      isCompact: false,
      isRandomSpawn: false,
      isCrowded: false,
      isHardNations: false,
      isAlliancesDisabled: false,
    };
  }
  async supportsCompactMapForTeams() {
    return false;
  }
}

// ---------- Per-agent state ----------

interface AgentState {
  id: string; // player id (e.g. "rl_agent_0")
  player: Player | null;
  prevAliveOpponents: number;
  prevTerritoryPct: number;
  lastBuildResult:
    | "success"
    | "invalid_tile"
    | "no_tile"
    | "no_port"
    | "no_gold"
    | "none";
  lastActionSucceeded: boolean;
  // Caches for observation generation (per-agent because they depend on
  // borderTiles / centroids of this specific player).
  neighborsCache: any[];
  wildernessBorderCache: Map<string, TileRef[]>;
  cachedWildernessData: {
    neighbors: any[];
    borderCache: Map<string, TileRef[]>;
  } | null;
  wildernessLastTick: number;
  cachedOurCentroids: Array<{ x: number; y: number }>;
  ourCentroidsLastTick: number;
  // Reward bookkeeping
  diedOnTick: number; // -1 if alive, tick number when dead
  alreadyGotDeathPenalty: boolean;
  alreadyGotWinBonus: boolean;
}

// ---------- Game State ----------

let game: Game | null = null;
let agents: AgentState[] = [];
let tickCount = 0;
let numOpponentsAtStart = 0; // total non-self players at game start (same for every agent)
let potentialAlpha = 100;
let anyAgentBeatAI = false;
let nonRLIds: Set<string> = new Set(); // ids of nations + tribes
const WILDERNESS_RECOMPUTE_INTERVAL = 10;
const OUR_CENTROIDS_RECOMPUTE_INTERVAL = 50;
const GAME_ID = "rl-training";

// ---------- Helper: Load map ----------

async function loadMap(mapName: string) {
  const mapsDirs = [
    path.join(__rldir, "../tests/testdata/maps"),
    path.join(__rldir, "../resources/maps"),
  ];
  let mapDir = "";
  for (const dir of mapsDirs) {
    const candidate = path.join(dir, mapName);
    if (fs.existsSync(candidate)) {
      mapDir = candidate;
      break;
    }
  }
  if (!mapDir) {
    const allMaps = mapsDirs.flatMap((d) =>
      fs.existsSync(d) ? fs.readdirSync(d) : [],
    );
    throw new Error(
      `Map not found: ${mapName}. Available: ${allMaps.join(", ")}`,
    );
  }

  const mapBin = fs.readFileSync(path.join(mapDir, "map.bin"));
  const miniMapBin = fs.readFileSync(path.join(mapDir, "map4x.bin"));
  const manifest = JSON.parse(
    fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"),
  ) as MapManifest;

  const gameMap = await genTerrainFromBin(manifest.map, mapBin);
  const miniGameMap = await genTerrainFromBin(manifest.map4x, miniMapBin);
  return { gameMap, miniGameMap, manifestNations: manifest.nations ?? [] };
}

// ---------- Zombie observation (for dead agents) ----------

function zombieObservation(): any {
  const width = game?.width() ?? 0;
  const height = game?.height() ?? 0;
  return {
    tick: tickCount,
    alive: false,
    myTiles: 0,
    myTroops: 0,
    myGold: 0,
    mapWidth: width,
    mapHeight: height,
    neighbors: [],
    units: [],
    borderTiles: [],
    incomingAttacks: 0,
    outgoingAttacks: 0,
    allPlayers: [],
    totalMapTiles: width * height,
    territoryPct: 0,
    hasSilo: false,
    hasPort: false,
    hasSAM: false,
    numWarships: 0,
    numNukes: 0,
    lastBuildResult: "none",
    lastActionSucceeded: false,
    // Action mask: only NOOP valid
    actionMask: [
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
    ],
    landTargetMask: new Array(16).fill(0),
    seaTargetMask: new Array(16).fill(0),
  };
}

// ---------- Observation extraction (per agent) ----------

function getObservation(agent: AgentState): any {
  if (!game || !agent.player || !agent.player.isAlive()) {
    return zombieObservation();
  }
  const rlPlayer = agent.player;

  const width = game.width();
  const height = game.height();

  const myTiles = rlPlayer.numTilesOwned();
  const myTroops = rlPlayer.troops();
  const myGold = Number(rlPlayer.gold());
  const alive = rlPlayer.isAlive();

  // All alive non-self players — land neighbors first, then others by size
  const landNeighborIds = new Set(
    rlPlayer
      .neighbors()
      .filter((n) => n.isPlayer())
      .map((n) => (n as Player).id()),
  );
  const neighbors: Array<{
    id: string;
    name: string;
    tiles: number;
    troops: number;
    alive: boolean;
    relation: number;
    isLandNeighbor: boolean;
    distance: number;
  }> = game
    .players()
    .filter((p) => p.id() !== rlPlayer.id() && p.isAlive())
    .map((p) => ({
      id: p.id(),
      name: p.name(),
      tiles: p.numTilesOwned(),
      troops: p.troops(),
      alive: p.isAlive(),
      relation: rlPlayer.relation(p),
      isLandNeighbor: landNeighborIds.has(p.id()),
      distance: 1.0,
    }));

  // ---- Wilderness CCs (per-agent cache) ----
  const currentTick = tickCount;
  if (
    !agent.cachedWildernessData ||
    currentTick - agent.wildernessLastTick >= WILDERNESS_RECOMPUTE_INTERVAL
  ) {
    agent.wildernessLastTick = currentTick;
    const newBorderCache = new Map<string, TileRef[]>();
    const newNeighbors: typeof neighbors = [];

    const borderSet = rlPlayer.borderTiles();
    const fringeTiles = new Set<TileRef>();
    const fringeToOurBorder = new Map<TileRef, TileRef[]>();

    for (const bt of borderSet) {
      for (const adj of game.neighbors(bt)) {
        if (!game.isLand(adj) || game.hasOwner(adj)) continue;
        fringeTiles.add(adj);
        if (!fringeToOurBorder.has(adj)) fringeToOurBorder.set(adj, []);
        fringeToOurBorder.get(adj)!.push(bt);
      }
    }

    const visited = new Set<TileRef>();
    let ccIndex = 0;
    const SIZE_EST_DEPTH = 3;

    for (const seed of fringeTiles) {
      if (visited.has(seed)) continue;
      visited.add(seed);

      const ccFringe: TileRef[] = [seed];
      const queue: TileRef[] = [seed];
      const ourBorderSet = new Set<TileRef>();

      while (queue.length > 0) {
        const curr = queue.pop()!;
        for (const bt of fringeToOurBorder.get(curr) || []) {
          ourBorderSet.add(bt);
        }
        for (const n of game.neighbors(curr)) {
          if (!visited.has(n) && fringeTiles.has(n)) {
            visited.add(n);
            ccFringe.push(n);
            queue.push(n);
          }
        }
      }

      const sizeVisited = new Set<TileRef>(ccFringe);
      let frontier = ccFringe;
      for (let d = 0; d < SIZE_EST_DEPTH; d++) {
        const nextFrontier: TileRef[] = [];
        for (const t of frontier) {
          for (const n of game.neighbors(t)) {
            if (!sizeVisited.has(n) && game.isLand(n) && !game.hasOwner(n)) {
              sizeVisited.add(n);
              nextFrontier.push(n);
            }
          }
        }
        frontier = nextFrontier;
      }

      // Wilderness CC IDs are per-agent-scoped so they don't collide between
      // agents (each agent has its own wildernessBorderCache).
      const ccId = `wilderness_${agent.id}_${ccIndex++}`;
      newBorderCache.set(ccId, Array.from(ourBorderSet));

      newNeighbors.push({
        id: ccId,
        name: "Wilderness",
        tiles: sizeVisited.size,
        troops: 0,
        alive: true,
        relation: 0,
        isLandNeighbor: true,
        distance: 0,
      });
    }

    agent.cachedWildernessData = {
      neighbors: newNeighbors,
      borderCache: newBorderCache,
    };
  }

  agent.wildernessBorderCache = agent.cachedWildernessData.borderCache;
  for (const wn of agent.cachedWildernessData.neighbors) {
    neighbors.push({ ...wn });
  }

  // ---- Our territory CC centroids (per-agent cache) ----
  if (
    currentTick - agent.ourCentroidsLastTick >=
    OUR_CENTROIDS_RECOMPUTE_INTERVAL
  ) {
    agent.ourCentroidsLastTick = currentTick;
    const ourBorder = rlPlayer.borderTiles();
    agent.cachedOurCentroids = [];
    if (ourBorder.size > 0) {
      const borderSet = ourBorder;
      const visited = new Set<TileRef>();
      const myId = game.ownerID(ourBorder.values().next().value!);
      for (const seed of ourBorder) {
        if (visited.has(seed)) continue;
        visited.add(seed);
        let sx = 0;
        let sy = 0;
        let count = 0;
        const queue: TileRef[] = [seed];
        while (queue.length > 0) {
          const curr = queue.pop()!;
          sx += game.x(curr);
          sy += game.y(curr);
          count++;
          for (const adj of game.neighbors(curr)) {
            if (visited.has(adj)) continue;
            if (borderSet.has(adj)) {
              visited.add(adj);
              queue.push(adj);
            } else if (game.ownerID(adj) === myId) {
              for (const adj2 of game.neighbors(adj)) {
                if (!visited.has(adj2) && borderSet.has(adj2)) {
                  visited.add(adj2);
                  queue.push(adj2);
                }
              }
            }
          }
        }
        agent.cachedOurCentroids.push({ x: sx / count, y: sy / count });
      }
    }
  }

  // Distance to each non-wilderness neighbor
  const mapDiag = width + height;
  for (const nb of neighbors) {
    if (nb.id.startsWith("wilderness_")) continue;
    const opponent = game.player(nb.id);
    if (!opponent) continue;
    const theirBorder = opponent.borderTiles();
    if (theirBorder.size === 0 || agent.cachedOurCentroids.length === 0)
      continue;
    let minDist = mapDiag;
    for (const t of theirBorder) {
      const tx = game.x(t);
      const ty = game.y(t);
      for (const c of agent.cachedOurCentroids) {
        const d = Math.abs(tx - c.x) + Math.abs(ty - c.y);
        if (d < minDist) minDist = d;
      }
    }
    nb.distance = minDist / mapDiag;
  }

  neighbors.sort((a, b) => {
    if (a.isLandNeighbor !== b.isLandNeighbor) return a.isLandNeighbor ? -1 : 1;
    return b.tiles - a.tiles;
  });

  agent.neighborsCache = neighbors;

  // Our units
  const units = rlPlayer.units().map((u) => ({
    type: u.type(),
    tile: u.tile(),
    x: game!.x(u.tile()),
    y: game!.y(u.tile()),
  }));

  // Border tiles (for action targeting) + coast detection
  const borderTilesArr: { x: number; y: number; ref: number }[] = [];
  let count = 0;
  let hasCoast = false;
  const borderSet = rlPlayer.borderTiles();
  for (const t of borderSet) {
    if (count++ > 200) break;
    borderTilesArr.push({ x: game.x(t), y: game.y(t), ref: t });
    if (!hasCoast) {
      for (const adj of game.neighbors(t)) {
        if (!game.isLand(adj)) {
          hasCoast = true;
          break;
        }
      }
    }
  }

  const incoming = rlPlayer.incomingAttacks().length;
  const outgoing = rlPlayer.outgoingAttacks().length;

  const allPlayers = game.players().map((p) => ({
    id: p.id(),
    name: p.name(),
    tiles: p.numTilesOwned(),
    troops: p.troops(),
    alive: p.isAlive(),
    isUs: p.id() === rlPlayer.id(),
  }));

  const hasSilo = units.some((u) => u.type === UnitType.MissileSilo);
  const hasPort = units.some((u) => u.type === UnitType.Port);
  const hasSAM = units.some((u) => u.type === UnitType.SAMLauncher);
  const numWarships = units.filter((u) => u.type === UnitType.Warship).length;
  const numNukes = units.filter(
    (u) =>
      u.type === UnitType.AtomBomb ||
      u.type === UnitType.HydrogenBomb ||
      u.type === UnitType.MIRV,
  ).length;

  const sampleTile =
    borderTilesArr.length > 0
      ? borderTilesArr[Math.floor(Math.random() * borderTilesArr.length)].ref
      : null;
  const canAffordCity =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.City, sampleTile) !== false;
  const canAffordDefense =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.DefensePost, sampleTile) !== false;
  const canAffordFactory =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.Factory, sampleTile) !== false;
  const canAffordPort =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.Port, sampleTile) !== false;
  const canAffordSilo =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.MissileSilo, sampleTile) !== false;
  const canAffordSAM =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.SAMLauncher, sampleTile) !== false;
  const canAffordWarship =
    sampleTile !== null &&
    rlPlayer.canBuild(UnitType.Warship, sampleTile) !== false;

  const canAffordAtom = hasSilo && myGold >= 750_000;
  const canAffordHBomb = hasSilo && myGold >= 5_000_000;
  const canAffordMIRV = hasSilo && myGold >= 10_000_000;

  const hasTroops = myTroops > 10;
  const hasOutgoingAttacks = outgoing > 0;
  const hasUnits = units.length > 0;
  const canDelete = hasUnits && rlPlayer.canDeleteUnit();

  const landTargetMask = new Array(16).fill(0);
  const seaTargetMask = new Array(16).fill(0);
  const canReachBySea = hasCoast || hasPort;
  neighbors.slice(0, 16).forEach((n, i) => {
    if (n.id.startsWith("wilderness_")) {
      landTargetMask[i] = 1;
    } else {
      const notAllied = !rlPlayer.isAlliedWith(game!.player(n.id));
      if (n.isLandNeighbor && notAllied) landTargetMask[i] = 1;
      if (notAllied && canReachBySea) seaTargetMask[i] = 1;
    }
  });
  const hasAttackableNeighbor = landTargetMask.some((v) => v === 1);
  const hasAttackableBySeaNeighbor = seaTargetMask.some((v) => v === 1);

  const actionMask = [
    true, // 0: NOOP
    hasAttackableNeighbor && hasTroops, // 1: ATTACK
    hasAttackableBySeaNeighbor && hasTroops, // 2: BOAT_ATTACK
    hasOutgoingAttacks, // 3: RETREAT
    canAffordCity, // 4: BUILD_CITY
    canAffordFactory, // 5: BUILD_FACTORY
    canAffordDefense, // 6: BUILD_DEFENSE
    canAffordPort && hasCoast, // 7: BUILD_PORT
    canAffordSAM, // 8: BUILD_SAM
    canAffordSilo, // 9: BUILD_SILO
    canAffordWarship && hasPort, // 10: BUILD_WARSHIP
    canAffordAtom && hasAttackableBySeaNeighbor, // 11: LAUNCH_ATOM
    canAffordHBomb && hasAttackableBySeaNeighbor, // 12: LAUNCH_HBOMB
    canAffordMIRV && hasAttackableBySeaNeighbor, // 13: LAUNCH_MIRV
    numWarships > 0 && hasAttackableBySeaNeighbor, // 14: MOVE_WARSHIP
    hasUnits && myGold >= 50_000, // 15: UPGRADE
    canDelete, // 16: DELETE_UNIT
  ];

  return {
    tick: tickCount,
    alive,
    myTiles,
    myTroops,
    myGold,
    mapWidth: width,
    mapHeight: height,
    neighbors,
    units,
    borderTiles: borderTilesArr,
    incomingAttacks: incoming,
    outgoingAttacks: outgoing,
    allPlayers,
    totalMapTiles: width * height,
    territoryPct: myTiles / (width * height),
    hasSilo,
    hasPort,
    hasSAM,
    numWarships,
    numNukes,
    lastBuildResult: agent.lastBuildResult,
    lastActionSucceeded: agent.lastActionSucceeded,
    actionMask,
    landTargetMask,
    seaTargetMask,
  };
}

// ---------- Reward calculation (per agent) ----------

function calculateReward(agent: AgentState): number {
  if (!game || !agent.player) return 0;

  // Zombie frames after death: emit no reward signal
  if (agent.alreadyGotDeathPenalty) return 0;

  let reward = 0;

  // Kill credit: any opponent (agent or non-RL) that died since last step
  const aliveOpponents = game
    .players()
    .filter((p) => p.id() !== agent.id && p.isAlive()).length;
  const newlyDead = agent.prevAliveOpponents - aliveOpponents;
  if (newlyDead > 0 && numOpponentsAtStart > 0) {
    reward += newlyDead / numOpponentsAtStart;
  }
  agent.prevAliveOpponents = aliveOpponents;

  // Engine-declared winner bonus
  const winner = game.getWinner();
  if (
    winner &&
    typeof (winner as any).id === "function" &&
    (winner as any).id() === agent.id &&
    !agent.alreadyGotWinBonus
  ) {
    reward += 1;
    agent.alreadyGotWinBonus = true;
  }

  // Death penalty (delivered once on death tick)
  if (!agent.player.isAlive()) {
    reward -= 1;
    agent.alreadyGotDeathPenalty = true;
    agent.diedOnTick = tickCount;
    // Freeze territory pct so future potential shaping is zero
    agent.prevTerritoryPct = 0;
    return reward;
  }

  // Potential-based reward shaping (alive only)
  const totalTiles = game.width() * game.height();
  const currentPct = agent.player.numTilesOwned() / totalTiles;
  reward += potentialAlpha * (currentPct - agent.prevTerritoryPct);
  agent.prevTerritoryPct = currentPct;

  return reward;
}

// ---------- Action execution ----------

interface RLAction {
  type:
    | "attack"
    | "boat_attack"
    | "retreat"
    | "build"
    | "launch_nuke"
    | "move_warship"
    | "upgrade"
    | "delete_unit"
    | "noop";
  targetPlayerId?: string;
  troopFraction?: number;
  unitType?: string;
  nukeType?: string;
}

function findClosestBorderTile(
  agent: AgentState,
  target: Player,
): TileRef | null {
  if (!game || !agent.player) return null;
  const borderArr = Array.from(agent.player.borderTiles());
  if (borderArr.length === 0) return null;
  const targetTiles = Array.from(target.borderTiles());
  if (targetTiles.length === 0) return null;

  const targetCenter = targetTiles[Math.floor(targetTiles.length / 2)];
  let bestTile = borderArr[0];
  let bestDist = Infinity;
  for (const bt of borderArr) {
    const d = game.manhattanDist(bt, targetCenter);
    if (d < bestDist) {
      bestDist = d;
      bestTile = bt;
    }
  }
  return bestTile;
}

/**
 * Pick the best tile for building a given unit type, relative to this agent's
 * territory.
 */
function pickBuildTile(agent: AgentState, unitType: UnitType): TileRef | null {
  if (!game || !agent.player) return null;
  const rlPlayer = agent.player;

  const borders = Array.from(rlPlayer.borderTiles());
  if (borders.length === 0) return null;

  const enemyBorders: TileRef[] = [];
  for (const n of rlPlayer.neighbors()) {
    if (n.isPlayer()) {
      const nb = Array.from((n as Player).borderTiles());
      for (let i = 0; i < Math.min(nb.length, 50); i++) {
        enemyBorders.push(nb[i]);
      }
    }
  }

  const scored = borders.map((t) => {
    let minEnemyDist = Infinity;
    for (const eb of enemyBorders) {
      const d = game!.manhattanDist(t, eb);
      if (d < minEnemyDist) minEnemyDist = d;
    }
    if (minEnemyDist === Infinity) {
      const cx = game!.width() / 2;
      const cy = game!.height() / 2;
      const tx = game!.x(t);
      const ty = game!.y(t);
      minEnemyDist = Math.abs(tx - cx) + Math.abs(ty - cy);
    }
    return { tile: t, dist: minEnemyDist };
  });

  scored.sort((a, b) => a.dist - b.dist);

  switch (unitType) {
    case UnitType.City:
    case UnitType.Factory: {
      const safeTiles = scored.slice(Math.floor(scored.length * 0.8));
      if (safeTiles.length === 0)
        return scored[scored.length - 1]?.tile ?? null;
      return safeTiles[Math.floor(Math.random() * safeTiles.length)].tile;
    }

    case UnitType.DefensePost:
    case UnitType.SAMLauncher: {
      const lo = Math.floor(scored.length * 0.2);
      const hi = Math.floor(scored.length * 0.4);
      const defTiles = scored.slice(lo, Math.max(hi, lo + 1));
      if (defTiles.length === 0)
        return scored[Math.floor(scored.length * 0.3)]?.tile ?? null;
      return defTiles[Math.floor(Math.random() * defTiles.length)].tile;
    }

    case UnitType.MissileSilo: {
      const lo = Math.floor(scored.length * 0.4);
      const hi = Math.floor(scored.length * 0.6);
      const siloTiles = scored.slice(lo, Math.max(hi, lo + 1));
      if (siloTiles.length === 0)
        return scored[Math.floor(scored.length * 0.5)]?.tile ?? null;
      return siloTiles[Math.floor(Math.random() * siloTiles.length)].tile;
    }

    case UnitType.Port: {
      const shuffled = [...scored].sort(() => Math.random() - 0.5);
      for (const s of shuffled.slice(0, 20)) {
        if (rlPlayer.canBuild(unitType, s.tile) !== false) {
          return s.tile;
        }
      }
      return scored[0]?.tile ?? null;
    }

    default:
      return borders[Math.floor(Math.random() * borders.length)];
  }
}

function executeAction(agent: AgentState, action: RLAction) {
  if (!game || !agent.player || !agent.player.isAlive()) return;
  const rlPlayer = agent.player;
  agent.lastBuildResult = "none";
  agent.lastActionSucceeded = false;

  switch (action.type) {
    case "attack": {
      if (!action.targetPlayerId) break;

      const fraction = Math.max(0.1, Math.min(1, action.troopFraction ?? 0.5));
      const troops = Math.floor(rlPlayer.troops() * fraction);
      if (troops < 10) break;

      if (action.targetPlayerId.startsWith("wilderness_")) {
        const borderTiles = agent.wildernessBorderCache.get(
          action.targetPlayerId,
        );
        if (!borderTiles || borderTiles.length === 0) break;
        const sourceTile =
          borderTiles[Math.floor(Math.random() * borderTiles.length)];
        game.addExecution(
          new AttackExecution(troops, rlPlayer, null, sourceTile, true),
        );
        agent.lastActionSucceeded = true;
      } else {
        if (!game.hasPlayer(action.targetPlayerId)) break;
        const target = game.player(action.targetPlayerId);
        if (!target.isAlive()) break;
        const tile = findClosestBorderTile(agent, target);
        if (!tile) break;
        game.addExecution(
          new AttackExecution(troops, rlPlayer, target.id(), tile, true),
        );
        agent.lastActionSucceeded = true;
      }
      break;
    }

    case "boat_attack": {
      if (!action.targetPlayerId || !game.hasPlayer(action.targetPlayerId))
        break;
      const target = game.player(action.targetPlayerId);
      if (!target.isAlive()) break;

      const fraction = Math.max(0.1, Math.min(1, action.troopFraction ?? 0.5));
      const troops = Math.floor(rlPlayer.troops() * fraction);
      if (troops < 10) break;

      const ports = rlPlayer.units().filter((u) => u.type() === UnitType.Port);
      if (ports.length === 0) break;

      const targetBorder = Array.from(target.borderTiles());
      if (targetBorder.length === 0) break;
      const dst = targetBorder[Math.floor(targetBorder.length / 2)];

      game.addExecution(new TransportShipExecution(rlPlayer, dst, troops));
      agent.lastActionSucceeded = true;
      break;
    }

    case "retreat": {
      const attacks = rlPlayer.outgoingAttacks();
      if (attacks.length === 0) break;
      const attack = attacks[attacks.length - 1];
      if (!attack.retreating()) {
        game.addExecution(new RetreatExecution(rlPlayer, attack.id()));
        agent.lastActionSucceeded = true;
      }
      break;
    }

    case "build": {
      if (!action.unitType) break;
      const unitTypeMap: Record<string, UnitType> = {
        city: UnitType.City,
        factory: UnitType.Factory,
        port: UnitType.Port,
        defense_post: UnitType.DefensePost,
        sam_launcher: UnitType.SAMLauncher,
        missile_silo: UnitType.MissileSilo,
        warship: UnitType.Warship,
      };
      const ut = unitTypeMap[action.unitType];
      if (!ut) break;

      if (ut === UnitType.Warship) {
        const ports = rlPlayer
          .units()
          .filter((u) => u.type() === UnitType.Port);
        if (ports.length === 0) {
          agent.lastBuildResult = "no_port";
          break;
        }
        const canBuild = rlPlayer.canBuild(ut, ports[0].tile());
        if (canBuild !== false) {
          game.addExecution(new ConstructionExecution(rlPlayer, ut, canBuild));
          agent.lastBuildResult = "success";
          agent.lastActionSucceeded = true;
        } else {
          agent.lastBuildResult = "invalid_tile";
        }
        break;
      }

      const tile = pickBuildTile(agent, ut);
      if (!tile) {
        agent.lastBuildResult = "no_tile";
        break;
      }

      const canBuild = rlPlayer.canBuild(ut, tile);
      if (canBuild !== false) {
        game.addExecution(new ConstructionExecution(rlPlayer, ut, canBuild));
        agent.lastBuildResult = "success";
        agent.lastActionSucceeded = true;
      } else {
        agent.lastBuildResult = "invalid_tile";
      }
      break;
    }

    case "launch_nuke": {
      if (!action.targetPlayerId || !game.hasPlayer(action.targetPlayerId))
        break;
      if (!action.nukeType) break;
      const target = game.player(action.targetPlayerId);
      if (!target.isAlive()) break;

      const nukeTypeMap: Record<string, UnitType> = {
        atom_bomb: UnitType.AtomBomb,
        hydrogen_bomb: UnitType.HydrogenBomb,
        mirv: UnitType.MIRV,
      };
      const nukeUt = nukeTypeMap[action.nukeType];
      if (!nukeUt) break;

      const silos = rlPlayer
        .units()
        .filter((u) => u.type() === UnitType.MissileSilo);
      if (silos.length === 0) break;

      const targetTiles = Array.from(target.borderTiles());
      if (targetTiles.length === 0) break;
      const dst = targetTiles[Math.floor(targetTiles.length / 2)];

      game.addExecution(
        new NukeExecution(nukeUt as any, rlPlayer, dst, silos[0].tile()),
      );
      agent.lastActionSucceeded = true;
      break;
    }

    case "move_warship": {
      if (!action.targetPlayerId || !game.hasPlayer(action.targetPlayerId))
        break;
      const target = game.player(action.targetPlayerId);

      const warships = rlPlayer
        .units()
        .filter((u) => u.type() === UnitType.Warship);
      if (warships.length === 0) break;

      const targetTiles = Array.from(target.borderTiles());
      if (targetTiles.length === 0) break;
      const dst = targetTiles[Math.floor(targetTiles.length / 2)];

      game.addExecution(
        new MoveWarshipExecution(rlPlayer, warships[0].id(), dst),
      );
      agent.lastActionSucceeded = true;
      break;
    }

    case "upgrade": {
      const units = rlPlayer.units();
      for (const u of units) {
        if (rlPlayer.canUpgradeUnit(u)) {
          game.addExecution(new UpgradeStructureExecution(rlPlayer, u.id()));
          agent.lastActionSucceeded = true;
          break;
        }
      }
      break;
    }

    case "delete_unit": {
      const units = rlPlayer.units();
      if (units.length === 0) break;
      if (rlPlayer.canDeleteUnit()) {
        game.addExecution(
          new DeleteUnitExecution(rlPlayer, units[units.length - 1].id()),
        );
        agent.lastActionSucceeded = true;
      }
      break;
    }

    case "noop":
    default:
      agent.lastActionSucceeded = true;
      break;
  }
}

// ---------- Reset: Create new game ----------

interface ResetConfig {
  map?: string;
  numAgents?: number;
  numNations?: number;
  numBots?: number;
  difficulty?: string;
  potentialAlpha?: number;
}

function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function resetGame(config: ResetConfig = {}) {
  const mapName = config.map || "plains";
  if (config.potentialAlpha !== undefined)
    potentialAlpha = config.potentialAlpha;
  const numAgents = Math.max(1, config.numAgents ?? 1);
  const numNations = Math.max(0, config.numNations ?? 0);
  const numBots = Math.max(0, config.numBots ?? 0);
  const difficulty = (config.difficulty as Difficulty) || Difficulty.Medium;

  const { gameMap, miniGameMap, manifestNations } = await loadMap(mapName);

  const gameConfig: GameConfig = {
    gameMap: GameMapType.Asia, // placeholder — unused for headless
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty,
    nations: "default",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };

  const serverConfig = new RLServerConfig();
  const cfgObj = new DefaultConfig(
    serverConfig as any,
    gameConfig,
    new UserSettings(),
    false,
  );

  // RL agents (PlayerType.Human → eligible for engine's winner check)
  const humans: PlayerInfo[] = [];
  for (let i = 0; i < numAgents; i++) {
    humans.push(
      new PlayerInfo(`RLAgent${i}`, PlayerType.Human, null, `rl_agent_${i}`),
    );
  }

  // Nations: random sample of manifest nations (fallback to synthetic spawn
  // locations if manifest has fewer entries than requested)
  const chosenManifest: ManifestNation[] = shuffleInPlace([
    ...manifestNations,
  ]).slice(0, Math.min(numNations, manifestNations.length));
  const nations: Nation[] = [];
  for (let i = 0; i < chosenManifest.length; i++) {
    const mn = chosenManifest[i];
    nations.push(
      new Nation(
        new Cell(mn.coordinates[0], mn.coordinates[1]),
        new PlayerInfo(mn.name, PlayerType.Nation, null, `nation_${i}`),
      ),
    );
  }
  // If numNations > manifest count, add extra nations without fixed spawn
  // cells (NationExecution will spawn them randomly).
  for (let i = chosenManifest.length; i < numNations; i++) {
    nations.push(
      new Nation(
        undefined,
        new PlayerInfo(`Nation${i}`, PlayerType.Nation, null, `nation_${i}`),
      ),
    );
  }

  game = createGame([...humans], nations, gameMap, miniGameMap, cfgObj);

  // Spawn RL agents
  for (const h of humans) {
    game.addExecution(new SpawnExecution(GAME_ID, h));
  }

  // Register Nation behaviors (each NationExecution will emit its own
  // SpawnExecution during the spawn phase).
  for (const n of nations) {
    game.addExecution(new NationExecution(GAME_ID, n));
  }

  // Spawn tribes (PlayerType.Bot — simple AI)
  for (let i = 0; i < numBots; i++) {
    const tribe = new PlayerInfo(
      `Tribe${i}`,
      PlayerType.Bot,
      null,
      `tribe_${i}`,
    );
    game.addExecution(new SpawnExecution(GAME_ID, tribe));
  }

  // Run spawn phase
  let spawnTicks = 0;
  while (game.inSpawnPhase() && spawnTicks < 500) {
    game.executeNextTick();
    spawnTicks++;
  }

  // Build agent state
  agents = [];
  for (let i = 0; i < numAgents; i++) {
    const id = `rl_agent_${i}`;
    const player = game.hasPlayer(id) ? game.player(id) : null;
    const territoryPct = player
      ? player.numTilesOwned() / (game.width() * game.height())
      : 0;
    agents.push({
      id,
      player,
      prevAliveOpponents: 0, // set below after all agents are built
      prevTerritoryPct: territoryPct,
      lastBuildResult: "none",
      lastActionSucceeded: false,
      neighborsCache: [],
      wildernessBorderCache: new Map(),
      cachedWildernessData: null,
      wildernessLastTick: -1,
      cachedOurCentroids: [],
      ourCentroidsLastTick: -1,
      diedOnTick: -1,
      alreadyGotDeathPenalty: false,
      alreadyGotWinBonus: false,
    });
  }

  // Total non-self players at start (same for every agent in a symmetric game)
  numOpponentsAtStart = Math.max(
    1,
    game.players().filter((p) => !p.id().startsWith("rl_agent_")).length +
      (numAgents - 1),
  );

  // Per-agent initial opponent count
  for (const agent of agents) {
    agent.prevAliveOpponents = game
      .players()
      .filter((p) => p.id() !== agent.id && p.isAlive()).length;
  }

  // Track which player IDs are NOT RL agents (for anyAgentBeatAI milestone)
  nonRLIds = new Set(
    game
      .players()
      .map((p) => p.id())
      .filter((id) => !id.startsWith("rl_agent_")),
  );

  tickCount = 0;
  anyAgentBeatAI = false;

  // Build initial observations and info
  const obs = agents.map((a) => getObservation(a));
  return {
    obs,
    rewards: agents.map(() => 0),
    dones: agents.map(() => false),
    gameDone: false,
    gameInfo: {
      anyAgentBeatAI,
      numAgentsAliveAtEnd: agents.filter((a) => a.player?.isAlive()).length,
      tickCount: 0,
      winner: null,
    },
  };
}

// ---------- Step: advance game N ticks ----------

function stepGame(actions: RLAction[], ticksPerStep: number = 10) {
  if (!game || agents.length === 0) {
    return {
      obs: [{ error: "no game" }],
      rewards: [0],
      dones: [true],
      gameDone: true,
      gameInfo: { anyAgentBeatAI: false, numAgentsAliveAtEnd: 0, tickCount: 0 },
    };
  }

  // Execute each agent's action (dead agents are skipped)
  for (let i = 0; i < agents.length; i++) {
    const agent = agents[i];
    if (
      agent.player &&
      agent.player.isAlive() &&
      !agent.alreadyGotDeathPenalty
    ) {
      executeAction(agent, actions[i] ?? { type: "noop" });
    }
  }

  // Advance ticks
  for (let t = 0; t < ticksPerStep; t++) {
    game.executeNextTick();
    tickCount++;
    if (game.getWinner()) break;
    if (agents.every((a) => !a.player || !a.player.isAlive())) break;
  }

  // Milestone: did agents wipe out all non-RL players this game?
  if (!anyAgentBeatAI) {
    const nonRLAlive = game
      .players()
      .filter((p) => nonRLIds.has(p.id()) && p.isAlive()).length;
    const anyAgentAlive = agents.some((a) => a.player && a.player.isAlive());
    if (nonRLAlive === 0 && anyAgentAlive) {
      anyAgentBeatAI = true;
    }
  }

  // Per-agent rewards and observations
  const rewards: number[] = [];
  const obsArr: any[] = [];
  for (const agent of agents) {
    rewards.push(calculateReward(agent));
    obsArr.push(getObservation(agent));
  }

  // Game-level done.
  // In FFA mode the engine only fires getWinner() on territory-% or timer
  // conditions — NOT simply when one player is last standing.  Mirror the old
  // behaviour: also end the episode as soon as all non-RL players are dead and
  // at most one RL agent is still alive (nobody left to fight).
  // With K>1, when multiple RL agents are alive after all bots/nations die the
  // game continues so agents can learn agent-vs-agent combat.
  const allRLDead = agents.every((a) => !a.player || !a.player.isAlive());
  const winnerExists = game.getWinner() !== null;
  const numRLAlive = agents.filter((a) => a.player?.isAlive()).length;
  const allNonRLDead =
    nonRLIds.size > 0 &&
    game.players().filter((p) => nonRLIds.has(p.id())).length === 0;
  const gameDone =
    allRLDead || winnerExists || (allNonRLDead && numRLAlive <= 1);

  // Per-slot dones are all aligned to game end (zombie frames emit done=false
  // until the game resolves; then all slots emit done=true together).
  const dones = agents.map(() => gameDone);

  const numAgentsAliveAtEnd = agents.filter((a) => a.player?.isAlive()).length;
  const winner = game.getWinner();
  const winnerName =
    winner && typeof (winner as any).name === "function"
      ? (winner as any).name()
      : null;

  return {
    obs: obsArr,
    rewards,
    dones,
    gameDone,
    gameInfo: {
      anyAgentBeatAI,
      numAgentsAliveAtEnd,
      tickCount,
      winner: winnerName,
    },
  };
}

// ---------- Main: JSON-line protocol over stdin/stdout ----------

async function main() {
  console.debug = () => {};
  console.warn = () => {};
  const origLog = console.log;
  console.log = () => {};

  const rl = readline.createInterface({ input: process.stdin });

  function send(obj: any) {
    process.stdout.write(JSON.stringify(obj) + "\n");
  }

  send({ status: "ready" });

  for await (const line of rl) {
    try {
      const msg = JSON.parse(line);

      if (msg.cmd === "reset") {
        const result = await resetGame(msg.config ?? {});
        send(result);
      } else if (msg.cmd === "step") {
        const result = stepGame(msg.actions ?? [], msg.ticksPerStep ?? 10);
        send(result);
      } else if (msg.cmd === "close") {
        send({ status: "closed" });
        process.exit(0);
      } else {
        send({ error: `unknown cmd: ${msg.cmd}` });
      }
    } catch (e: any) {
      process.stderr.write(`[env_server] error: ${e.message}\n${e.stack}\n`);
      send({ error: e.message });
    }
  }
}

main();
