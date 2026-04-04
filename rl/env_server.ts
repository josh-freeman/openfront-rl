/**
 * OpenFront RL Environment Server
 *
 * Runs the headless OpenFront game and communicates with a Python
 * training script via JSON lines over stdin/stdout.
 *
 * Protocol:
 *   Python sends: { "cmd": "reset", "config": {...} }
 *   Server sends: { "obs": {...}, "reward": 0, "done": false, "info": {...} }
 *
 *   Python sends: { "cmd": "step", "action": {...} }
 *   Server sends: { "obs": {...}, "reward": 0.5, "done": false, "info": {...} }
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
import { NukeExecution } from "../src/core/execution/NukeExecution";
import { RetreatExecution } from "../src/core/execution/RetreatExecution";
import { SpawnExecution } from "../src/core/execution/SpawnExecution";
import { TransportShipExecution } from "../src/core/execution/TransportShipExecution";
import { UpgradeStructureExecution } from "../src/core/execution/UpgradeStructureExecution";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../src/core/game/Game";
import { createGame } from "../src/core/game/GameImpl";
import { TileRef } from "../src/core/game/GameMap";
import {
  genTerrainFromBin,
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

// ---------- Game State ----------

let game: Game | null = null;
let rlPlayer: Player | null = null;
let tickCount = 0;
let landTiles = 0; // cached at reset for reward normalization
let prevAliveCount = 0;
let numOpponentsAtStart = 0;
// Wilderness CC cache: maps synthetic IDs to our border tiles adjacent to that CC
let wildernessBorderCache: Map<string, TileRef[]> = new Map();
const GAME_ID = "rl-training";

// ---------- Helper: Load map ----------

async function loadMap(mapName: string) {
  // Try testdata maps first, then resources/maps
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
  return { gameMap, miniGameMap };
}

// ---------- Observation extraction ----------

function getObservation() {
  if (!game || !rlPlayer) {
    return { error: "no game" };
  }

  const width = game.width();
  const height = game.height();

  // Our player state
  const myTiles = rlPlayer.numTilesOwned();
  const myTroops = rlPlayer.troops();
  const myGold = Number(rlPlayer.gold());
  const alive = rlPlayer.isAlive();

  // All alive players (not just land neighbors) — sorted by relevance
  // Land neighbors first (immediate threats), then others sorted by size
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
  }> = game
    .players()
    .filter((p) => p.id() !== rlPlayer!.id() && p.isAlive())
    .map((p) => ({
      id: p.id(),
      name: p.name(),
      tiles: p.numTilesOwned(),
      troops: p.troops(),
      alive: p.isAlive(),
      relation: rlPlayer!.relation(p),
      isLandNeighbor: landNeighborIds.has(p.id()),
    }));

  // Wilderness connected components: unclaimed land regions bordering us
  wildernessBorderCache.clear();
  const visited = new Set<TileRef>();
  const borderSet = rlPlayer.borderTiles();
  let ccIndex = 0;
  const CC_CAP = 5000;

  for (const bt of borderSet) {
    for (const adj of game.neighbors(bt)) {
      if (!game.isLand(adj) || game.hasOwner(adj) || visited.has(adj)) continue;

      // BFS to find this connected component of unclaimed land
      const ccTiles = new Set<TileRef>();
      const queue: TileRef[] = [adj];
      ccTiles.add(adj);
      visited.add(adj);

      while (queue.length > 0 && ccTiles.size < CC_CAP) {
        const curr = queue.pop()!;
        for (const n of game.neighbors(curr)) {
          if (!visited.has(n) && game.isLand(n) && !game.hasOwner(n)) {
            visited.add(n);
            ccTiles.add(n);
            queue.push(n);
          }
        }
      }

      const ccId = `wilderness_${ccIndex++}`;

      // Find our border tiles adjacent to this CC
      const ourBorderTiles: TileRef[] = [];
      for (const bt2 of borderSet) {
        for (const n of game.neighbors(bt2)) {
          if (ccTiles.has(n)) {
            ourBorderTiles.push(bt2);
            break;
          }
        }
      }

      wildernessBorderCache.set(ccId, ourBorderTiles);

      neighbors.push({
        id: ccId,
        name: "Wilderness",
        tiles: ccTiles.size,
        troops: 0,
        alive: true,
        relation: 0,
        isLandNeighbor: true,
      });
    }
  }

  // Sort: land neighbors first, then by territory size
  neighbors.sort((a, b) => {
    if (a.isLandNeighbor !== b.isLandNeighbor) return a.isLandNeighbor ? -1 : 1;
    return b.tiles - a.tiles;
  });

  // Our units
  const units = rlPlayer.units().map((u) => ({
    type: u.type(),
    tile: u.tile(),
    x: game!.x(u.tile()),
    y: game!.y(u.tile()),
  }));

  // Border tiles (for action targeting)
  const borderTilesArr: { x: number; y: number; ref: number }[] = [];
  let count = 0;
  for (const t of borderSet) {
    if (count++ > 200) break; // Cap for performance
    borderTilesArr.push({ x: game.x(t), y: game.y(t), ref: t });
  }

  // Incoming/outgoing attacks
  const incoming = rlPlayer.incomingAttacks().length;
  const outgoing = rlPlayer.outgoingAttacks().length;

  // All players summary
  const allPlayers = game.players().map((p) => ({
    id: p.id(),
    name: p.name(),
    tiles: p.numTilesOwned(),
    troops: p.troops(),
    alive: p.isAlive(),
    isUs: p.id() === rlPlayer!.id(),
  }));

  // Unit type counts
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

  // Affordability flags — tell the model what it can actually build right now
  // Uses canBuild on a random border tile as a proxy (checks gold + unit constraints)
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

  // Nuke affordability (need silo + gold)
  const canAffordAtom = hasSilo && myGold >= 750_000;
  const canAffordHBomb = hasSilo && myGold >= 5_000_000;
  const canAffordMIRV = hasSilo && myGold >= 10_000_000;

  const hasNeighbors = neighbors.length > 0;
  const hasTroops = myTroops > 10;
  const hasOutgoingAttacks = outgoing > 0;
  const hasUnits = units.length > 0;
  const canDelete = hasUnits && rlPlayer.canDeleteUnit();

  // Per-target masks: which neighbors are valid targets for land vs sea actions
  const landTargetMask = new Array(16).fill(0);
  const seaTargetMask = new Array(16).fill(0);
  neighbors.slice(0, 16).forEach((n, i) => {
    if (n.id.startsWith("wilderness_")) {
      landTargetMask[i] = 1; // wilderness always attackable on land
      // seaTargetMask stays 0 — can't boat-attack wilderness
    } else {
      const notAllied = !rlPlayer!.isAlliedWith(game!.player(n.id));
      if (n.isLandNeighbor && notAllied) landTargetMask[i] = 1;
      if (notAllied) seaTargetMask[i] = 1; // boat/nuke/warship
    }
  });
  const hasAttackableNeighbor = landTargetMask.some((v) => v === 1);
  const hasAttackableBySeaNeighbor = seaTargetMask.some((v) => v === 1);

  // Action mask: 17 booleans, true = action is valid right now
  // Indices match env.py action IDs exactly
  const actionMask = [
    true, // 0: NOOP — always valid
    hasAttackableNeighbor && hasTroops, // 1: ATTACK
    hasAttackableBySeaNeighbor && hasTroops && hasPort, // 2: BOAT_ATTACK
    hasOutgoingAttacks, // 3: RETREAT
    canAffordCity, // 4: BUILD_CITY
    canAffordFactory, // 5: BUILD_FACTORY
    canAffordDefense, // 6: BUILD_DEFENSE
    canAffordPort, // 7: BUILD_PORT
    canAffordSAM, // 8: BUILD_SAM
    canAffordSilo, // 9: BUILD_SILO
    canAffordWarship && hasPort, // 10: BUILD_WARSHIP
    canAffordAtom && hasAttackableBySeaNeighbor, // 11: LAUNCH_ATOM
    canAffordHBomb && hasAttackableBySeaNeighbor, // 12: LAUNCH_HBOMB
    canAffordMIRV && hasAttackableBySeaNeighbor, // 13: LAUNCH_MIRV
    numWarships > 0 && hasAttackableBySeaNeighbor, // 14: MOVE_WARSHIP
    hasUnits && myGold >= 50_000, // 15: UPGRADE (rough check)
    canDelete, // 16: DELETE_UNIT (respects 300-tick cooldown)
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
    lastBuildResult,
    lastActionSucceeded,
    canAffordCity,
    canAffordDefense,
    canAffordFactory,
    canAffordPort,
    canAffordSilo,
    canAffordSAM,
    canAffordWarship,
    actionMask,
    landTargetMask,
    seaTargetMask,
  };
}

// ---------- Reward calculation ----------

function calculateReward(): number {
  if (!rlPlayer || !game) return 0;

  let reward = 0;

  const aliveCount = game
    .players()
    .filter((p) => p.id() !== rlPlayer!.id() && p.isAlive()).length;
  const newlyDead = prevAliveCount - aliveCount;
  if (newlyDead > 0) reward += newlyDead / numOpponentsAtStart;
  prevAliveCount = aliveCount;

  // Winner bonus: credit for all remaining alive opponents
  const winner = game.getWinner();
  if (winner && winner.id() === "rl_agent" && aliveCount > 0) {
    reward += aliveCount / numOpponentsAtStart;
  }

  if (!rlPlayer.isAlive()) reward -= 1.0;

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

function findClosestBorderTile(target: Player): TileRef | null {
  if (!game || !rlPlayer) return null;
  const borderArr = Array.from(rlPlayer.borderTiles());
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

// Track build outcomes so the model can learn what works
let lastBuildResult:
  | "success"
  | "invalid_tile"
  | "no_tile"
  | "no_port"
  | "no_gold"
  | "none" = "none";
let lastActionSucceeded = false;

/**
 * Pick the best tile for building a given unit type.
 * - Cities/Factories: far from borders (safe interior) for economy
 * - DefensePost/SAMLauncher: near borders but ~3-5 tiles back (time to build before enemy reaches)
 * - MissileSilo: medium distance from border (protected but in range)
 * - Port: needs to be near water (handled by canBuild)
 */
function pickBuildTile(unitType: UnitType): TileRef | null {
  if (!game || !rlPlayer) return null;

  const borders = Array.from(rlPlayer.borderTiles());
  if (borders.length === 0) return null;

  // Compute distance from each border tile to nearest enemy border
  // We'll use this to rank tiles
  const enemyBorders: TileRef[] = [];
  for (const n of rlPlayer.neighbors()) {
    if (n.isPlayer()) {
      const nb = Array.from((n as Player).borderTiles());
      for (let i = 0; i < Math.min(nb.length, 50); i++) {
        enemyBorders.push(nb[i]);
      }
    }
  }

  // Score each of our border tiles by distance to nearest enemy
  const scored = borders.map((t) => {
    let minEnemyDist = Infinity;
    for (const eb of enemyBorders) {
      const d = game!.manhattanDist(t, eb);
      if (d < minEnemyDist) minEnemyDist = d;
    }
    // If no enemies, use distance from map center as proxy
    if (minEnemyDist === Infinity) {
      const cx = game!.width() / 2;
      const cy = game!.height() / 2;
      const tx = game!.x(t);
      const ty = game!.y(t);
      minEnemyDist = Math.abs(tx - cx) + Math.abs(ty - cy);
    }
    return { tile: t, dist: minEnemyDist };
  });

  // Sort by distance to enemy
  scored.sort((a, b) => a.dist - b.dist);

  switch (unitType) {
    case UnitType.City:
    case UnitType.Factory: {
      // Try top 20% farthest tiles, pick a random one from those // Economy buildings: pick from the FARTHEST tiles from enemies (safe interior)
      const safeTiles = scored.slice(Math.floor(scored.length * 0.8));
      if (safeTiles.length === 0)
        return scored[scored.length - 1]?.tile ?? null;
      return safeTiles[Math.floor(Math.random() * safeTiles.length)].tile;
    }

    case UnitType.DefensePost:
    case UnitType.SAMLauncher: {
      // Pick tiles at 20-40% from the front (some buffer to finish building) // Defensive buildings: near borders but not ON the border
      const lo = Math.floor(scored.length * 0.2);
      const hi = Math.floor(scored.length * 0.4);
      const defTiles = scored.slice(lo, Math.max(hi, lo + 1));
      if (defTiles.length === 0)
        return scored[Math.floor(scored.length * 0.3)]?.tile ?? null;
      return defTiles[Math.floor(Math.random() * defTiles.length)].tile;
    }

    case UnitType.MissileSilo: {
      // Silos: medium distance, ~40-60% back from front
      const lo = Math.floor(scored.length * 0.4);
      const hi = Math.floor(scored.length * 0.6);
      const siloTiles = scored.slice(lo, Math.max(hi, lo + 1));
      if (siloTiles.length === 0)
        return scored[Math.floor(scored.length * 0.5)]?.tile ?? null;
      return siloTiles[Math.floor(Math.random() * siloTiles.length)].tile;
    }

    case UnitType.Port: {
      // Ports need water adjacency — try multiple border tiles, canBuild will validate
      const shuffled = [...scored].sort(() => Math.random() - 0.5);
      for (const s of shuffled.slice(0, 20)) {
        if (rlPlayer.canBuild(unitType, s.tile) !== false) {
          return s.tile;
        }
      }
      return scored[0]?.tile ?? null;
    }

    default:
      // Fallback: random border tile
      return borders[Math.floor(Math.random() * borders.length)];
  }
}

function executeAction(action: RLAction) {
  if (!game || !rlPlayer || !rlPlayer.isAlive()) return;
  lastBuildResult = "none";
  lastActionSucceeded = false;

  switch (action.type) {
    case "attack": {
      if (!action.targetPlayerId) break;

      const fraction = Math.max(0.1, Math.min(1, action.troopFraction ?? 0.5));
      const troops = Math.floor(rlPlayer.troops() * fraction);
      if (troops < 10) break;

      if (action.targetPlayerId.startsWith("wilderness_")) {
        // Attack unclaimed territory
        const borderTiles = wildernessBorderCache.get(action.targetPlayerId);
        if (!borderTiles || borderTiles.length === 0) break;
        const sourceTile =
          borderTiles[Math.floor(Math.random() * borderTiles.length)];
        game.addExecution(
          new AttackExecution(troops, rlPlayer, null, sourceTile, true),
        );
        lastActionSucceeded = true;
      } else {
        if (!game.hasPlayer(action.targetPlayerId)) break;
        const target = game.player(action.targetPlayerId);
        if (!target.isAlive()) break;
        const tile = findClosestBorderTile(target);
        if (!tile) break;
        game.addExecution(
          new AttackExecution(troops, rlPlayer, target.id(), tile, true),
        );
        lastActionSucceeded = true;
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

      // ref must be a tile owned by the target enemy — TransportShipExecution
      // derives the target player and landing destination from this tile
      const targetBorder = Array.from(target.borderTiles());
      if (targetBorder.length === 0) break;
      const dst = targetBorder[Math.floor(targetBorder.length / 2)];

      game.addExecution(new TransportShipExecution(rlPlayer, dst, troops));
      lastActionSucceeded = true;
      break;
    }

    case "retreat": {
      const attacks = rlPlayer.outgoingAttacks();
      if (attacks.length === 0) break;
      // Retreat the most recent attack
      const attack = attacks[attacks.length - 1];
      if (!attack.retreating()) {
        game.addExecution(new RetreatExecution(rlPlayer, attack.id()));
        lastActionSucceeded = true;
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

      // For warships, need a port
      if (ut === UnitType.Warship) {
        const ports = rlPlayer
          .units()
          .filter((u) => u.type() === UnitType.Port);
        if (ports.length === 0) {
          lastBuildResult = "no_port";
          break;
        }
        const canBuild = rlPlayer.canBuild(ut, ports[0].tile());
        if (canBuild !== false) {
          game.addExecution(new ConstructionExecution(rlPlayer, ut, canBuild));
          lastBuildResult = "success";
          lastActionSucceeded = true;
        } else {
          lastBuildResult = "invalid_tile";
        }
        break;
      }

      // Smart tile selection based on unit type
      const tile = pickBuildTile(ut);
      if (!tile) {
        lastBuildResult = "no_tile";
        break;
      }

      const canBuild = rlPlayer.canBuild(ut, tile);
      if (canBuild !== false) {
        game.addExecution(new ConstructionExecution(rlPlayer, ut, canBuild));
        lastBuildResult = "success";
        lastActionSucceeded = true;
      } else {
        lastBuildResult = "invalid_tile";
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

      // Find a missile silo
      const silos = rlPlayer
        .units()
        .filter((u) => u.type() === UnitType.MissileSilo);
      if (silos.length === 0) break;

      // Target center of enemy territory
      const targetTiles = Array.from(target.borderTiles());
      if (targetTiles.length === 0) break;
      const dst = targetTiles[Math.floor(targetTiles.length / 2)];

      game.addExecution(
        new NukeExecution(nukeUt as any, rlPlayer, dst, silos[0].tile()),
      );
      lastActionSucceeded = true;
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

      // Move warship toward target's territory
      const targetTiles = Array.from(target.borderTiles());
      if (targetTiles.length === 0) break;
      const dst = targetTiles[Math.floor(targetTiles.length / 2)];

      game.addExecution(
        new MoveWarshipExecution(rlPlayer, warships[0].id(), dst),
      );
      lastActionSucceeded = true;
      break;
    }

    case "upgrade": {
      // Upgrade the first upgradeable structure
      const units = rlPlayer.units();
      for (const u of units) {
        if (rlPlayer.canUpgradeUnit(u)) {
          game.addExecution(new UpgradeStructureExecution(rlPlayer, u.id()));
          lastActionSucceeded = true;
          break;
        }
      }
      break;
    }

    case "delete_unit": {
      // Delete the least useful unit (e.g. last built)
      const units = rlPlayer.units();
      if (units.length === 0) break;
      if (rlPlayer.canDeleteUnit()) {
        game.addExecution(
          new DeleteUnitExecution(rlPlayer, units[units.length - 1].id()),
        );
        lastActionSucceeded = true;
      }
      break;
    }

    case "noop":
    default:
      lastActionSucceeded = true; // noop always "succeeds"
      break;
  }
}

// ---------- Reset: Create new game ----------

interface ResetConfig {
  map?: string;
  numOpponents?: number;
  difficulty?: string;
  ticksPerStep?: number;
  maxTicks?: number;
}

async function resetGame(config: ResetConfig = {}) {
  const mapName = config.map || "plains";
  const numOpponents = config.numOpponents ?? 3;
  const difficulty = (config.difficulty as Difficulty) || Difficulty.Medium;

  const { gameMap, miniGameMap } = await loadMap(mapName);

  const gameConfig: GameConfig = {
    gameMap: GameMapType.Asia, // doesn't matter for headless
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

  // Create our RL player
  const rlInfo = new PlayerInfo("RLAgent", PlayerType.Human, null, "rl_agent");

  game = createGame([rlInfo], [], gameMap, miniGameMap, cfgObj);

  // Add our spawn
  game.addExecution(new SpawnExecution(GAME_ID, rlInfo));

  // Add opponent bots (Nations)
  for (let i = 0; i < numOpponents; i++) {
    const botInfo = new PlayerInfo(`Bot${i}`, PlayerType.Bot, null, `bot_${i}`);
    game.addExecution(new SpawnExecution(GAME_ID, botInfo));
  }

  // Run spawn phase
  let spawnTicks = 0;
  while (game.inSpawnPhase() && spawnTicks < 500) {
    game.executeNextTick();
    spawnTicks++;
  }

  rlPlayer = game.player("rl_agent");
  tickCount = 0;

  // Cache land tile count for reward normalization
  landTiles = 0;
  const total = game.width() * game.height();
  for (let t = 0; t < total; t++) {
    if (!game.isWater(t as TileRef)) landTiles++;
  }
  if (landTiles === 0) landTiles = total; // fallback

  numOpponentsAtStart = game
    .players()
    .filter((p) => p.id() !== rlPlayer!.id()).length;
  prevAliveCount = numOpponentsAtStart;

  return getObservation();
}

// ---------- Step: advance game N ticks ----------

function stepGame(action: RLAction, ticksPerStep: number = 10) {
  if (!game || !rlPlayer) {
    return { obs: { error: "no game" }, reward: 0, done: true, info: {} };
  }

  // Execute RL agent's action
  executeAction(action);

  // Advance N ticks
  for (let i = 0; i < ticksPerStep; i++) {
    game.executeNextTick();
    tickCount++;

    if (!rlPlayer.isAlive()) break;
    if (game.getWinner()) break;
    // Early exit if all opponents eliminated
    if (
      game
        .players()
        .filter((p) => p.id() !== rlPlayer!.id())
        .every((p) => !p.isAlive())
    )
      break;
  }

  const reward = calculateReward();

  // Check if all opponents are dead (game.getWinner() doesn't fire in FFA)
  const allOpponentsDead = game
    .players()
    .filter((p) => p.id() !== rlPlayer!.id())
    .every((p) => !p.isAlive());
  const done =
    !rlPlayer.isAlive() || game.getWinner() !== null || allOpponentsDead;

  const obs = getObservation();
  const info = {
    tickCount,
    winner: game.getWinner()?.name() ?? null,
    weWon:
      rlPlayer.isAlive() &&
      (game.getWinner()?.id() === "rl_agent" || allOpponentsDead),
  };

  return { obs, reward, done, info };
}

// ---------- Main: JSON-line protocol over stdin/stdout ----------

async function main() {
  // Suppress console.debug/warn to keep stdout clean for protocol
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
        const obs = await resetGame(msg.config ?? {});
        send({ obs, reward: 0, done: false, info: { tickCount: 0 } });
      } else if (msg.cmd === "step") {
        const result = stepGame(
          msg.action ?? { type: "noop" },
          msg.ticksPerStep ?? 10,
        );
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
