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
import { SpawnExecution } from "../src/core/execution/SpawnExecution";
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
let prevTiles = 0;
let prevTroops = 0;
let prevGold = 0n;
let tickCount = 0;
const GAME_ID = "rl-training";

// ---------- Helper: Load map ----------

async function loadMap(mapName: string) {
  const mapsDir = path.join(__rldir, "../tests/testdata/maps");
  const mapDir = path.join(mapsDir, mapName);

  if (!fs.existsSync(mapDir)) {
    throw new Error(
      `Map not found: ${mapDir}. Available: ${fs.readdirSync(mapsDir).join(", ")}`,
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

  // Neighbors (other players bordering us)
  const neighbors = rlPlayer
    .neighbors()
    .filter((n) => n.isPlayer())
    .map((n) => {
      const p = n as Player;
      return {
        id: p.id(),
        name: p.name(),
        tiles: p.numTilesOwned(),
        troops: p.troops(),
        alive: p.isAlive(),
        relation: rlPlayer!.relation(p),
      };
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
  const borderSet = rlPlayer.borderTiles();
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
  };
}

// ---------- Reward calculation ----------

function calculateReward(): number {
  if (!rlPlayer || !game) return 0;

  let reward = 0;

  const curTiles = rlPlayer.numTilesOwned();
  const curTroops = rlPlayer.troops();
  const curGold = Number(rlPlayer.gold());

  // Territory expansion (most important)
  const tileDelta = curTiles - prevTiles;
  reward += tileDelta * 0.01;

  // Troop growth
  const troopDelta = curTroops - prevTroops;
  reward += troopDelta * 0.0001;

  // Death penalty
  if (!rlPlayer.isAlive()) {
    reward -= 10;
  }

  // Win bonus
  if (game.getWinner()?.id() === rlPlayer.id()) {
    reward += 100;
  }

  // Update prev state
  prevTiles = curTiles;
  prevTroops = curTroops;
  prevGold = BigInt(curGold);

  return reward;
}

// ---------- Action execution ----------

interface RLAction {
  type: "attack" | "build" | "noop";
  // For attack: targetPlayerId, troops (fraction 0-1)
  targetPlayerId?: string;
  troopFraction?: number;
  // For build: unitType, tileRef
  unitType?: string;
  tileX?: number;
  tileY?: number;
}

function executeAction(action: RLAction) {
  if (!game || !rlPlayer || !rlPlayer.isAlive()) return;

  switch (action.type) {
    case "attack": {
      if (!action.targetPlayerId) break;
      if (!game.hasPlayer(action.targetPlayerId)) break;

      const target = game.player(action.targetPlayerId);
      if (!target.isAlive()) break;

      const fraction = Math.max(0.1, Math.min(1, action.troopFraction ?? 0.5));
      const troops = Math.floor(rlPlayer.troops() * fraction);
      if (troops < 10) break;

      // Find a border tile closest to target
      const borderArr = Array.from(rlPlayer.borderTiles());
      if (borderArr.length === 0) break;

      // Pick border tile closest to target's territory
      const targetTiles = Array.from(target.borderTiles());
      if (targetTiles.length === 0) break;

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

      game.addExecution(
        new AttackExecution(troops, rlPlayer, target.id(), bestTile, true),
      );
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
      };
      const ut = unitTypeMap[action.unitType];
      if (!ut) break;

      let tile: TileRef;
      if (action.tileX !== undefined && action.tileY !== undefined) {
        tile = game.ref(action.tileX, action.tileY);
      } else {
        // Pick a random border tile
        const borders = Array.from(rlPlayer.borderTiles());
        if (borders.length === 0) break;
        tile = borders[Math.floor(Math.random() * borders.length)];
      }

      // Check if we can build
      const canBuild = rlPlayer.canBuild(ut, tile);
      if (canBuild !== false) {
        game.addExecution(new ConstructionExecution(rlPlayer, ut, canBuild));
      }
      break;
    }

    case "noop":
    default:
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
  prevTiles = rlPlayer.numTilesOwned();
  prevTroops = rlPlayer.troops();
  prevGold = rlPlayer.gold();
  tickCount = 0;

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
  }

  const reward = calculateReward();
  const done =
    !rlPlayer.isAlive() || game.getWinner() !== undefined || tickCount > 30000; // ~50 min max

  const obs = getObservation();
  const info = {
    tickCount,
    winner: game.getWinner()?.name() ?? null,
    weWon: game.getWinner()?.id() === "rl_agent",
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
      send({ error: e.message });
    }
  }
}

main();
