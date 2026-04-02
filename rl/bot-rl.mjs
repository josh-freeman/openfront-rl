/**
 * OpenFront RL Bot — Puppeteer bot driven by a trained RL policy server.
 *
 * Strategy: Use the game's 'c' key to center camera on our nation.
 * After centering, screen center = our territory, edges = borders/enemies.
 * No pixel color detection needed — just spatial reasoning relative to center.
 *
 * Usage:
 *   1. Start the policy server:
 *      cd /Users/joshua/openfront-rl/rl
 *      python play.py --model checkpoints/best_model.pt --mode server --port 8765
 *
 *   2. Run this bot:
 *      node bot-rl.mjs [BotName] [policyServerUrl]
 */

import puppeteer from "puppeteer-extra";
import AdblockerPlugin from "puppeteer-extra-plugin-adblocker";
import StealthPlugin from "puppeteer-extra-plugin-stealth";

puppeteer.use(StealthPlugin());
puppeteer.use(AdblockerPlugin({ blockTrackers: true }));

const BOT_NAME = process.argv[2] || "xXDarkLord42Xx";
const POLICY_URL = process.argv[3] || "http://localhost:8765";

// Must match training: ticksPerStep (10) × turnIntervalMs (100) = 1000ms
// This is defined in env_server.ts and env.py as ticks_per_step=10
const TICKS_PER_STEP = 10;
const TURN_INTERVAL_MS = 100;
const POLICY_INTERVAL_MS = TICKS_PER_STEP * TURN_INTERVAL_MS; // 1000ms

// Track actual game scale (game default is 1.8, range [0.2, 20])
// The game zooms via: scale /= (1 + delta/600), clamped to [0.2, 20]
let currentScale = 1.8;
let currentAttackRatio = 0.2; // Game default, adjusted via T/Y keys
let lastGold = 0;
let lastTerritoryPct = 0;
let lastBuildResult = "none"; // "success", "fail", or "none"
let lastActionSucceeded = false;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

async function snap(page, name) {
  try {
    const path = `/tmp/openfront-${name}.png`;
    await page.screenshot({ path });
    log(`Screenshot → ${path}`);
  } catch {}
}

// ── Policy server query ──────────────────────────────────────────

async function queryPolicy(obs) {
  try {
    const resp = await fetch(POLICY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obs),
    });
    return await resp.json();
  } catch (e) {
    log(`Policy server error: ${e.message}`);
    return null;
  }
}

// ── Safe page access ─────────────────────────────────────────────

async function safePage(browser) {
  const pages = await browser.pages();
  for (const p of pages) {
    try {
      if (p.url().includes("openfront.io")) return p;
    } catch {}
  }
  if (pages.length > 0) return pages[0];
  return browser.newPage();
}

async function safeEval(page, fn, ...args) {
  try {
    return await page.evaluate(fn, ...args);
  } catch (e) {
    if (e.message.includes("detached") || e.message.includes("disposed"))
      return null;
    throw e;
  }
}

// ── Lobby: join a public game ────────────────────────────────────

async function joinGame(page) {
  log("Going to openfront.io...");
  await page.goto("https://openfront.io", {
    waitUntil: "domcontentloaded",
    timeout: 60_000,
  });
  await sleep(5000);

  // Set our bot name so we can identify ourselves on the leaderboard
  await safeEval(
    page,
    (name) => {
      const input = document.querySelector("username-input");
      if (input) {
        input.baseUsername = name;
        // Also set any underlying input field
        const textInput =
          input.querySelector("input[type='text']") ||
          input.querySelector("input");
        if (textInput) {
          textInput.value = name;
          textInput.dispatchEvent(new Event("input", { bubbles: true }));
        }
        console.log("Set bot name to:", name);
      }
    },
    BOT_NAME,
  );
  log(`Set player name to "${BOT_NAME}"`);
  await sleep(3000);
  await snap(page, "01-lobby");

  const cardClicked = await safeEval(page, () => {
    const allEls = [...document.querySelectorAll("*")];
    document.querySelectorAll("*").forEach((el) => {
      if (el.shadowRoot) {
        for (const c of el.shadowRoot.querySelectorAll("*")) allEls.push(c);
      }
    });
    for (const el of allEls) {
      const text = el.childNodes.length <= 3 ? el.textContent?.trim() : "";
      if (text === "FREE FOR ALL") {
        let card = el.parentElement;
        for (let i = 0; i < 8 && card; i++) {
          const r = card.getBoundingClientRect();
          if (r.width > 300 && r.height > 150) {
            return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
          }
          card = card.parentElement;
        }
      }
    }
    return null;
  });

  if (cardClicked) {
    log("Found game card, clicking...");
    await page.mouse.click(cardClicked.x, cardClicked.y);
  } else {
    log("No card found, clicking featured area...");
    const vp = await safeEval(page, () => ({
      w: window.innerWidth,
      h: window.innerHeight,
    }));
    if (vp) await page.mouse.click(vp.w * 0.42, vp.h * 0.48);
  }

  await sleep(3000);
  log("Waiting for game canvas...");
  try {
    await page.waitForSelector("canvas", { timeout: 90_000 });
    log("Canvas detected!");
  } catch {
    log("No canvas found");
    await snap(page, "02-stuck");
  }
  await sleep(3000);
}

// ── Parse HUD for game state ─────────────────────────────────────

function parseHudNumber(s) {
  if (!s) return 0;
  s = s.trim().replace(/,/g, "");
  if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
  if (s.endsWith("K")) return parseFloat(s) * 1000;
  return parseFloat(s) || 0;
}

async function extractGameState(page, botName) {
  return safeEval(
    page,
    (botName) => {
      const state = {
        myTiles: 0,
        myTroops: 0,
        myGold: 0,
        territoryPct: 0,
        totalMapTiles: 2000000, // Normal map: 2000x1000
        incomingAttacks: 0,
        outgoingAttacks: 0,
        units: [],
        neighbors: [],
        tick: 0,
        hasSilo: false,
        hasPort: false,
        hasSAM: false,
        numWarships: 0,
        numNukes: 0,
      };

      // Fix 3: Read actual map dimensions and tick from game object
      const sidebar = document.querySelector("game-right-sidebar");
      if (sidebar && sidebar.game) {
        const g = sidebar.game;
        const w = typeof g.width === "function" ? g.width() : g.width;
        const h = typeof g.height === "function" ? g.height() : g.height;
        if (w && h) state.totalMapTiles = w * h;
        // Read tick directly from game engine (always increasing)
        const ticks = typeof g.ticks === "function" ? g.ticks() : g.ticks;
        if (typeof ticks === "number") state.tick = ticks;
      }

      // Parse HUD from the <control-panel> custom element
      const panel = document.querySelector("control-panel");
      if (panel) {
        const parse = (s) => {
          if (!s) return 0;
          s = s.trim();
          if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
          if (s.endsWith("K")) return parseFloat(s) * 1000;
          return parseFloat(s) || 0;
        };

        // Gold: yellow-bordered div with gold coin icon
        const goldDivs = panel.querySelectorAll(".border-yellow-400 span");
        for (const span of goldDivs) {
          const t = span.textContent?.trim();
          if (t && /^[\d.]+[KMB]?$/.test(t)) {
            state.myGold = parse(t);
            break;
          }
        }

        // Troops: "X / Y" pattern in the panel (renderTroops divides by 10)
        const allText = panel.textContent || "";
        const troopMatch = allText.match(/([\d.]+[KM]?)\s*\/\s*([\d.]+[KM]?)/);
        if (troopMatch) {
          state.myTroops = parse(troopMatch[1]) * 10; // undo renderTroops /10
        }

        // NOTE: Territory % is NOT in the control panel — it's in the leaderboard.
        // The "X%" in the control panel is the attack ratio, NOT territory.
      }

      // Fix 4: Read unit counts directly from <unit-display> Lit properties
      // instead of fragile positional span parsing
      const unitDisplay = document.querySelector("unit-display");
      if (unitDisplay) {
        const cities = unitDisplay._cities || 0;
        const factories = unitDisplay._factories || 0;
        const ports = unitDisplay._port || 0;
        const defenses = unitDisplay._defensePost || 0;
        const silos = unitDisplay._missileSilo || 0;
        const sams = unitDisplay._samLauncher || 0;
        const warships = unitDisplay._warships || 0;
        state.hasSilo = silos > 0;
        state.hasPort = ports > 0;
        state.hasSAM = sams > 0;
        state.numWarships = warships;
        state.numCities = cities;
        state.numFactories = factories;
        state.numPorts = ports;
        state.numDefenses = defenses;
        state.numSilos = silos;
        state.numSAMs = sams;
        // units array: one entry per building for the model's len(units)/20 calc
        const totalUnits =
          cities + factories + ports + defenses + silos + sams + warships;
        state.units = new Array(totalUnits).fill("unit");
        state.hasUnits = totalUnits > 0;
      }

      // Parse incoming/outgoing attacks from <attacks-display> (light DOM)
      const attacksDisplay = document.querySelector("attacks-display");
      if (attacksDisplay) {
        // Count visible attack rows — incoming have sword icons, outgoing have different styling
        // The element stores arrays directly as properties
        if (attacksDisplay.incomingAttacks)
          state.incomingAttacks = attacksDisplay.incomingAttacks.length;
        if (attacksDisplay.outgoingAttacks)
          state.outgoingAttacks = attacksDisplay.outgoingAttacks.length;
        if (attacksDisplay.outgoingLandAttacks)
          state.outgoingAttacks += attacksDisplay.outgoingLandAttacks.length;
        if (attacksDisplay.outgoingBoats)
          state.outgoingAttacks += attacksDisplay.outgoingBoats.length;
        if (attacksDisplay.incomingBoats)
          state.incomingAttacks += attacksDisplay.incomingBoats.length;

        // Fix 2: Extract attacker names for relation tracking
        const attackerNames = [];
        const game = attacksDisplay.game;
        if (game && attacksDisplay.incomingAttacks) {
          for (const atk of attacksDisplay.incomingAttacks) {
            const p = game.playerBySmallID?.(atk.attackerID);
            if (p) attackerNames.push(p.displayName?.() || "");
          }
        }
        state._attackerNames = attackerNames;
      }

      // Fallback tick from sidebar timer if game.ticks() wasn't available
      if (state.tick === 0 && sidebar && typeof sidebar.timer === "number") {
        state.tick = Math.round(sidebar.timer * 10);
      }

      // Parse neighbor info from NameLayer DOM elements (much better than leaderboard)
      // NameLayer.ts creates div elements for each player with:
      //   .player-name-span → display name
      //   .player-troops → troop count text (e.g., "12.3K")
      // These exist in DOM even when hidden, but troop text is stale when display:none.
      // Parse a display number like "32.2K" → 32200
      const parseDisplayNum = (s) => {
        if (!s) return 0;
        s = s.trim();
        if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
        if (s.endsWith("K")) return parseFloat(s) * 1000;
        if (s.endsWith("B")) return parseFloat(s) * 1_000_000_000;
        return parseFloat(s) || 0;
      };
      // Troop displays use renderTroops(v) = renderNumber(v/10), so multiply by 10
      const parseTroops = (s) => parseDisplayNum(s) * 10;

      // Also parse leaderboard for territory % data
      const lbData = {}; // name -> { pct, gold, troops }
      const lb2 = document.querySelector("leader-board");
      if (lb2) {
        const root = lb2.shadowRoot || lb2;
        for (const el of root.querySelectorAll("tr, div, span")) {
          const t = el.textContent?.trim() || "";
          // Match rows like: "1 PlayerName 0.5% 148K 64.5K"
          const rowMatch = t.match(
            /^(\d+)\s+(.+?)\s+([\d.]+)%\s+([\d.]+[KMB]?)\s+([\d.]+[KMB]?)$/,
          );
          if (rowMatch) {
            lbData[rowMatch[2].trim()] = {
              pct: parseFloat(rowMatch[3]) / 100,
              gold: parseDisplayNum(rowMatch[4]), // gold uses renderNumber (no /10)
              troops: parseTroops(rowMatch[5]), // troops uses renderTroops (/10)
            };
          }
        }
      }

      // Read all player labels from the NameLayer DOM
      const playerMap = new Map(); // name -> { troops, visible }
      const nameSpans = document.querySelectorAll(".player-name-span");
      for (const span of nameSpans) {
        const name = span.textContent?.trim();
        if (!name) continue;
        // Walk up to the container element to find the sibling troops div
        const container = span.closest(".player-name")?.parentElement;
        if (!container) continue;
        const troopsEl = container.querySelector(".player-troops");
        const troops = troopsEl ? parseTroops(troopsEl.textContent) : 0;
        const visible = container.style.display !== "none";
        // Get label position (NameLayer positions labels at territory center)
        const rect = container.getBoundingClientRect();
        const labelX = rect.left + rect.width / 2;
        const labelY = rect.top + rect.height / 2;
        playerMap.set(name, { troops, visible, labelX, labelY });
      }

      // Detect land neighbors via canvas color matching:
      // 1. Sample each player's territory color at their label position
      // 2. Find our territory border pixels and sample adjacent non-ours colors
      // 3. Match adjacent colors to player colors → those are land neighbors
      const landNeighborNames = new Set();
      const canvas = document.querySelector("canvas");
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const W = canvas.width,
            H = canvas.height;
          const canvasRect = canvas.getBoundingClientRect();
          const centerData = ctx.getImageData(
            Math.floor(W / 2),
            Math.floor(H / 2),
            1,
            1,
          ).data;
          const myR = centerData[0],
            myG = centerData[1],
            myB = centerData[2];

          if (myR + myG + myB > 60 && !(myB > myR + myG + 50)) {
            // Step 1: Sample territory color at each player's label position
            const playerColors = []; // [{name, r, g, b}]
            for (const [name, info] of playerMap) {
              if (name === botName || name.includes(botName.slice(0, 6)))
                continue;
              if (!info.labelX || !info.labelY) continue;
              // Convert page coords → canvas pixel coords
              const cx = Math.floor(
                ((info.labelX - canvasRect.left) / canvasRect.width) * W,
              );
              const cy = Math.floor(
                ((info.labelY - canvasRect.top) / canvasRect.height) * H,
              );
              if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
              const px = ctx.getImageData(cx, cy, 1, 1).data;
              // Skip water/UI pixels
              if (px[0] + px[1] + px[2] < 60) continue;
              if (px[2] > px[0] + px[1] + 50) continue;
              playerColors.push({ name, r: px[0], g: px[1], b: px[2] });
            }

            // Step 2: Scan grid, find our border, collect adjacent non-ours colors
            const STEP = 16;
            const imgData = ctx.getImageData(0, 0, W, H).data;
            const cols = Math.floor(W / STEP),
              rows = Math.floor(H / STEP);
            const mineGrid = [];
            for (let gx = 0; gx < cols; gx++) {
              mineGrid[gx] = [];
              for (let gy = 0; gy < rows; gy++) {
                const px = gx * STEP + 8,
                  py = gy * STEP + 8;
                const idx = (py * W + px) * 4;
                const dr = imgData[idx] - myR,
                  dg = imgData[idx + 1] - myG,
                  db = imgData[idx + 2] - myB;
                mineGrid[gx][gy] = Math.sqrt(dr * dr + dg * dg + db * db) < 45;
              }
            }

            // Collect colors of non-ours cells adjacent to our border
            const adjacentColors = [];
            for (let gx = 1; gx < cols - 1; gx++) {
              for (let gy = 1; gy < rows - 1; gy++) {
                if (!mineGrid[gx][gy]) continue;
                const dirs = [
                  [gx - 1, gy],
                  [gx + 1, gy],
                  [gx, gy - 1],
                  [gx, gy + 1],
                ];
                for (const [nx, ny] of dirs) {
                  if (mineGrid[nx][ny]) continue;
                  const px = nx * STEP + 8,
                    py = ny * STEP + 8;
                  const idx = (py * W + px) * 4;
                  const r = imgData[idx],
                    g = imgData[idx + 1],
                    b = imgData[idx + 2];
                  // Skip water (dark or very blue)
                  if (r + g + b < 60) continue;
                  if (b > r + g + 50) continue;
                  adjacentColors.push({ r, g, b });
                }
              }
            }

            // Step 3: Match adjacent colors → player colors
            const COLOR_THRESHOLD = 50;
            for (const ac of adjacentColors) {
              for (const pc of playerColors) {
                const dr = ac.r - pc.r,
                  dg = ac.g - pc.g,
                  db = ac.b - pc.b;
                if (Math.sqrt(dr * dr + dg * dg + db * db) < COLOR_THRESHOLD) {
                  landNeighborNames.add(pc.name);
                }
              }
            }
          }
        }
      }

      // Build neighbors list from DOM labels, enriched with leaderboard data
      const neighbors = [];
      for (const [name, info] of playerMap) {
        if (name === botName || name.includes(botName.slice(0, 6))) continue;
        if (name === "") continue;
        // Keep Wilderness in the list — model learned to attack it for expansion

        const lb = lbData[name] || {};
        let tiles = lb.pct ? Math.round(lb.pct * state.totalMapTiles) : 0;
        if (tiles === 0 && info.troops > 0) {
          // Rough estimate: ~0.2 tiles per troop (troops already x10 corrected)
          tiles = Math.round(info.troops * 0.2);
        }

        neighbors.push({
          id: name,
          name: name,
          tiles,
          troops: info.troops || lb.troops || 0,
          relation: 2, // Relation.Neutral = 2 (matches training env)
          alive: true,
          isLandNeighbor: landNeighborNames.has(name),
          visible: info.visible,
          labelX: info.labelX,
          labelY: info.labelY,
        });
      }

      // Sort to match training: land neighbors first, then by territory size
      neighbors.sort((a, b) => {
        if (a.isLandNeighbor !== b.isLandNeighbor)
          return a.isLandNeighbor ? -1 : 1;
        return b.tiles - a.tiles;
      });
      state.neighbors = neighbors.slice(0, 16);

      // Find our territory % from the <leader-board> Lit element.
      // Access the component's `players` array directly (light DOM Lit component).
      const lb = document.querySelector("leader-board");
      if (lb && lb.players) {
        const myEntry = lb.players.find((p) => p.isMyPlayer);
        if (myEntry && myEntry.score) {
          const pctMatch = myEntry.score.match(/([\d.]+)%/);
          if (pctMatch) {
            state.territoryPct = parseFloat(pctMatch[1]) / 100;
          }
        }
      }

      // Derive myTiles from territoryPct (so vec[0] = myTiles/total is non-zero)
      state.myTiles = Math.round(state.territoryPct * state.totalMapTiles);

      return state;
    },
    botName,
  );
}

// ── Canvas helpers ───────────────────────────────────────────────

async function getCanvasBounds(page) {
  const canvas = await page.$("canvas");
  if (!canvas) return null;
  const box = await canvas.boundingBox();
  return box;
}

// ── Territory pixel scanner ──────────────────────────────────────
// Sample canvas pixels to find our territory tiles vs enemy/water/empty.
// Returns { interior: [{x,y},...], border: [{x,y},...], myColor: {r,g,b} }
// Call AFTER centerCamera() so screen center = our territory.
async function scanTerritory(page) {
  return safeEval(page, () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const W = canvas.width;
    const H = canvas.height;
    const STEP = 12; // sample every 12px (~120x75 grid on 1400x900)

    // Read center pixel — that's our territory color (after centering)
    const centerData = ctx.getImageData(
      Math.floor(W / 2),
      Math.floor(H / 2),
      1,
      1,
    ).data;
    const myR = centerData[0],
      myG = centerData[1],
      myB = centerData[2];

    // Skip if center looks like water (very dark or very blue)
    if (myR + myG + myB < 60) return null;
    if (myB > myR + myG + 50) return null;

    // Sample the canvas in a grid
    const grid = []; // [{x, y, mine: bool}]
    const cols = Math.floor(W / STEP);
    const rows = Math.floor(H / STEP);
    // Read all pixels at once (much faster than per-pixel getImageData)
    const imgData = ctx.getImageData(0, 0, W, H).data;

    for (let gx = 0; gx < cols; gx++) {
      for (let gy = 0; gy < rows; gy++) {
        const px = gx * STEP + Math.floor(STEP / 2);
        const py = gy * STEP + Math.floor(STEP / 2);
        const idx = (py * W + px) * 4;
        const r = imgData[idx],
          g = imgData[idx + 1],
          b = imgData[idx + 2];

        // Color similarity — allow some tolerance for shading/borders
        const dr = r - myR,
          dg = g - myG,
          db = b - myB;
        const dist = Math.sqrt(dr * dr + dg * dg + db * db);
        grid.push({ x: px, y: py, gx, gy, mine: dist < 45 });
      }
    }

    // Build a 2D lookup grid for O(1) neighbor checks
    const mineGrid = new Array(cols)
      .fill(null)
      .map(() => new Array(rows).fill(false));
    for (const cell of grid) {
      if (cell.mine) mineGrid[cell.gx][cell.gy] = true;
    }

    // Classify: interior = surrounded by our tiles, border = adjacent to non-ours
    const interior = [];
    const border = [];

    for (const cell of grid) {
      if (!cell.mine) continue;
      // Skip UI regions: top-left leaderboard, bottom HUD
      if (cell.x < W * 0.15 && cell.y < H * 0.35) continue;
      if (cell.y > H * 0.82) continue;
      if (cell.x > W * 0.9) continue;

      const { gx, gy } = cell;
      const allMine =
        (gx > 0 ? mineGrid[gx - 1][gy] : false) &&
        (gx < cols - 1 ? mineGrid[gx + 1][gy] : false) &&
        (gy > 0 ? mineGrid[gx][gy - 1] : false) &&
        (gy < rows - 1 ? mineGrid[gx][gy + 1] : false);

      if (allMine) {
        interior.push({ x: cell.x, y: cell.y });
      } else {
        border.push({ x: cell.x, y: cell.y });
      }
    }

    return {
      interior,
      border,
      myColor: { r: myR, g: myG, b: myB },
      totalMine: interior.length + border.length,
    };
  });
}

// Get the "safe zone" — game area excluding UI overlays:
// - Top-left: leaderboard (~15% width, ~30% height)
// - Bottom-left: ad banner (~15% width, ~30% height)
// - Top-right: settings/clock (~10% width)
// - Bottom: HUD bar (~15% height)
function getSafeZone(box) {
  return {
    left: box.x + box.width * 0.15, // clear of leaderboard + ad
    right: box.x + box.width * 0.9, // clear of settings gear
    top: box.y + box.height * 0.05,
    bottom: box.y + box.height * 0.82, // clear of bottom HUD
    cx: box.x + box.width * 0.52, // slightly right of center (away from leaderboard)
    cy: box.y + box.height * 0.42, // slightly above center (away from HUD)
    width: box.width * 0.75, // effective playable width
    height: box.height * 0.77, // effective playable height
  };
}

// Ensure the canvas has focus so keyboard events reach the game.
// Clicks near center — this also sends troops, which is fine.
async function focusCanvas(page) {
  const canvas = await page.$("canvas");
  if (canvas) {
    await canvas.focus().catch(() => {});
    // Move mouse to canvas center but DON'T click — clicking center sends troops
    // to our own territory (wastes them on internal movement)
    const box = await canvas.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await sleep(30);
    }
  }
}

// Center camera on our territory.
// Strategy: try clicking our row in the leaderboard (reliable), fall back to 'c' key.
async function centerCamera(page) {
  // Method 1: Find our name in the leaderboard and click it
  const clicked = await safeEval(
    page,
    (botName) => {
      // The leaderboard rows contain player names. Find ours and click it.
      for (const el of document.querySelectorAll("*")) {
        const text = el.textContent?.trim() || "";
        const r = el.getBoundingClientRect();
        // Leaderboard is in the top-left, rows are narrow
        if (r.x > window.innerWidth * 0.3 || r.y > window.innerHeight * 0.6)
          continue;
        if (r.width < 50 || r.height < 10 || r.height > 40) continue;
        if (text.includes(botName)) {
          el.click();
          return {
            x: r.x + r.width / 2,
            y: r.y + r.height / 2,
            method: "leaderboard",
          };
        }
      }
      return null;
    },
    BOT_NAME,
  );

  if (clicked) {
    log(`Center: clicked leaderboard row for "${BOT_NAME}"`);
    await sleep(800);
    return;
  }

  // Method 2: Fall back to 'c' key
  await focusCanvas(page);
  await page.keyboard.press("c").catch(() => {});
  await sleep(800);
  log("Center: used 'c' key fallback");
}

// ── Zoom helpers ──────────────────────────────────────────────────
// Apply a single zoom step matching the game's formula:
//   scale /= (1 + delta/600), clamped [0.2, 20]
// Use small deltas (~100) for ~20% zoom per step, NOT 500 (which is 6x per step!).
async function zoomStep(page, delta) {
  const box = await getCanvasBounds(page);
  if (!box) return;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel({ deltaY: delta });
  const zoomFactor = 1 + delta / 600;
  currentScale = Math.max(0.2, Math.min(20, currentScale / zoomFactor));
  await sleep(80);
}

// Generate a random point well inside our territory (tight around center after 'c')
function randomInterior(zone) {
  const spread = 30;
  return {
    x: zone.cx + (Math.random() - 0.5) * spread * 2,
    y: zone.cy + (Math.random() - 0.5) * spread * 2,
  };
}

// Generate a random point at/beyond our border (for expansion/attacks).
// After centering + dynamic zoom, our territory is always roughly centered.
// Use 80-200px to ensure clicks reach BEYOND our territory into wilderness.
// Clicking our own tiles does nothing (canAttack=false for own tiles).
function randomBorder(zone) {
  const angle = Math.random() * Math.PI * 2;
  const dist = 80 + Math.random() * 120;
  const x = zone.cx + Math.cos(angle) * dist;
  const y = zone.cy + Math.sin(angle) * dist;
  return {
    x: Math.max(zone.left, Math.min(zone.right, x)),
    y: Math.max(zone.top, Math.min(zone.bottom, y)),
  };
}

// ── Execute RL action via keyboard/mouse ─────────────────────────

// Action type IDs matching env.py's 17-action space
const ACTION_NOOP = 0;
const ACTION_ATTACK = 1;
const ACTION_BOAT_ATTACK = 2;
const ACTION_RETREAT = 3;
const ACTION_BUILD_CITY = 4;
const ACTION_BUILD_FACTORY = 5;
const ACTION_BUILD_DEFENSE = 6;
const ACTION_BUILD_PORT = 7;
const ACTION_BUILD_SAM = 8;
const ACTION_BUILD_SILO = 9;
const ACTION_BUILD_WARSHIP = 10;
const ACTION_LAUNCH_ATOM = 11;
const ACTION_LAUNCH_HBOMB = 12;
const ACTION_LAUNCH_MIRV = 13;
const ACTION_MOVE_WARSHIP = 14;
const ACTION_UPGRADE = 15;
const ACTION_DELETE_UNIT = 16;

// Keyboard shortcuts from InputHandler.ts keybinds
// Only actual build actions (not nukes — those need silo targeting)
const BUILD_KEYS = {
  [ACTION_BUILD_CITY]: { key: "1", name: "city", minGold: 125_000 },
  [ACTION_BUILD_FACTORY]: { key: "2", name: "factory", minGold: 125_000 },
  [ACTION_BUILD_PORT]: { key: "3", name: "port", minGold: 125_000 },
  [ACTION_BUILD_DEFENSE]: { key: "4", name: "defense", minGold: 50_000 },
  [ACTION_BUILD_SILO]: { key: "5", name: "silo", minGold: 1_000_000 },
  [ACTION_BUILD_SAM]: { key: "6", name: "SAM", minGold: 1_500_000 },
  [ACTION_BUILD_WARSHIP]: { key: "7", name: "warship", minGold: 250_000 },
};

// Nuke launch keys (separate from builds — these require a silo)
const NUKE_KEYS = {
  [ACTION_LAUNCH_ATOM]: { key: "8", name: "atom bomb", minGold: 750_000 },
  [ACTION_LAUNCH_HBOMB]: { key: "9", name: "H-bomb", minGold: 5_000_000 },
  [ACTION_LAUNCH_MIRV]: { key: "0", name: "MIRV", minGold: 10_000_000 },
};

// Clear ghost structure (build mode) so clicks register as attacks.
// If ghostStructure is set from a previous build key, ALL attack clicks are
// silently blocked by ClientGameRunner.inputEvent.
async function clearBuildMode(page) {
  await page.keyboard.press("Escape").catch(() => {});
  await sleep(50);
}

async function executeRLAction(page, action, zone, goldAmount, neighbors) {
  if (!action || !zone) return;

  const actionType = action.actionType;
  const troopFraction = action.troopFraction || 0.5;
  const gold = goldAmount || 0;
  neighbors = neighbors || [];

  // Always clear build mode before non-build actions
  if (!BUILD_KEYS[actionType] && !NUKE_KEYS[actionType]) {
    await clearBuildMode(page);
  }

  if (actionType === ACTION_NOOP) {
    // Do nothing — trust the model
  } else if (
    actionType === ACTION_ATTACK ||
    actionType === ACTION_BOAT_ATTACK
  ) {
    // Adjust attack ratio using T (down 10%) / Y (up 10%) to match troopFraction
    // troopFraction: 0.2, 0.4, 0.6, 0.8, or 1.0
    const steps = Math.round((troopFraction - currentAttackRatio) / 0.1);
    if (steps !== 0) {
      await focusCanvas(page);
      const key = steps > 0 ? "y" : "t";
      for (let i = 0; i < Math.abs(steps); i++) await page.keyboard.press(key);
      currentAttackRatio = troopFraction;
      await sleep(30);
    }

    // Aim toward the target neighbor's screen position (from NameLayer labels)
    const targetIdx = action.targetIdx || 0;
    const target = neighbors[targetIdx];
    let x, y;
    if (target && target.labelX && target.labelY) {
      // Click on the line between our center and the target's label,
      // at our border distance (80-200px from center) in that direction
      const dx = target.labelX - zone.cx;
      const dy = target.labelY - zone.cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 10) {
        // Aim toward target, 80-200px from our center (our border region)
        const clickDist = 80 + Math.random() * 120;
        x = zone.cx + (dx / dist) * clickDist;
        y = zone.cy + (dy / dist) * clickDist;
      } else {
        // Target is very close to center — fall back to random border
        ({ x, y } = randomBorder(zone));
      }
    } else {
      // No position data for target — fall back to random border
      ({ x, y } = randomBorder(zone));
    }

    await page.mouse.click(x, y);
    await sleep(50);
    const targetName = target ? target.name : "unknown";
    log(
      `RL: Attack ${targetName} at (${Math.round(x)},${Math.round(y)}) ratio=${troopFraction}`,
    );
  } else if (actionType === ACTION_RETREAT) {
    log("RL: Retreat (noop in live)");
  } else if (BUILD_KEYS[actionType]) {
    const { key, name, minGold } = BUILD_KEYS[actionType];

    // Don't waste time trying to build if we can't afford it
    if (gold < minGold) {
      log(
        `RL: Skip build ${name} — need ${(minGold / 1000).toFixed(0)}K gold, have ${(gold / 1000).toFixed(0)}K. Expanding instead.`,
      );
      for (let i = 0; i < 5; i++) {
        const { x, y } = randomBorder(zone);
        await page.mouse.click(x, y);
        await sleep(50);
      }
      return;
    }

    // Center + zoom in close for precise build placement
    await centerCamera(page);
    const savedScale = currentScale;
    // Zoom in to scale ~12 so our territory fills the screen
    const zoomInSteps = Math.max(
      0,
      Math.round(Math.log(12 / currentScale) / Math.log(1.2)),
    );
    for (let i = 0; i < Math.min(zoomInSteps, 10); i++)
      await zoomStep(page, -100);
    await sleep(200);

    const box2 = await getCanvasBounds(page);
    if (!box2) return;
    const trueCx = box2.x + box2.width / 2;
    const trueCy = box2.y + box2.height / 2;

    const isBorderBuilding =
      actionType === ACTION_BUILD_DEFENSE ||
      actionType === ACTION_BUILD_SAM ||
      actionType === ACTION_BUILD_PORT;

    // At high zoom, our territory should fill most of screen after centering
    // Just click near center (interior) or slightly off-center (border)
    let spot;
    if (isBorderBuilding) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 60 + Math.random() * 40;
      spot = {
        x: trueCx + Math.cos(angle) * dist,
        y: trueCy + Math.sin(angle) * dist,
      };
    } else {
      spot = {
        x: trueCx + (Math.random() - 0.5) * 60,
        y: trueCy + (Math.random() - 0.5) * 60,
      };
    }

    // Step 1: Move mouse to spot
    await page.mouse.move(spot.x, spot.y);
    await sleep(100);
    // Step 2: Press build key → ghost structure appears at mouse location
    await focusCanvas(page);
    await page.keyboard.press(key).catch(() => {});
    await sleep(500);
    // Step 3: Click to confirm placement
    await page.mouse.click(spot.x, spot.y);
    await sleep(200);
    // Step 4: Escape to exit build mode if placement failed
    await page.keyboard.press("Escape").catch(() => {});
    // Track build feedback: if gold dropped significantly, build likely succeeded
    lastBuildResult = "none"; // will be updated on next extractGameState
    lastActionSucceeded = true; // optimistic — model learns from actual results
    log(
      `RL: Build ${name} at (${Math.round(spot.x)},${Math.round(spot.y)}) scale=${currentScale.toFixed(1)} [gold=${(gold / 1000).toFixed(0)}K]`,
    );

    // Zoom back out to previous level
    const zoomOutSteps = Math.max(
      0,
      Math.round(Math.log(currentScale / savedScale) / Math.log(1.2)),
    );
    for (let i = 0; i < Math.min(zoomOutSteps, 10); i++)
      await zoomStep(page, 100);
  } else if (NUKE_KEYS[actionType]) {
    const { key, name, minGold } = NUKE_KEYS[actionType];
    if (gold < minGold) {
      log(`RL: Skip ${name} — need ${(minGold / 1000).toFixed(0)}K gold`);
      return;
    }
    // Nukes: press key, then click on enemy territory
    await page.keyboard.press(key).catch(() => {});
    await sleep(300);
    // Aim toward target player's label position
    const nukeTargetIdx = action.targetIdx || 0;
    const nukeTarget = neighbors[nukeTargetIdx];
    let nx, ny;
    if (nukeTarget && nukeTarget.labelX && nukeTarget.labelY) {
      // Click near/at the target's label (their territory center)
      nx = nukeTarget.labelX + (Math.random() - 0.5) * 40;
      ny = nukeTarget.labelY + (Math.random() - 0.5) * 40;
    } else {
      const angle = Math.random() * Math.PI * 2;
      const dist = 150 + Math.random() * 100;
      nx = zone.cx + Math.cos(angle) * dist;
      ny = zone.cy + Math.sin(angle) * dist;
    }
    await page.mouse.move(nx, ny);
    await sleep(300);
    await page.mouse.click(nx, ny);
    await sleep(200);
    await page.keyboard.press("Escape").catch(() => {});
    const nukeTargetName = nukeTarget ? nukeTarget.name : "unknown";
    log(
      `RL: Launch ${name} at ${nukeTargetName} (${Math.round(nx)},${Math.round(ny)})`,
    );
    return { nukeLaunched: true };
  } else if (actionType === ACTION_UPGRADE) {
    const spot = randomInterior(zone);
    await page.mouse.click(spot.x, spot.y);
    await sleep(200);
    log("RL: Upgrade");
  } else if (actionType === ACTION_DELETE_UNIT) {
    log("RL: Delete unit (noop)");
  }
}

// ── Main ─────────────────────────────────────────────────────────

async function main() {
  log(`RL Bot "${BOT_NAME}" starting (policy: ${POLICY_URL})...`);

  // Check policy server is up
  try {
    const test = await queryPolicy({
      myTiles: 0,
      myTroops: 0,
      myGold: 0,
      territoryPct: 0,
      totalMapTiles: 2000000,
      incomingAttacks: 0,
      outgoingAttacks: 0,
      units: [],
      neighbors: [],
    });
    if (test) log(`Policy server OK — test action: ${JSON.stringify(test)}`);
    else throw new Error("no response");
  } catch (e) {
    log(`ERROR: Policy server at ${POLICY_URL} not reachable: ${e.message}`);
    log(
      `Start it with: cd /Users/joshua/openfront-rl/rl && python play.py --model checkpoints/best_model.pt --mode server --port 8765`,
    );
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: null,
    userDataDir: "/tmp/openfront-bot-profile",
    args: [
      "--window-size=1400,900",
      "--disable-features=site-per-process",
      "--autoplay-policy=no-user-gesture-required",
      "--disable-blink-features=AutomationControlled",
      "--no-restore-session-state",
      "--disable-session-crashed-bubble",
      "--disable-notifications",
    ],
  });

  const allP = await browser.pages();
  for (let i = 1; i < allP.length; i++) await allP[i].close().catch(() => {});

  let page = allP[0];
  page.setDefaultNavigationTimeout(60_000);
  page.on("dialog", async (d) => {
    await d.accept().catch(() => {});
  });

  browser.on("targetcreated", async (target) => {
    if (target.type() === "page") {
      await sleep(2000);
      try {
        const p = await target.page();
        if (
          p &&
          !p.url().includes("openfront.io") &&
          p.url() !== "about:blank"
        ) {
          log(`Closing ad tab: ${p.url().slice(0, 50)}`);
          await p.close();
          page = await safePage(browser);
          await page.bringToFront();
        }
      } catch {}
    }
  });

  // Block ad requests at the network level (before navigation)
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    const url = req.url();
    const blocked = [
      "doubleclick.net",
      "googlesyndication.com",
      "googleadservices.com",
      "adservice.google",
      "pagead2.googlesyndication",
      "ads.google.com",
      "hrblock.com",
      "amazon-adsystem",
      "adnxs.com",
      "adsrvr.org",
      "criteo.com",
      "taboola.com",
      "outbrain.com",
      "ad.doubleclick",
      "googletagmanager.com/gtag",
      "facebook.net/tr",
    ];
    if (blocked.some((d) => url.includes(d))) {
      req.abort().catch(() => {});
    } else {
      req.continue().catch(() => {});
    }
  });
  log("Ad blocking enabled (network-level request interception)");

  // ── Join game ──
  await joinGame(page);

  log("=== RL Game loop ===");

  const t0 = Date.now();
  let tick = 0;
  let spawned = false;
  let spawnAttempts = 0;
  let lastSnap = 0;
  let lastPolicyQuery = 0;
  let currentAction = null;
  let lastNeighbors = [];
  let lastRecenter = 0;
  lastGold = 0;
  currentScale = 1.8; // reset to game default at game start
  lastTerritoryPct = 0;
  let spawnTime = 0;
  let lastNeighborScan = 0; // Brief zoom-out to refresh neighbor label data
  const playerRelations = new Map(); // name -> Relation (0=Hostile,1=Distrustful,2=Neutral,3=Friendly)
  let nukeCount = 0; // Track nukes launched (increment on launch, decrement on resolve)

  while (true) {
    try {
      tick++;
      const elapsed = (Date.now() - t0) / 1000;

      if (tick % 30 === 0) page = await safePage(browser);

      // Periodic screenshot
      if (Date.now() - lastSnap > 30_000) {
        lastSnap = Date.now();
        await snap(page, `game-${Math.floor(elapsed)}s`);
      }

      // ── DEATH DETECTION ──
      const deathCheck = await safeEval(page, () => {
        for (const el of document.querySelectorAll("*")) {
          const t = el.textContent?.trim() || "";
          if (t === "You died") {
            const r = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            if (
              r.width > 0 &&
              r.height > 0 &&
              style.display !== "none" &&
              style.visibility !== "hidden" &&
              style.opacity !== "0"
            ) {
              return { dead: true };
            }
          }
        }
        return { dead: false };
      });

      if (deathCheck?.dead) {
        log(
          `====== DEAD after ${((Date.now() - t0) / 1000).toFixed(0)}s. GG! ======`,
        );
        await sleep(3000);
        await snap(page, "final-death");
        // Click Exit Game
        for (let attempt = 0; attempt < 5; attempt++) {
          const exitBtn = await safeEval(page, () => {
            for (const el of document.querySelectorAll("*")) {
              if (el.textContent?.trim() === "Exit Game") {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                if (r.width > 30 && r.height > 15 && s.display !== "none") {
                  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
                }
              }
            }
            return null;
          });
          if (exitBtn) {
            await page.mouse.click(exitBtn.x, exitBtn.y);
            await sleep(1500);
          } else {
            await sleep(2000);
          }
        }
        log("Done. Exiting RL bot.");
        process.exit(0);
      }

      // ── SPAWN PHASE ──
      // Keep clicking the map to pick a starting location.
      // Detect spawn by checking if the troop HUD appeared (troops > 0).
      if (!spawned) {
        const box = await getCanvasBounds(page);
        if (box) {
          const cx = box.x + box.width * (0.3 + Math.random() * 0.4);
          const cy = box.y + box.height * (0.2 + Math.random() * 0.4);
          log(`Spawn click at (${Math.round(cx)},${Math.round(cy)})`);
          await page.mouse.click(cx, cy);
        }
        spawnAttempts++;
        await sleep(2000);

        // Check if we spawned: look for the troops HUD (only visible after spawning)
        const hudState = await extractGameState(page, BOT_NAME);
        // Troops show ~1K even before spawn. Gold > 0 only after spawning.
        // Also allow fallback after enough attempts.
        const didSpawn =
          (hudState && hudState.myGold > 100) || spawnAttempts >= 15;

        if (didSpawn || spawnAttempts > 15) {
          spawned = true;
          spawnTime = Date.now();
          log(
            `Spawned! troops=${hudState?.myTroops || "?"}, territory=${((hudState?.territoryPct || 0) * 100).toFixed(1)}%`,
          );

          // Center camera and zoom in to ~7x (1.8 → ~6.4)
          await centerCamera(page);
          for (let i = 0; i < 7; i++) await zoomStep(page, -100);
          await sleep(500);
          lastRecenter = Date.now();
        }
        continue;
      }

      const box = await getCanvasBounds(page);
      if (!box) {
        await sleep(500);
        continue;
      }
      const zone = getSafeZone(box);

      // ── RECENTER & DYNAMIC ZOOM every 10 seconds ──
      if (Date.now() - lastRecenter > 10000) {
        lastRecenter = Date.now();
        await centerCamera(page);

        // Zoom targets: need enough zoom to see and click on our territory
        const tPct = lastTerritoryPct;
        let targetScale;
        if (tPct < 0.005) targetScale = 7.0;
        else if (tPct < 0.01) targetScale = 5.0;
        else if (tPct < 0.03) targetScale = 3.5;
        else if (tPct < 0.05) targetScale = 2.5;
        else targetScale = 1.8;

        const scaleRatio = currentScale / targetScale;
        if (scaleRatio > 1.5 || scaleRatio < 0.67) {
          const stepsNeeded = Math.round(
            Math.log(targetScale / currentScale) / Math.log(1.2),
          );
          const clamped = Math.max(-6, Math.min(6, stepsNeeded));
          if (clamped !== 0) {
            const delta = clamped > 0 ? -100 : 100;
            for (let i = 0; i < Math.abs(clamped); i++)
              await zoomStep(page, delta);
            log(
              `Zoom: ${currentScale.toFixed(1)} → target ${targetScale.toFixed(1)}, ${Math.abs(clamped)} steps ${clamped > 0 ? "in" : "out"} (territory=${(tPct * 100).toFixed(2)}%)`,
            );
          }
        }
      }

      // ── NEIGHBOR SCAN: briefly zoom out to refresh player labels ──
      // NameLayer only updates troop text when elements are visible (display !== none).
      // Visibility requires scale * baseSize >= 7. By zooming to ~1.5, even small
      // players (baseSize ~5) become visible, refreshing their troop counts.
      // Fix 6: scan every 10s instead of 30s for better isLandNeighbor accuracy
      if (
        Date.now() - lastNeighborScan > 10000 &&
        Date.now() - spawnTime > 5000
      ) {
        lastNeighborScan = Date.now();
        const savedScale2 = currentScale;
        // Zoom out to ~1.5 to make most labels visible
        if (currentScale > 2.0) {
          const outSteps = Math.round(
            Math.log(currentScale / 1.5) / Math.log(1.2),
          );
          const clamped = Math.min(outSteps, 8);
          for (let i = 0; i < clamped; i++) await zoomStep(page, 100);
          await sleep(600); // Wait for NameLayer to refresh (500ms rate)
          // Zoom back in
          for (let i = 0; i < clamped; i++) await zoomStep(page, -100);
          log(`Neighbor scan: zoomed out ${clamped} steps to refresh labels`);
        }
      }

      // ── QUERY POLICY SERVER (interval matches training: TICKS_PER_STEP × TURN_INTERVAL_MS) ──
      if (Date.now() - lastPolicyQuery > POLICY_INTERVAL_MS) {
        lastPolicyQuery = Date.now();

        const gameState = await extractGameState(page, BOT_NAME);
        if (gameState) {
          // Fix 2: Update relation tracking from incoming attacks
          if (gameState._attackerNames) {
            for (const name of gameState._attackerNames) {
              if (name) playerRelations.set(name, 0); // Hostile
            }
          }
          // Apply tracked relations to neighbors
          for (const n of gameState.neighbors) {
            if (playerRelations.has(n.name)) {
              n.relation = playerRelations.get(n.name);
            }
          }
          // Fix 5: Pass nuke count from bot state
          gameState.numNukes = nukeCount;

          // Inject Wilderness as first neighbor so model can target it for expansion
          const borderPt = randomBorder(zone);
          gameState.neighbors.unshift({
            id: "wilderness",
            name: "Wilderness",
            tiles: Math.round(
              gameState.totalMapTiles * (1 - (gameState.territoryPct || 0)),
            ),
            troops: 0,
            relation: 2, // Neutral
            alive: true,
            isLandNeighbor: true,
            visible: true,
            labelX: borderPt.x,
            labelY: borderPt.y,
          });
          // Keep max 16 neighbors
          if (gameState.neighbors.length > 16) {
            gameState.neighbors = gameState.neighbors.slice(0, 16);
          }

          lastGold = gameState.myGold || 0;
          lastTerritoryPct = gameState.territoryPct || 0;
          const g = lastGold;
          const hasN = gameState.neighbors.length > 0;
          const hasT = gameState.myTroops > 10;
          const hasOut = (gameState.outgoingAttacks || 0) > 0;
          const hasUnits = gameState.hasUnits || false;
          const hasSilo = gameState.hasSilo || false;
          const hasPort = gameState.hasPort || false;
          const numWarships = gameState.numWarships || 0;
          const numCities = gameState.numCities || 0;

          // City cost scales: min(1M, 2^numCities * 125K) — matches DefaultConfig.ts
          const cityCost = Math.min(
            1_000_000,
            Math.pow(2, numCities) * 125_000,
          );

          // Action mask matching env_server.ts canBuild checks
          gameState.actionMask = [
            true, // 0: NOOP
            hasN && hasT, // 1: ATTACK
            hasN && hasT && hasPort, // 2: BOAT_ATTACK
            hasOut, // 3: RETREAT
            g >= cityCost, // 4: BUILD_CITY (exponential cost)
            g >= 125_000, // 5: BUILD_FACTORY
            g >= 50_000, // 6: BUILD_DEFENSE
            g >= 125_000, // 7: BUILD_PORT
            g >= 1_500_000, // 8: BUILD_SAM
            g >= 1_000_000, // 9: BUILD_SILO
            g >= 250_000 && hasPort, // 10: BUILD_WARSHIP
            hasSilo && hasN && g >= 750_000, // 11: LAUNCH_ATOM
            hasSilo && hasN && g >= 5_000_000, // 12: LAUNCH_HBOMB
            hasSilo && hasN && g >= 10_000_000, // 13: LAUNCH_MIRV
            numWarships > 0 && hasN, // 14: MOVE_WARSHIP
            hasUnits && g >= 50_000, // 15: UPGRADE
            hasUnits, // 16: DELETE_UNIT
          ];
          // Add build feedback (tracked from previous action)
          gameState.lastBuildResult = lastBuildResult;
          gameState.lastActionSucceeded = lastActionSucceeded;
          lastBuildResult = "none"; // reset after sending

          const action = await queryPolicy(gameState);
          if (action) {
            currentAction = action;
            lastNeighbors = gameState.neighbors || [];
            if (tick % 10 === 0) {
              log(
                `RL action: type=${action.actionType} target=${action.targetIdx} troops=${action.troopFraction} gold=${(lastGold / 1000).toFixed(0)}K`,
              );
            }
          }
        }
      }

      // ── EXECUTE RL POLICY ACTION (model controls all actions including expansion) ──
      const actionResult = await executeRLAction(
        page,
        currentAction,
        zone,
        lastGold,
        lastNeighbors,
      );
      if (actionResult?.nukeLaunched) nukeCount++;

      // ── Accept alliance requests ──
      if (tick % 10 === 0) {
        await safeEval(page, () => {
          for (const el of document.querySelectorAll("*")) {
            if (el.textContent?.trim() === "Accept") {
              const r = el.getBoundingClientRect();
              if (r.width > 20 && r.width < 200) {
                el.click();
                return true;
              }
            }
          }
          return false;
        });
      }

      // ── Status ──
      if (tick % 30 === 0) {
        const gs = await extractGameState(page, BOT_NAME);
        const pct = gs ? (gs.territoryPct * 100).toFixed(1) : "?";
        const troops = gs ? gs.myTroops : "?";
        const nNeighbors = gs ? gs.neighbors.length : 0;
        const visNeighbors = gs
          ? gs.neighbors.filter((n) => n.visible).length
          : 0;
        const units = gs ? gs.units.length : 0;
        const attacks = gs
          ? `in=${gs.incomingAttacks} out=${gs.outgoingAttacks}`
          : "";
        const bldgs = gs
          ? `silo=${gs.hasSilo} port=${gs.hasPort} sam=${gs.hasSAM} ships=${gs.numWarships}`
          : "";
        log(
          `[${elapsed.toFixed(0)}s] ${pct}% terr, troops=${troops}, gold=${(lastGold / 1000).toFixed(0)}K, tiles=${gs?.myTiles || 0}, tick=${gs?.tick || 0}, units=${units}, ${attacks}, ${bldgs}, neighbors=${nNeighbors}(${visNeighbors}vis), scale=${currentScale.toFixed(1)}`,
        );
      }
    } catch (err) {
      log(`Error: ${err.message.slice(0, 80)}`);
      if (
        err.message.includes("detached") ||
        err.message.includes("disposed") ||
        err.message.includes("closed")
      ) {
        await sleep(3000);
        page = await safePage(browser);
      }
    }

    await sleep(250);
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
