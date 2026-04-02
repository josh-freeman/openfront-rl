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
let lastGold = 0;
let lastTerritoryPct = 0;

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

async function extractGameState(page) {
  return safeEval(page, () => {
    const state = {
      myTiles: 0,
      myTroops: 0,
      myGold: 0,
      territoryPct: 0,
      totalMapTiles: 10000,
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

    // The bottom HUD bar contains: troops/maxTroops, territory%, gold
    // Look for elements in the bottom 15% of the screen
    const bottomEls = [];
    for (const el of document.querySelectorAll("*")) {
      const r = el.getBoundingClientRect();
      if (r.y > window.innerHeight * 0.85 && r.width > 20 && r.height > 5) {
        bottomEls.push({ el, text: el.textContent?.trim() || "", r });
      }
    }

    for (const { text } of bottomEls) {
      // Match troops: "17.3K / 49.7K"
      const troopMatch = text.match(/^([\d.]+[KM]?)\s*\/\s*([\d.]+[KM]?)$/);
      if (troopMatch) {
        const parse = (s) => {
          if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
          if (s.endsWith("K")) return parseFloat(s) * 1000;
          return parseFloat(s);
        };
        state.myTroops = parse(troopMatch[1]);
      }
      // Match territory: "20% (3.47K)" or just "20%"
      const pctMatch = text.match(/([\d.]+)%/);
      if (pctMatch && !troopMatch) {
        state.territoryPct = parseFloat(pctMatch[1]) / 100;
      }
      // Match gold: look for standalone number near the gold icon (bottom-right HUD area)
      const goldMatch = text.match(/^([\d.]+[KM]?)$/);
      if (goldMatch && !troopMatch) {
        const parse = (s) => {
          if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
          if (s.endsWith("K")) return parseFloat(s) * 1000;
          return parseFloat(s);
        };
        const val = parse(goldMatch[1]);
        // Gold is typically larger than territory% and troops number
        if (val > state.myGold) state.myGold = val;
      }
    }

    // Parse leaderboard (top-left table) for neighbor info
    const rows = [];
    for (const el of document.querySelectorAll("*")) {
      const t = el.textContent?.trim() || "";
      const r = el.getBoundingClientRect();
      // Leaderboard is in the top-left area
      if (r.x > window.innerWidth * 0.3 || r.y > window.innerHeight * 0.5)
        continue;
      // Match rows like: "1 PlayerName 0.5% 148K 64.5K"
      const rowMatch = t.match(
        /^(\d+)\s+(.+?)\s+([\d.]+)%\s+([\d.]+[KM]?)\s+([\d.]+[KM]?)$/,
      );
      if (rowMatch) {
        const parse = (s) => {
          if (s.endsWith("M")) return parseFloat(s) * 1_000_000;
          if (s.endsWith("K")) return parseFloat(s) * 1000;
          return parseFloat(s);
        };
        rows.push({
          id: rowMatch[2].trim(),
          name: rowMatch[2].trim(),
          tiles: Math.round(
            (parseFloat(rowMatch[3]) / 100) * state.totalMapTiles,
          ),
          troops: parse(rowMatch[5]),
          relation: 0,
          alive: true,
        });
      }
    }
    state.neighbors = rows.slice(0, 16);

    return state;
  });
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
    // Also click to ensure the game's event handlers see activity
    const box = await canvas.boundingBox();
    if (box) {
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
      await sleep(50);
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

// Estimate what fraction of the screen our territory occupies (after 'c' centering).
// At min zoom (scale 0.2) the full map fills the screen, so territory linear fraction
// ≈ sqrt(territoryPct). At higher scales we see less map, so territory appears bigger.
// Result is territory's approximate linear fraction of the viewport (0 to ~1).
function territoryScreenFraction() {
  let tPct = lastTerritoryPct;
  if (tPct < 0.0001) {
    // HUD shows 0.0% for small nations — use gold as very conservative proxy.
    // A nation earning 1K/s with 100K gold has been alive ~100s and probably
    // owns ~0.002% of a large map. Be conservative to keep clicks tight.
    tPct = Math.min(0.01, lastGold / 50_000_000);
  }
  const linear = Math.sqrt(Math.max(tPct, 0.00001));
  // Calibration: at scale 5.4, 0.01% territory → sqrt(0.0001)*5.4*2 = 0.108 → 11%
  // at scale 5.4, 0.1% territory → sqrt(0.001)*5.4*2 = 0.34 → 34%
  // at scale 1.8, 1% territory → sqrt(0.01)*1.8*2 = 0.36 → 36%
  return Math.min(0.4, linear * currentScale * 2);
}

// Generate a random point well inside our territory (tight around center after 'c')
function randomInterior(zone) {
  const frac = territoryScreenFraction();
  // Stay at 40% of territory radius — guaranteed to be ours
  const spread = Math.max(0.01, frac * 0.4);
  return {
    x: zone.cx + (Math.random() - 0.5) * zone.width * spread * 2,
    y: zone.cy + (Math.random() - 0.5) * zone.height * spread * 2,
  };
}

// Generate a random point at/beyond our border (for expansion/attacks)
function randomBorder(zone) {
  const frac = territoryScreenFraction();
  const spread = Math.max(0.03, frac);
  const angle = Math.random() * Math.PI * 2;
  // Ring around the edge of territory, extending slightly beyond
  const dist = spread * (0.8 + Math.random() * 0.5);
  const x = zone.cx + Math.cos(angle) * zone.width * dist;
  const y = zone.cy + Math.sin(angle) * zone.height * dist;
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

async function executeRLAction(page, action, zone, goldAmount) {
  if (!action || !zone) return;

  const actionType = action.actionType;
  const troopFraction = action.troopFraction || 0.5;
  const gold = goldAmount || 0;

  if (actionType === ACTION_NOOP) {
    // Expand into borders — click on border ring
    for (let i = 0; i < 5; i++) {
      const { x, y } = randomBorder(zone);
      await page.mouse.click(x, y);
      await sleep(50);
    }
  } else if (
    actionType === ACTION_ATTACK ||
    actionType === ACTION_BOAT_ATTACK
  ) {
    // Click on enemy territory (outer ring beyond our borders)
    const nClicks = Math.max(5, Math.floor(15 * troopFraction));
    for (let i = 0; i < nClicks; i++) {
      const { x, y } = randomBorder(zone); // Click borders to expand into enemies
      await page.mouse.click(x, y);
      await sleep(30);
    }
    log(`RL: Attack (${nClicks} clicks, troop=${troopFraction.toFixed(1)})`);
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

    // Center camera so screen center = our territory, then scan pixels
    await centerCamera(page);
    const box2 = await getCanvasBounds(page);
    if (!box2) return;

    const isBorderBuilding =
      actionType === ACTION_BUILD_DEFENSE ||
      actionType === ACTION_BUILD_SAM ||
      actionType === ACTION_BUILD_PORT;

    // Try pixel-based territory detection for precise placement
    const scan = await scanTerritory(page);
    let spot;

    if (scan && scan.totalMine > 5) {
      // Pick from scanned territory tiles
      const candidates = isBorderBuilding
        ? scan.border.length > 0
          ? scan.border
          : scan.interior
        : scan.interior.length > 0
          ? scan.interior
          : scan.border;
      const pick = candidates[Math.floor(Math.random() * candidates.length)];
      // Convert canvas pixel coords to page coords
      // canvas.width/height = internal resolution, box2 = CSS layout size
      const canvasDims = await safeEval(page, () => {
        const c = document.querySelector("canvas");
        return c ? { w: c.width, h: c.height } : null;
      });
      const cw = canvasDims?.w || box2.width;
      const ch = canvasDims?.h || box2.height;
      spot = {
        x: box2.x + (pick.x / cw) * box2.width,
        y: box2.y + (pick.y / ch) * box2.height,
      };
      log(
        `RL: Scan found ${scan.interior.length} interior + ${scan.border.length} border tiles (color=${scan.myColor.r},${scan.myColor.g},${scan.myColor.b})`,
      );
    } else {
      // Fallback to fixed offset from center
      const trueCx = box2.x + box2.width / 2;
      const trueCy = box2.y + box2.height / 2;
      if (isBorderBuilding) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 80 + Math.random() * 70;
        spot = {
          x: trueCx + Math.cos(angle) * dist,
          y: trueCy + Math.sin(angle) * dist,
        };
      } else {
        spot = {
          x: trueCx + (Math.random() - 0.5) * 80,
          y: trueCy + (Math.random() - 0.5) * 80,
        };
      }
      log("RL: Scan failed, using center fallback");
    }

    // Step 1: Move mouse to spot
    await page.mouse.move(spot.x, spot.y);
    await sleep(100);
    // Step 2: Press build key → ghost structure appears at mouse location
    await focusCanvas(page);
    await page.keyboard.press(key).catch(() => {});
    await sleep(500); // ghost needs time to initialize + check buildability
    // Step 3: Click to confirm placement
    await page.mouse.click(spot.x, spot.y);
    await sleep(200);
    // Step 4: Escape to exit build mode if placement failed
    await page.keyboard.press("Escape").catch(() => {});
    log(
      `RL: Build ${name} at (${Math.round(spot.x)},${Math.round(spot.y)}) cx=${Math.round(trueCx)} cy=${Math.round(trueCy)} [gold=${(gold / 1000).toFixed(0)}K]`,
    );
  } else if (NUKE_KEYS[actionType]) {
    const { key, name, minGold } = NUKE_KEYS[actionType];
    if (gold < minGold) {
      log(`RL: Skip ${name} — need ${(minGold / 1000).toFixed(0)}K gold`);
      return;
    }
    // Nukes: press key, then click on enemy territory (outer ring)
    await page.keyboard.press(key).catch(() => {});
    await sleep(300);
    // Click far from center — enemy territory
    const angle = Math.random() * Math.PI * 2;
    const dist = 0.15 + Math.random() * 0.1; // Beyond border ring
    const x = zone.cx + Math.cos(angle) * zone.width * dist;
    const y = zone.cy + Math.sin(angle) * zone.height * dist;
    await page.mouse.move(x, y);
    await sleep(300);
    await page.mouse.click(x, y);
    await sleep(200);
    await page.keyboard.press("Escape").catch(() => {});
    log(`RL: Launch ${name} at (${Math.round(x)},${Math.round(y)})`);
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
      totalMapTiles: 1,
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
  let lastRecenter = 0;
  lastGold = 0;
  currentScale = 1.8; // reset to game default at game start
  lastTerritoryPct = 0;
  let spawnTime = 0;

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
        const hudState = await extractGameState(page);
        const didSpawn =
          hudState && (hudState.myTroops > 0 || hudState.territoryPct > 0);

        if (didSpawn || spawnAttempts > 15) {
          spawned = true;
          spawnTime = Date.now();
          log(
            `Spawned! troops=${hudState?.myTroops || "?"}, territory=${((hudState?.territoryPct || 0) * 100).toFixed(1)}%`,
          );

          // CRITICAL: center camera on our territory and zoom in moderately
          await centerCamera(page);
          // Zoom in ~3x from default (1.8 → ~5.4) using sane deltas
          // Each -100 delta → scale *= 1.2 (20% zoom in). 6 steps ≈ 1.2^6 ≈ 3x
          for (let i = 0; i < 6; i++) await zoomStep(page, -100);
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

        // Target: territory fills ~30-50% of screen. territoryScreenFraction() tells us
        // current coverage. If too small, zoom in; if too big, zoom out.
        const screenFrac = territoryScreenFraction();
        const TARGET_FRAC = 0.35; // we want ~35% of screen to be our territory
        // ratio > 1 means territory too big on screen → zoom out
        // ratio < 1 means territory too small → zoom in
        const ratio = screenFrac / TARGET_FRAC;

        if (ratio > 1.5 || ratio < 0.5) {
          // Calculate how many 20%-zoom steps to get there
          // Each step changes scale by 1.2x. We need scale to change by 1/ratio
          // (since screenFrac ∝ scale). Steps = log(1/ratio) / log(1.2)
          const stepsNeeded = Math.round(Math.log(1 / ratio) / Math.log(1.2));
          const clamped = Math.max(-8, Math.min(8, stepsNeeded));
          if (clamped !== 0) {
            // Positive steps = zoom in (negative delta), negative = zoom out (positive delta)
            const delta = clamped > 0 ? -100 : 100;
            for (let i = 0; i < Math.abs(clamped); i++)
              await zoomStep(page, delta);
            log(
              `Zoom: scale=${currentScale.toFixed(1)}, territory screen=${(screenFrac * 100).toFixed(0)}% → ${Math.abs(clamped)} steps ${clamped > 0 ? "in" : "out"}`,
            );
          }
        }
      }

      // ── QUERY POLICY SERVER (interval matches training: TICKS_PER_STEP × TURN_INTERVAL_MS) ──
      if (Date.now() - lastPolicyQuery > POLICY_INTERVAL_MS) {
        lastPolicyQuery = Date.now();

        const gameState = await extractGameState(page);
        if (gameState) {
          lastGold = gameState.myGold || 0;
          lastTerritoryPct = gameState.territoryPct || 0;
          const g = lastGold;
          const hasN = gameState.neighbors.length > 0;
          const hasT = gameState.myTroops > 10;
          // Build action mask based on gold thresholds (matches env_server.ts logic)
          gameState.actionMask = [
            true, // 0: NOOP
            hasN && hasT, // 1: ATTACK
            false, // 2: BOAT_ATTACK (no port detection in HUD)
            false, // 3: RETREAT (no outgoing attack detection)
            g >= 125_000, // 4: BUILD_CITY
            g >= 125_000, // 5: BUILD_FACTORY
            g >= 50_000, // 6: BUILD_DEFENSE
            g >= 125_000, // 7: BUILD_PORT
            g >= 1_500_000, // 8: BUILD_SAM
            g >= 1_000_000, // 9: BUILD_SILO
            g >= 250_000, // 10: BUILD_WARSHIP
            g >= 750_000, // 11: LAUNCH_ATOM
            g >= 5_000_000, // 12: LAUNCH_HBOMB
            g >= 10_000_000, // 13: LAUNCH_MIRV
            false, // 14: MOVE_WARSHIP
            g >= 50_000, // 15: UPGRADE
            false, // 16: DELETE_UNIT
          ];
          const action = await queryPolicy(gameState);
          if (action) {
            currentAction = action;
            if (tick % 10 === 0) {
              log(
                `RL action: type=${action.actionType} target=${action.targetIdx} troops=${action.troopFraction} gold=${(lastGold / 1000).toFixed(0)}K`,
              );
            }
          }
        }
      }

      // ── EXPAND: click borders to grow ──
      // Use explicit mousedown+mouseup at same position with gap between clicks
      // to avoid rapid clicks being interpreted as drags (which pan the camera)
      for (let i = 0; i < 3; i++) {
        const { x, y } = randomBorder(zone);
        await page.mouse.move(x, y);
        await sleep(20);
        await page.mouse.down();
        await sleep(20);
        await page.mouse.up();
        await sleep(60);
      }

      // ── EXECUTE RL POLICY ACTION ──
      await executeRLAction(page, currentAction, zone, lastGold);

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
        const gs = await extractGameState(page);
        const pct = gs ? (gs.territoryPct * 100).toFixed(1) : "?";
        const troops = gs ? gs.myTroops : "?";
        log(
          `[${elapsed.toFixed(0)}s] ${pct}% territory, troops=${troops}, gold=${(lastGold / 1000).toFixed(0)}K, scale=${currentScale.toFixed(1)}, screenFrac=${(territoryScreenFraction() * 100).toFixed(0)}%, action=${currentAction?.actionType ?? "none"}`,
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
