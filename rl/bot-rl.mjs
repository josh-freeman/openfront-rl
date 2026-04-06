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
const GAME_URL = process.argv[4] || "https://openfront.io";
const HEADLESS =
  process.env.HEADLESS === "1" || process.env.HEADLESS === "true";

// Training: ticksPerStep=10, turnInterval=100ms → 1 action per 1000ms game time.
// Live server ticks faster, so poll more frequently to stay responsive.
const TICKS_PER_STEP = 10;
const TURN_INTERVAL_MS = 100;
const POLICY_INTERVAL_MS = 500; // Query policy every 500ms for responsiveness

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

// ── Debug overlay: show click markers on screen ──────────────────
async function showDebugMarker(
  page,
  x,
  y,
  originX,
  originY,
  label,
  color = "red",
) {
  await page.evaluate(
    (x, y, ox, oy, label, color) => {
      let overlay = document.getElementById("rl-debug-overlay");
      if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "rl-debug-overlay";
        overlay.style.cssText =
          "position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:99999";
        document.body.appendChild(overlay);
      }
      const dx = x - ox,
        dy = y - oy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
      const line = document.createElement("div");
      line.style.cssText = `position:absolute;left:${ox}px;top:${oy}px;width:${dist}px;height:2px;background:${color};opacity:0.6;transform-origin:0 50%;transform:rotate(${angle}deg);pointer-events:none;`;
      const dot = document.createElement("div");
      dot.style.cssText = `position:absolute;left:${x - 8}px;top:${y - 8}px;width:16px;height:16px;border-radius:50%;background:${color};opacity:0.8;pointer-events:none;border:2px solid white;`;
      const lbl = document.createElement("div");
      lbl.textContent = label;
      lbl.style.cssText = `position:absolute;left:${x + 12}px;top:${y - 10}px;color:white;font-size:11px;font-weight:bold;text-shadow:1px 1px 2px black;pointer-events:none;white-space:nowrap;`;
      const orig = document.createElement("div");
      orig.style.cssText = `position:absolute;left:${ox - 4}px;top:${oy - 4}px;width:8px;height:8px;border-radius:50%;background:cyan;opacity:0.8;pointer-events:none;`;
      overlay.appendChild(line);
      overlay.appendChild(dot);
      overlay.appendChild(lbl);
      overlay.appendChild(orig);
      setTimeout(() => {
        [line, dot, lbl, orig].forEach((el) => {
          el.style.transition = "opacity 0.5s";
          el.style.opacity = "0";
          setTimeout(() => el.remove(), 500);
        });
      }, 2000);
    },
    x,
    y,
    originX,
    originY,
    label,
    color,
  );
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
      const gameHost = new URL(GAME_URL).host;
      if (p.url().includes(gameHost)) return p;
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
  log(`Going to ${GAME_URL}...`);
  await page.goto(GAME_URL, {
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

  // Helper: find an element by exact text across shadow DOMs
  const findByText = (texts) => {
    const allEls = [...document.querySelectorAll("*")];
    document.querySelectorAll("*").forEach((el) => {
      if (el.shadowRoot) {
        for (const c of el.shadowRoot.querySelectorAll("*")) allEls.push(c);
      }
    });
    for (const el of allEls) {
      const text = el.childNodes.length <= 3 ? el.textContent?.trim() : "";
      if (texts.includes(text)) {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0)
          return { x: r.x + r.width / 2, y: r.y + r.height / 2, text };
      }
    }
    return null;
  };

  // Try clicking the featured game card to select it (not required, but keeps a FFA map selected)
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
    log("Found game card, clicking to select...");
    await page.mouse.click(cardClicked.x, cardClicked.y);
    await sleep(1000);
  } else {
    log("No card found, clicking featured area...");
    const vp = await safeEval(page, () => ({
      w: window.innerWidth,
      h: window.innerHeight,
    }));
    if (vp) await page.mouse.click(vp.w * 0.3, vp.h * 0.72);
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
    async (botName) => {
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

      // Fix 4: Read unit counts from <unit-display> Lit properties,
      // falling back to DOM icon-adjacent span parsing
      const unitDisplay = document.querySelector("unit-display");
      if (unitDisplay) {
        let cities = 0,
          factories = 0,
          ports = 0,
          defenses = 0,
          silos = 0,
          sams = 0,
          warships = 0;

        // Try Lit properties first (private fields accessible at runtime)
        if (typeof unitDisplay._cities === "number") {
          cities = unitDisplay._cities;
          factories = unitDisplay._factories || 0;
          ports = unitDisplay._port || 0;
          defenses = unitDisplay._defensePost || 0;
          silos = unitDisplay._missileSilo || 0;
          sams = unitDisplay._samLauncher || 0;
          warships = unitDisplay._warships || 0;
        } else {
          // Fallback: match count spans by adjacent icon alt/title/src
          const iconMap = {
            CityIcon: "cities",
            FactoryIcon: "factories",
            PortIcon: "ports",
            ShieldIcon: "defenses",
            MissileSiloIcon: "silos",
            SamLauncherIcon: "sams",
            BattleshipIcon: "warships",
          };
          const counts = {
            cities: 0,
            factories: 0,
            ports: 0,
            defenses: 0,
            silos: 0,
            sams: 0,
            warships: 0,
          };
          for (const img of unitDisplay.querySelectorAll("img")) {
            const src = img.src || img.getAttribute("src") || "";
            for (const [iconKey, countKey] of Object.entries(iconMap)) {
              if (src.includes(iconKey)) {
                // The count is in the closest button/container's text
                const btn = img.closest("button") || img.parentElement;
                if (btn) {
                  const text = btn.textContent?.trim() || "";
                  const num = parseInt(text.replace(/\D/g, ""), 10);
                  if (!isNaN(num)) counts[countKey] = num;
                }
                break;
              }
            }
          }
          cities = counts.cities;
          factories = counts.factories;
          ports = counts.ports;
          defenses = counts.defenses;
          silos = counts.silos;
          sams = counts.sams;
          warships = counts.warships;
        }

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
        const totalUnits =
          cities + factories + ports + defenses + silos + sams + warships;
        state.units = new Array(totalUnits).fill("unit");
        state.hasUnits = totalUnits > 0;
      }

      // Read delete cooldown from game engine (PlayerView.deleteUnitCooldown())
      // Returns remaining seconds; 0 means can delete now
      if (sidebar && sidebar.game) {
        const me = sidebar.game.myPlayer?.();
        if (me && typeof me.deleteUnitCooldown === "function") {
          state.canDeleteUnit = me.deleteUnitCooldown() === 0;
        }
        // Query real build availability from game API (accounts for scaling costs)
        if (me && typeof me.buildables === "function") {
          try {
            const bu = await me.buildables();
            // Map UnitType string -> canBuild boolean
            state.canBuildFromAPI = {};
            state._buildCosts = {};
            for (const entry of bu) {
              state.canBuildFromAPI[entry.type] = entry.canBuild !== false;
              state._buildCosts[entry.type] = Number(entry.cost);
            }
            state._buildDebug = bu.map((e) => ({
              type: e.type,
              canBuild: e.canBuild !== false,
              cost: Number(e.cost),
            }));
          } catch (e) {
            state._buildError = e.message;
          }
        }
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

        // Fix 2: Extract attacker names from rendered attack rows
        // The incoming attack rows render attacker names as visible text with
        // class "truncate" inside red-colored buttons. Read directly from DOM.
        const attackerNames = [];
        const attackRows = attacksDisplay.querySelectorAll(
          ".text-red-400 .truncate",
        );
        for (const el of attackRows) {
          const name = el.textContent?.trim();
          if (name) attackerNames.push(name);
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

      // Read leaderboard data from <leader-board> Lit component's players array.
      // Each entry has { name, score, player } where player is a PlayerView
      // with numTilesOwned() and troops() methods giving exact values.
      const lbData = {}; // name -> { tiles, troops }
      const lb2 = document.querySelector("leader-board");
      if (lb2 && lb2.players) {
        for (const entry of lb2.players) {
          const name = entry.name;
          if (!name) continue;
          const pv = entry.player;
          if (pv) {
            // Direct access to PlayerView — exact values, no parsing
            const tiles =
              typeof pv.numTilesOwned === "function"
                ? pv.numTilesOwned()
                : pv.numTilesOwned || 0;
            const troops =
              typeof pv.troops === "function" ? pv.troops() : pv.troops || 0;
            lbData[name] = { tiles, troops };
          } else {
            // Fallback: parse score string "X.X%" for tiles
            const pctMatch = entry.score?.match(/([\d.]+)%/);
            const pct = pctMatch ? parseFloat(pctMatch[1]) / 100 : 0;
            lbData[name] = {
              tiles: Math.round(pct * state.totalMapTiles),
              troops: 0,
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

      // Use game API for accurate land neighbor detection
      const landNeighborNames = new Set();
      const apiNeighborData = new Map(); // displayName -> {tiles, troops, relation, isLand, isAllied}
      if (sidebar && sidebar.game) {
        const me2 = sidebar.game.myPlayer?.();
        const g = sidebar.game;
        if (me2) {
          try {
            // Client-side sharesBorderWith: scan our border tiles, collect ownerIDs
            // of adjacent tiles. This replicates PlayerImpl.sharesBorderWith which
            // doesn't exist on the client-side PlayerView.
            let myBorder = [];
            try {
              if (typeof me2.borderTiles === "function") {
                const bt = await me2.borderTiles();
                myBorder = bt?.borderTiles ? [...bt.borderTiles] : [];
              }
            } catch (e) {}
            const borderNeighborIDs = new Set();
            const mySmallID =
              typeof me2.smallID === "function" ? me2.smallID() : -1;
            // Wilderness CC detection: fringe-only scan (1-layer from our border)
            const wildernessCCs = []; // {id, tileCount, ourBorderCoords, centroidX, centroidY}
            let wildCCIndex = 0;
            const hasIsLand = typeof g.isLand === "function";
            let hasCoast = false;

            if (
              myBorder.length > 0 &&
              typeof g.neighbors === "function" &&
              typeof g.ownerID === "function"
            ) {
              const step = Math.max(1, Math.floor(myBorder.length / 300));
              // Step 1: find all unclaimed fringe tiles + track border associations
              const fringeTiles = new Set();
              const fringeToOurBorder = new Map(); // fringe tile → [border tiles]
              for (let i = 0; i < myBorder.length; i += step) {
                const adjTiles = g.neighbors(myBorder[i]);
                for (const adj of adjTiles) {
                  const oid = g.ownerID(adj);
                  if (oid !== 0 && oid !== mySmallID) {
                    borderNeighborIDs.add(oid);
                  }
                  // Check for coast: border tile adjacent to water
                  if (hasIsLand && !g.isLand(adj)) {
                    hasCoast = true;
                  }
                  if (oid === 0 && (!hasIsLand || g.isLand(adj))) {
                    fringeTiles.add(adj);
                    if (!fringeToOurBorder.has(adj))
                      fringeToOurBorder.set(adj, []);
                    fringeToOurBorder.get(adj).push(myBorder[i]);
                  }
                }
              }
              // Step 2: group fringe tiles into CCs (BFS among fringe only)
              const visited = new Set();
              const SIZE_EST_DEPTH = 3;
              for (const seed of fringeTiles) {
                if (visited.has(seed)) continue;
                visited.add(seed);
                const ccFringe = [seed];
                const queue = [seed];
                const ourBorderSet = new Set();
                let sx = 0,
                  sy = 0;
                while (queue.length > 0) {
                  const curr = queue.pop();
                  sx += g.x(curr);
                  sy += g.y(curr);
                  for (const bt of fringeToOurBorder.get(curr) || []) {
                    ourBorderSet.add(bt);
                  }
                  for (const n of g.neighbors(curr)) {
                    if (!visited.has(n) && fringeTiles.has(n)) {
                      visited.add(n);
                      ccFringe.push(n);
                      queue.push(n);
                    }
                  }
                }
                // Expand outward by SIZE_EST_DEPTH layers for size estimate
                const sizeVisited = new Set(ccFringe);
                let frontier = ccFringe;
                for (let d = 0; d < SIZE_EST_DEPTH; d++) {
                  const nextFrontier = [];
                  for (const t of frontier) {
                    for (const n of g.neighbors(t)) {
                      if (
                        !sizeVisited.has(n) &&
                        g.ownerID(n) === 0 &&
                        (!hasIsLand || g.isLand(n))
                      ) {
                        sizeVisited.add(n);
                        nextFrontier.push(n);
                      }
                    }
                  }
                  frontier = nextFrontier;
                }
                // Store border tile coords (not refs, which don't survive serialization)
                const ourBorderCoords = Array.from(ourBorderSet).map((t) => ({
                  x: g.x(t),
                  y: g.y(t),
                }));
                // Skip tiny wilderness CCs (< 20 tiles estimated) — not worth attacking
                if (sizeVisited.size < 20) continue;
                wildernessCCs.push({
                  id: `wilderness_${wildCCIndex++}`,
                  tileCount: sizeVisited.size,
                  ourBorderCoords,
                  centroidX: sx / ccFringe.length,
                  centroidY: sy / ccFringe.length,
                });
              }
            }

            // Get all players from game API (not just leaderboard top 10)
            const allPlayers =
              typeof g.players === "function" ? g.players() : [];
            for (const n of allPlayers) {
              if (!n || (typeof n.isAlive === "function" && !n.isAlive()))
                continue;
              const nSmallID =
                typeof n.smallID === "function" ? n.smallID() : -1;
              if (nSmallID === mySmallID || nSmallID <= 0) continue;
              const dname =
                typeof n.displayName === "function" ? n.displayName() : "";
              if (!dname) continue;
              const isLand = borderNeighborIDs.has(nSmallID);
              const rel =
                typeof me2.relation === "function" ? me2.relation(n) : 2;
              const tiles =
                typeof n.numTilesOwned === "function" ? n.numTilesOwned() : 0;
              const troops = typeof n.troops === "function" ? n.troops() : 0;
              const allied =
                typeof me2.isAlliedWith === "function"
                  ? me2.isAlliedWith(n)
                  : false;
              if (isLand) landNeighborNames.add(dname);
              apiNeighborData.set(dname, {
                tiles,
                troops,
                relation: rel,
                isLand,
                isAllied: allied,
              });
            }
            state.hasCoast = hasCoast;
            state._borderDebug = {
              myBorderLen: myBorder.length,
              mySmallID,
              hasCoast,
              hasNeighborsFn: typeof g.neighbors === "function",
              hasOwnerIDFn: typeof g.ownerID === "function",
              hasPlayersFn: typeof g.players === "function",
              numPlayers: allPlayers.length,
              borderNeighborIDs: [...borderNeighborIDs],
              landNames: [...landNeighborNames],
              apiNeighborCount: apiNeighborData.size,
              wildernessCCCount: wildernessCCs.length,
            };
            state._wildernessCCs = wildernessCCs;
          } catch (e) {
            state._borderDebugError = e?.message || String(e);
          }
        }
        // Also read our own exact stats from game API
        if (me2) {
          try {
            const myTiles =
              typeof me2.numTilesOwned === "function" ? me2.numTilesOwned() : 0;
            const myTroops =
              typeof me2.troops === "function" ? me2.troops() : 0;
            const myGold =
              typeof me2.gold === "function" ? Number(me2.gold()) : 0;
            if (myTiles > 0) state.myTiles = myTiles;
            if (myTroops > 0) state.myTroops = myTroops;
            if (myGold > 0) state.myGold = myGold;
          } catch (e) {}
        }
      }

      // Capture our own label position for attack targeting
      for (const [name, info] of playerMap) {
        if (name === botName || name.includes(botName.slice(0, 6))) {
          if (info.labelX && info.labelY) {
            state.myLabelX = info.labelX;
            state.myLabelY = info.labelY;
          }
          break;
        }
      }

      // Build neighbors list from DOM labels, enriched with leaderboard data
      const neighbors = [];
      for (const [name, info] of playerMap) {
        if (name === botName || name.includes(botName.slice(0, 6))) continue;
        if (name === "") continue;
        if (name.toLowerCase() === "wilderness") continue; // skip DOM wilderness label — we use CCs instead

        const lb = lbData[name] || {};
        // Use exact values from leaderboard PlayerView when available,
        // fall back to NameLayer DOM values (only top ~5 players are on leaderboard)
        const troops = lb.troops || info.troops || 0;
        let tiles = lb.tiles || 0;
        if (tiles === 0 && troops > 0) {
          // Rough estimate: tiles ≈ troops * 0.02 (troops are raw values)
          tiles = Math.round(troops * 0.02);
        }

        // Use game API data when available (exact values), fall back to DOM/leaderboard
        const api = apiNeighborData.get(name);
        neighbors.push({
          id: name,
          name: name,
          tiles: api?.tiles || tiles,
          troops: api?.troops || troops,
          relation: api?.relation ?? 2, // Game API relation (0-3), default Neutral
          alive: true,
          isLandNeighbor: landNeighborNames.has(name),
          isAllied: api?.isAllied || false,
          visible: info.visible,
          labelX: info.labelX,
          labelY: info.labelY,
        });
      }

      // Add wilderness CCs as neighbors (computed from game API)
      if (state._wildernessCCs) {
        for (const wcc of state._wildernessCCs) {
          neighbors.push({
            id: wcc.id,
            name: wcc.id,
            tiles: wcc.tileCount,
            troops: 0,
            relation: 0,
            alive: true,
            isLandNeighbor: true,
            isAllied: false,
            visible: true,
            labelX: null,
            labelY: null,
            _wildernessCC: wcc, // stash for attack targeting
          });
        }
      }

      // Filter out dead players — leaderboard only shows alive players,
      // so use that as the alive set. Also keep anyone visible with troops > 0
      // (leaderboard may lag by a tick). Wilderness CCs always pass.
      const aliveNames = new Set(Object.keys(lbData));
      const filteredNeighbors = neighbors.filter(
        (n) =>
          n.id.startsWith("wilderness_") ||
          aliveNames.has(n.name) ||
          n.troops > 0,
      );

      // Sort to match training: land neighbors first, then by territory size
      state.neighbors = filteredNeighbors
        .sort((a, b) => {
          if (a.isLandNeighbor !== b.isLandNeighbor)
            return a.isLandNeighbor ? -1 : 1;
          return b.tiles - a.tiles;
        })
        .slice(0, 16);

      // Compute territory % from exact tile count (game API) or fall back to leaderboard
      if (state.myTiles > 0 && state.totalMapTiles > 0) {
        state.territoryPct = state.myTiles / state.totalMapTiles;
      } else {
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
        state.myTiles = Math.round(state.territoryPct * state.totalMapTiles);
      }

      // Read unit screen positions for upgrade/delete targeting
      // Uses game.myPlayer().units() → tile positions → transform to screen coords
      state.unitPositions = [];
      try {
        const buildMenu = document.querySelector("build-menu");
        const th = buildMenu?.transformHandler;
        const g = sidebar?.game;
        if (th && g) {
          const me = g.myPlayer?.();
          if (me) {
            const allUnits = typeof me.units === "function" ? me.units() : [];
            for (const u of allUnits) {
              const tile = typeof u.tile === "function" ? u.tile() : u.tile;
              if (tile == null) continue;
              const gx = typeof g.x === "function" ? g.x(tile) : 0;
              const gy = typeof g.y === "function" ? g.y(tile) : 0;
              if (!gx && !gy) continue;
              const screen = th.worldToScreenCoordinates({ x: gx, y: gy });
              if (screen && screen.x > 0 && screen.y > 0) {
                const typeName =
                  typeof u.type === "function" ? u.type() : u.type;
                const active =
                  typeof u.isActive === "function" ? u.isActive() : true;
                state.unitPositions.push({
                  x: screen.x,
                  y: screen.y,
                  type: typeName,
                  active,
                });
              }
            }
          }
        }
      } catch (e) {
        // Silently fail — unit positions are optional
      }

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
    top: box.y + box.height * 0.2, // clear of top banner/player info bar
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

// Center camera on our territory using arrow keys + game API.
// Computes our territory centroid in screen space, then pans to bring it to center.
// Falls back to 'c' key if game API isn't available.
async function centerCamera(page) {
  // PRIMARY: Use the game's built-in center-on-player via 'c' key
  await focusCanvas(page);
  await page.keyboard.press("c").catch(() => {});
  await sleep(800); // wait for the centering animation to finish

  // Sync actual game scale
  const scaleInfo = await safeEval(page, () => {
    const bm = document.querySelector("build-menu");
    return bm?.transformHandler?.scale ?? null;
  });
  if (scaleInfo) currentScale = scaleInfo;
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

// Zoom in to a minimum scale for precise clicking. Returns steps taken (for restoreClickZoom).
const MIN_CLICK_SCALE = 7;
async function ensureClickZoom(page) {
  const realScale = await safeEval(page, () => {
    const bm = document.querySelector("build-menu");
    return bm?.transformHandler?.scale ?? null;
  });
  if (realScale) currentScale = realScale;
  if (currentScale >= MIN_CLICK_SCALE) return 0;
  const stepsIn = Math.min(
    8,
    Math.round(Math.log(MIN_CLICK_SCALE / currentScale) / Math.log(1.2)),
  );
  for (let i = 0; i < stepsIn; i++) await zoomStep(page, -100);
  await sleep(100);
  return stepsIn;
}

async function restoreClickZoom(page, stepsIn) {
  if (stepsIn > 0) {
    for (let i = 0; i < stepsIn; i++) await zoomStep(page, 100);
  }
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

// ── Raycast helpers ──────────────────────────────────────────────
// Walk from screen center outward along a direction until we leave our
// territory color. Returns the first non-our-territory pixel as page
// coordinates. Works at ANY zoom level because it reads actual pixels.

async function raycastToBorder(page, angle) {
  return safeEval(
    page,
    (angle) => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return null;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      const W = canvas.width,
        H = canvas.height;
      const cx = Math.floor(W / 2),
        cy = Math.floor(H / 2);

      // Center pixel = our territory color (camera was centered on us)
      const cd = ctx.getImageData(cx, cy, 1, 1).data;
      const myR = cd[0],
        myG = cd[1],
        myB = cd[2];
      if (myR + myG + myB < 60) return null;
      if (myB > myR + myG + 50) return null;

      const imgData = ctx.getImageData(0, 0, W, H).data;
      const step = 5;
      const maxDist = Math.max(W, H);
      const cosA = Math.cos(angle),
        sinA = Math.sin(angle);

      for (let i = 3; i * step < maxDist; i++) {
        const px = Math.floor(cx + cosA * i * step);
        const py = Math.floor(cy + sinA * i * step);
        if (px < 0 || px >= W || py < 0 || py >= H) break;

        const idx = (py * W + px) * 4;
        const dr = imgData[idx] - myR;
        const dg = imgData[idx + 1] - myG;
        const db = imgData[idx + 2] - myB;

        if (Math.sqrt(dr * dr + dg * dg + db * db) > 45) {
          // Found a non-our-territory pixel — step a bit further past the border
          const landPx = Math.floor(cx + cosA * (i + 2) * step);
          const landPy = Math.floor(cy + sinA * (i + 2) * step);
          const clampedPx = Math.max(0, Math.min(W - 1, landPx));
          const clampedPy = Math.max(0, Math.min(H - 1, landPy));

          const rect = canvas.getBoundingClientRect();
          return {
            x: rect.left + (clampedPx / W) * rect.width,
            y: rect.top + (clampedPy / H) * rect.height,
            stepsToEdge: i,
            angle: angle,
          };
        }
      }
      return null; // Our territory extends to screen edge in this direction
    },
    angle,
  );
}

// Cast a fan of rays around centerAngle, return the hit with fewest steps
// (closest border point roughly in the target's direction).
async function raycastFan(page, centerAngle, fanDeg = 40, numRays = 9) {
  return safeEval(
    page,
    (centerAngle, fanRad, numRays) => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return null;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      const W = canvas.width,
        H = canvas.height;
      const cx = Math.floor(W / 2),
        cy = Math.floor(H / 2);

      const cd = ctx.getImageData(cx, cy, 1, 1).data;
      const myR = cd[0],
        myG = cd[1],
        myB = cd[2];
      if (myR + myG + myB < 60) return null;
      if (myB > myR + myG + 50) return null;

      const imgData = ctx.getImageData(0, 0, W, H).data;
      const step = 5;
      const maxDist = Math.max(W, H);
      const rect = canvas.getBoundingClientRect();

      let bestHit = null;
      let bestSteps = Infinity;

      for (let r = 0; r < numRays; r++) {
        const angle = centerAngle + (r / (numRays - 1) - 0.5) * fanRad;
        const cosA = Math.cos(angle),
          sinA = Math.sin(angle);

        for (let i = 3; i * step < maxDist; i++) {
          const px = Math.floor(cx + cosA * i * step);
          const py = Math.floor(cy + sinA * i * step);
          if (px < 0 || px >= W || py < 0 || py >= H) break;

          const idx = (py * W + px) * 4;
          const dr = imgData[idx] - myR;
          const dg = imgData[idx + 1] - myG;
          const db = imgData[idx + 2] - myB;

          if (Math.sqrt(dr * dr + dg * dg + db * db) > 45) {
            if (i < bestSteps) {
              const landPx = Math.floor(cx + cosA * (i + 2) * step);
              const landPy = Math.floor(cy + sinA * (i + 2) * step);
              const clampedPx = Math.max(0, Math.min(W - 1, landPx));
              const clampedPy = Math.max(0, Math.min(H - 1, landPy));
              bestSteps = i;
              bestHit = {
                x: rect.left + (clampedPx / W) * rect.width,
                y: rect.top + (clampedPy / H) * rect.height,
                stepsToEdge: i,
                angle: angle,
              };
            }
            break;
          }
        }
      }
      return bestHit;
    },
    centerAngle,
    (fanDeg * Math.PI) / 180,
    numRays,
  );
}

// ── Pan helpers ──────────────────────────────────────────────────
// Hold arrow keys to pan camera in a direction. The game's InputHandler
// pans every frame while keys are held, so holding for ~400ms moves
// roughly half a screen at typical zoom.

async function panInDirection(page, angle, durationMs = 400) {
  await focusCanvas(page);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  // Map angle to arrow keys (can press two for diagonal)
  const keys = [];
  if (cos > 0.4) keys.push("ArrowRight");
  if (cos < -0.4) keys.push("ArrowLeft");
  if (sin > 0.4) keys.push("ArrowDown");
  if (sin < -0.4) keys.push("ArrowUp");
  if (keys.length === 0) keys.push("ArrowRight"); // fallback

  for (const k of keys) await page.keyboard.down(k);
  await sleep(durationMs);
  for (const k of keys) await page.keyboard.up(k);
  await sleep(150);
}

// After panning, screen center might not be our territory anymore.
// This scanner doesn't assume center = ours. Instead it takes our known
// territory color (from before the pan) and scans for the first non-matching,
// non-water pixel — that's a clickable enemy/wilderness tile.
// If center IS still ours, it raycasts outward like before.
// If center is NOT ours, it clicks center directly (we panned onto enemy territory).
async function findClickAfterPan(page, myColorRGB, targetAngle) {
  return safeEval(
    page,
    (myR, myG, myB, targetAngle) => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return null;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      const W = canvas.width,
        H = canvas.height;
      const cx = Math.floor(W / 2),
        cy = Math.floor(H / 2);
      const rect = canvas.getBoundingClientRect();

      // Check if center is still our territory
      const cd = ctx.getImageData(cx, cy, 1, 1).data;
      const dr = cd[0] - myR,
        dg = cd[1] - myG,
        db = cd[2] - myB;
      const centerIsOurs = Math.sqrt(dr * dr + dg * dg + db * db) < 45;

      // Skip water check
      const isWater = (r, g, b) => r + g + b < 60 || b > r + g + 50;

      if (!centerIsOurs && !isWater(cd[0], cd[1], cd[2])) {
        // We panned past our border — center is enemy/wilderness. Click here!
        return {
          x: rect.left + (cx / W) * rect.width,
          y: rect.top + (cy / H) * rect.height,
          stepsToEdge: 0,
          method: "pan-center-is-enemy",
        };
      }

      if (!centerIsOurs) {
        // Center is water after pan — scan radially for any land
        const imgData = ctx.getImageData(0, 0, W, H).data;
        const step = 8;
        for (let deg = 0; deg < 360; deg += 30) {
          const a = (deg * Math.PI) / 180;
          const cosA = Math.cos(a),
            sinA = Math.sin(a);
          for (let i = 5; i * step < Math.max(W, H) / 2; i++) {
            const px = Math.floor(cx + cosA * i * step);
            const py = Math.floor(cy + sinA * i * step);
            if (px < 0 || px >= W || py < 0 || py >= H) break;
            const idx = (py * W + px) * 4;
            const r = imgData[idx],
              g = imgData[idx + 1],
              b = imgData[idx + 2];
            if (!isWater(r, g, b)) {
              // Check it's not our territory
              const ddr = r - myR,
                ddg = g - myG,
                ddb = b - myB;
              if (Math.sqrt(ddr * ddr + ddg * ddg + ddb * ddb) > 45) {
                return {
                  x: rect.left + (px / W) * rect.width,
                  y: rect.top + (py / H) * rect.height,
                  stepsToEdge: i,
                  method: "pan-scan-land",
                };
              }
            }
          }
        }
        return null;
      }

      // Center is still ours — raycast outward in a fan toward target
      const imgData = ctx.getImageData(0, 0, W, H).data;
      const step = 5;
      const fanRad = (80 * Math.PI) / 180; // wider fan after pan
      const numRays = 13;
      let bestHit = null;
      let bestSteps = Infinity;

      for (let r = 0; r < numRays; r++) {
        const angle = targetAngle + (r / (numRays - 1) - 0.5) * fanRad;
        const cosA = Math.cos(angle),
          sinA = Math.sin(angle);
        for (let i = 3; i * step < Math.max(W, H); i++) {
          const px = Math.floor(cx + cosA * i * step);
          const py = Math.floor(cy + sinA * i * step);
          if (px < 0 || px >= W || py < 0 || py >= H) break;
          const idx = (py * W + px) * 4;
          const ddr = imgData[idx] - myR;
          const ddg = imgData[idx + 1] - myG;
          const ddb = imgData[idx + 2] - myB;
          if (Math.sqrt(ddr * ddr + ddg * ddg + ddb * ddb) > 45) {
            if (i < bestSteps) {
              const landPx = Math.floor(cx + cosA * (i + 2) * step);
              const landPy = Math.floor(cy + sinA * (i + 2) * step);
              bestSteps = i;
              bestHit = {
                x:
                  rect.left +
                  (Math.max(0, Math.min(W - 1, landPx)) / W) * rect.width,
                y:
                  rect.top +
                  (Math.max(0, Math.min(H - 1, landPy)) / H) * rect.height,
                stepsToEdge: i,
                method: "pan-raycast",
              };
            }
            break;
          }
        }
      }
      return bestHit;
    },
    myColorRGB?.r ?? 128,
    myColorRGB?.g ?? 128,
    myColorRGB?.b ?? 128,
    targetAngle,
  );
}

// Read our territory color from screen center (call while centered on our territory)
async function readOurColor(page) {
  return safeEval(page, () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const W = canvas.width,
      H = canvas.height;
    const cd = ctx.getImageData(
      Math.floor(W / 2),
      Math.floor(H / 2),
      1,
      1,
    ).data;
    if (cd[0] + cd[1] + cd[2] < 60) return null;
    if (cd[2] > cd[0] + cd[1] + 50) return null;
    return { r: cd[0], g: cd[1], b: cd[2] };
  });
}

// ── Game-API-based targeting ─────────────────────────────────────
// These functions use the game engine's internal state to compute exact
// screen coordinates for click targets. No pixel scanning needed.

// Compute the screen position for attacking a target player.
// Uses borderTiles() to find the closest enemy tile to our border,
// then worldToScreenCoordinates to get exact screen position.
// Returns { x, y, onScreen, method } or null if game API unavailable.
async function getAttackClickPos(page, targetName) {
  return safeEval(
    page,
    async (targetName) => {
      try {
        const sidebar = document.querySelector("game-right-sidebar");
        if (!sidebar?.game) return null;
        const g = sidebar.game;
        const buildMenu = document.querySelector("build-menu");
        const th = buildMenu?.transformHandler;
        if (!th) return null;
        const me = g.myPlayer?.();
        if (!me) return null;

        // Find target player from leaderboard or game API neighbors
        let target = null;

        // Try leaderboard first (has PlayerView objects)
        const lb = document.querySelector("leader-board");
        if (lb?.players) {
          for (const entry of lb.players) {
            if (entry.name === targetName && entry.player) {
              target = entry.player;
              break;
            }
          }
        }

        // Fallback: search all game players (works for ALL players, not just top 10)
        if (!target && typeof g.players === "function") {
          try {
            for (const n of g.players()) {
              const dn =
                typeof n.displayName === "function" ? n.displayName() : "";
              if (dn === targetName) {
                target = n;
                break;
              }
            }
          } catch (e) {}
        }

        if (!target) return null;

        // Get border tiles for both players (async on client)
        let ourBorder = [];
        try {
          if (typeof me.borderTiles === "function") {
            ourBorder = [...(await me.borderTiles()).borderTiles];
          }
        } catch (e) {
          return null;
        }

        let targetBorder = [];
        try {
          if (typeof target.borderTiles === "function") {
            targetBorder = [...(await target.borderTiles()).borderTiles];
          }
        } catch (e) {}

        const vw = window.innerWidth,
          vh = window.innerHeight;

        if (targetBorder.length > 0 && ourBorder.length > 0) {
          // Find closest pair of border tiles (sample for perf)
          const s1 = Math.max(1, Math.floor(ourBorder.length / 80));
          const s2 = Math.max(1, Math.floor(targetBorder.length / 80));
          let bestDist = Infinity,
            bestTile = null;

          for (let i = 0; i < ourBorder.length; i += s1) {
            const ox = g.x(ourBorder[i]),
              oy = g.y(ourBorder[i]);
            for (let j = 0; j < targetBorder.length; j += s2) {
              const tx = g.x(targetBorder[j]),
                ty = g.y(targetBorder[j]);
              const d = Math.abs(ox - tx) + Math.abs(oy - ty);
              if (d < bestDist) {
                bestDist = d;
                bestTile = targetBorder[j];
              }
            }
          }

          if (bestTile != null) {
            // Also find the closest OUR border tile to this target border tile
            let bestOurs = ourBorder[0];
            let bestOursDist = Infinity;
            const tx = g.x(bestTile),
              ty = g.y(bestTile);
            for (let i = 0; i < ourBorder.length; i += s1) {
              const d =
                Math.abs(g.x(ourBorder[i]) - tx) +
                Math.abs(g.y(ourBorder[i]) - ty);
              if (d < bestOursDist) {
                bestOursDist = d;
                bestOurs = ourBorder[i];
              }
            }
            // If borders are far apart (>10 tiles), target is across water — skip
            if (bestDist > 10) {
              return {
                x: 0,
                y: 0,
                onScreen: false,
                method: "too-far",
                dist: bestDist,
              };
            }
            // Walk 2 tiles deeper into target territory from their border tile
            const targetSmallID =
              typeof target.smallID === "function"
                ? target.smallID()
                : g.ownerID(bestTile);
            let deepTile = bestTile;
            for (let depth = 0; depth < 3; depth++) {
              let found = false;
              for (const adj of g.neighbors(deepTile)) {
                if (g.ownerID(adj) === targetSmallID) {
                  deepTile = adj;
                  found = true;
                  break;
                }
              }
              if (!found) break;
            }
            const deepScreen = th.worldToScreenCoordinates({
              x: g.x(deepTile),
              y: g.y(deepTile),
            });
            const ourScreen = th.worldToScreenCoordinates({
              x: g.x(bestOurs),
              y: g.y(bestOurs),
            });
            // Offset further away from our border for safety
            const dx = deepScreen.x - ourScreen.x;
            const dy = deepScreen.y - ourScreen.y;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const fx = deepScreen.x + (dx / len) * 10;
            const fy = deepScreen.y + (dy / len) * 10;
            return {
              x: fx,
              y: fy,
              onScreen: fx > 50 && fx < vw - 50 && fy > 30 && fy < vh - 100,
              method: "border-tiles",
            };
          }
        }

        // Fallback: use target's territory centroid
        if (targetBorder.length > 0) {
          let sx = 0,
            sy = 0,
            c = 0;
          const step = Math.max(1, Math.floor(targetBorder.length / 30));
          for (let i = 0; i < targetBorder.length; i += step) {
            sx += g.x(targetBorder[i]);
            sy += g.y(targetBorder[i]);
            c++;
          }
          const screen = th.worldToScreenCoordinates({ x: sx / c, y: sy / c });
          return {
            x: screen.x,
            y: screen.y,
            onScreen:
              screen.x > 50 &&
              screen.x < vw - 50 &&
              screen.y > 30 &&
              screen.y < vh - 100,
            method: "target-centroid",
          };
        }

        return null;
      } catch (e) {
        return null;
      }
    },
    targetName,
  );
}

// Compute direction from viewport center to a target's territory.
// Returns angle in radians, or null. Used for panning toward off-screen targets.
async function getDirectionToTarget(page, targetName) {
  return safeEval(
    page,
    async (targetName) => {
      try {
        const sidebar = document.querySelector("game-right-sidebar");
        if (!sidebar?.game) return null;
        const g = sidebar.game;
        const buildMenu = document.querySelector("build-menu");
        const th = buildMenu?.transformHandler;
        if (!th) return null;

        // Find target from leaderboard or game API
        let target = null;
        const lb = document.querySelector("leader-board");
        if (lb?.players) {
          for (const entry of lb.players) {
            if (entry.name === targetName && entry.player) {
              target = entry.player;
              break;
            }
          }
        }
        if (!target && typeof g.players === "function") {
          try {
            for (const n of g.players()) {
              const dn =
                typeof n.displayName === "function" ? n.displayName() : "";
              if (dn === targetName) {
                target = n;
                break;
              }
            }
          } catch (e) {}
        }
        if (!target) return null;

        let targetBorder = [];
        try {
          if (typeof target.borderTiles === "function") {
            targetBorder = [...(await target.borderTiles()).borderTiles];
          }
        } catch (e) {}
        if (targetBorder.length === 0) return null;

        // Compute target centroid screen position
        let sx = 0,
          sy = 0,
          c = 0;
        const step = Math.max(1, Math.floor(targetBorder.length / 20));
        for (let i = 0; i < targetBorder.length; i += step) {
          sx += g.x(targetBorder[i]);
          sy += g.y(targetBorder[i]);
          c++;
        }
        const screen = th.worldToScreenCoordinates({ x: sx / c, y: sy / c });
        const dx = screen.x - window.innerWidth / 2;
        const dy = screen.y - window.innerHeight / 2;
        return Math.atan2(dy, dx);
      } catch (e) {
        return null;
      }
    },
    targetName,
  );
}

// Get a good screen position for placing a building.
// Uses our own border/interior tiles + worldToScreenCoordinates.
async function getBuildClickPos(page, isBorderBuilding) {
  return safeEval(
    page,
    async (isBorderBuilding) => {
      try {
        const sidebar = document.querySelector("game-right-sidebar");
        if (!sidebar?.game) return null;
        const g = sidebar.game;
        const buildMenu = document.querySelector("build-menu");
        const th = buildMenu?.transformHandler;
        if (!th) return null;
        const me = g.myPlayer?.();
        if (!me) return null;

        let border = [];
        try {
          if (typeof me.borderTiles === "function") {
            const bt = await me.borderTiles();
            border = bt?.borderTiles ? [...bt.borderTiles] : [];
          }
        } catch (e) {}
        if (border.length === 0) return null;

        const vw = window.innerWidth,
          vh = window.innerHeight;

        // Find which of our border tiles are adjacent to enemy territory
        // by checking ownerID of neighboring tiles (no need to fetch enemy borderTiles)
        const mySmallID = typeof me.smallID === "function" ? me.smallID() : -1;
        const enemyAdjacentBorder = []; // our border tiles that touch enemy
        try {
          const step = Math.max(1, Math.floor(border.length / 200));
          for (let i = 0; i < border.length; i += step) {
            const bt = border[i];
            const adjTiles = g.neighbors(bt);
            for (const adj of adjTiles) {
              const oid = g.ownerID(adj);
              if (oid !== 0 && oid !== mySmallID) {
                enemyAdjacentBorder.push(bt);
                break;
              }
            }
          }
        } catch (e) {}

        // Get existing unit positions to avoid stacking
        const unitWorldPositions = [];
        try {
          const myUnits = typeof me.units === "function" ? me.units() : [];
          for (const u of myUnits) {
            const t = typeof u.tile === "function" ? u.tile() : null;
            if (t != null) {
              unitWorldPositions.push({ x: g.x(t), y: g.y(t) });
            }
          }
        } catch (e) {}

        // Separate border tiles into enemy-adjacent (for defenses) and safe (for cities)
        const enemySet = new Set(enemyAdjacentBorder);
        const nearEnemy = [];
        const safe = [];
        const MIN_UNIT_DIST = 8;
        const step2 = Math.max(1, Math.floor(border.length / 100));
        for (let i = 0; i < border.length; i += step2) {
          const t = border[i];
          const tx = g.x(t),
            ty = g.y(t);
          let tooClose = false;
          for (const u of unitWorldPositions) {
            if (Math.abs(tx - u.x) + Math.abs(ty - u.y) < MIN_UNIT_DIST) {
              tooClose = true;
              break;
            }
          }
          if (tooClose) continue;
          const screen = th.worldToScreenCoordinates({ x: tx, y: ty });
          const onScreen =
            screen.x > 80 &&
            screen.x < vw - 80 &&
            screen.y > 50 &&
            screen.y < vh - 120;
          if (!onScreen) continue;
          const pos = { x: screen.x, y: screen.y };
          if (enemySet.has(t)) nearEnemy.push(pos);
          else safe.push(pos);
        }

        if (isBorderBuilding) {
          // Defense/SAM/Port: place near enemy border
          const pool = nearEnemy.length > 0 ? nearEnemy : safe;
          if (pool.length === 0) return null;
          return pool[Math.floor(Math.random() * pool.length)];
        } else {
          // City/Factory/Silo: place at territory CENTER
          // Use our name label position as the centroid (game places it at territory center)
          const sidebar2 = document.querySelector("game-right-sidebar");
          const myLabel = document.querySelectorAll(".player-name-span");
          const myName =
            typeof me.displayName === "function" ? me.displayName() : "";
          for (const span of myLabel) {
            if (span.textContent?.includes(myName)) {
              const container = span.closest(".player-name")?.parentElement;
              if (!container) continue;
              const r = container.getBoundingClientRect();
              const cx = r.left + r.width / 2;
              const cy = r.top + r.height / 2;
              if (cx > 80 && cx < vw - 80 && cy > 50 && cy < vh - 120) {
                // Add small random offset to avoid stacking
                const ox = (Math.random() - 0.5) * 40;
                const oy = (Math.random() - 0.5) * 40;
                return { x: cx + ox, y: cy + oy };
              }
            }
          }
          // Fallback: average of all border tile positions (geometric center)
          if (border.length > 0) {
            let sx = 0,
              sy = 0,
              cnt = 0;
            const step3 = Math.max(1, Math.floor(border.length / 50));
            for (let i = 0; i < border.length; i += step3) {
              const screen = th.worldToScreenCoordinates({
                x: g.x(border[i]),
                y: g.y(border[i]),
              });
              sx += screen.x;
              sy += screen.y;
              cnt++;
            }
            const cx = sx / cnt,
              cy = sy / cnt;
            if (cx > 80 && cx < vw - 80 && cy > 50 && cy < vh - 120) {
              return {
                x: cx + (Math.random() - 0.5) * 30,
                y: cy + (Math.random() - 0.5) * 30,
              };
            }
          }
          // Last fallback: safe border tiles
          const pool = safe.length > 0 ? safe : nearEnemy;
          if (pool.length === 0) return null;
          return pool[Math.floor(Math.random() * pool.length)];
        }
      } catch (e) {
        return null;
      }
    },
    isBorderBuilding,
  );
}

// Check if a point is too close to an existing unit (for build placement)
function isNearUnit(x, y, unitPositions, minDist = 35) {
  for (const u of unitPositions) {
    const dx = x - u.x;
    const dy = y - u.y;
    if (dx * dx + dy * dy < minDist * minDist) return true;
  }
  return false;
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
  // Press Escape twice — first dismisses player info popup, second clears build mode
  await page.keyboard.press("Escape").catch(() => {});
  await sleep(30);
  await page.keyboard.press("Escape").catch(() => {});
  await sleep(50);
}

async function executeRLAction(
  page,
  action,
  zone,
  goldAmount,
  neighbors,
  unitPositions,
  gameState,
  cachedNeighborDirections,
) {
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
    // Always focus canvas before attack clicks
    await focusCanvas(page);
    // Adjust attack ratio using T (down 10%) / Y (up 10%) to match troopFraction
    // troopFraction: 0.2, 0.4, 0.6, 0.8, or 1.0
    const steps = Math.round((troopFraction - currentAttackRatio) / 0.1);
    if (steps !== 0) {
      const key = steps > 0 ? "y" : "t";
      for (let i = 0; i < Math.abs(steps); i++) await page.keyboard.press(key);
      currentAttackRatio = troopFraction;
      await sleep(30);
    }

    const targetIdx = action.targetIdx || 0;
    const target = neighbors[targetIdx];

    // Zoom in for wilderness attacks — tiles are tiny at normal zoom
    const atkZoomSteps =
      target && target._wildernessCC ? await ensureClickZoom(page) : 0;

    // Guard: land attacks require land neighbor, boat attacks require non-land
    if (target && !target._wildernessCC) {
      if (actionType === ACTION_ATTACK && !target.isLandNeighbor) {
        log(`RL: Skip ATTACK on ${target.name} — not a land neighbor`);
        return;
      }
    }

    const originX = gameState?.myLabelX || zone.cx;
    const originY = gameState?.myLabelY || zone.cy;

    let x, y;
    let method = "none";
    let didPan = false;

    // Wilderness targeting: click on unclaimed territory adjacent to our border
    if (target && target._wildernessCC) {
      const wcc = target._wildernessCC;
      const wildPos = await safeEval(
        page,
        async (ccData) => {
          try {
            const sidebar = document.querySelector("game-right-sidebar");
            if (!sidebar?.game) return null;
            const g = sidebar.game;
            const buildMenu = document.querySelector("build-menu");
            const th = buildMenu?.transformHandler;
            if (!th) return null;
            const me = g.myPlayer?.();
            if (!me) return null;

            // Use the CC's centroid to find wilderness tiles near this specific CC
            const vw = window.innerWidth,
              vh = window.innerHeight;
            const mySmallID =
              typeof me.smallID === "function" ? me.smallID() : -1;

            // Get our border tiles
            let ourBorder = [];
            try {
              if (typeof me.borderTiles === "function") {
                ourBorder = [...(await me.borderTiles()).borderTiles];
              }
            } catch (e) {
              return null;
            }

            // Find border tiles closest to this CC's centroid
            const ccX = ccData.centroidX,
              ccY = ccData.centroidY;
            const scored = ourBorder.map((t, idx) => ({
              t,
              idx,
              d: Math.abs(g.x(t) - ccX) + Math.abs(g.y(t) - ccY),
            }));
            scored.sort((a, b) => a.d - b.d);

            // Check the closest border tiles for adjacent wilderness
            const checkCount = Math.min(scored.length, 50);
            for (let i = 0; i < checkCount; i++) {
              const borderTile = scored[i].t;
              for (const adj of g.neighbors(borderTile)) {
                if (g.ownerID(adj) === 0 && (!g.isLand || g.isLand(adj))) {
                  // Found unclaimed tile — walk 2 tiles deeper into wilderness
                  let deepTile = adj;
                  for (let depth = 0; depth < 2; depth++) {
                    let found = false;
                    for (const adj2 of g.neighbors(deepTile)) {
                      if (
                        g.ownerID(adj2) === 0 &&
                        (!g.isLand || g.isLand(adj2))
                      ) {
                        deepTile = adj2;
                        found = true;
                        break;
                      }
                    }
                    if (!found) break;
                  }
                  // Get screen coords of deep tile and our border tile
                  const deepScreen = th.worldToScreenCoordinates({
                    x: g.x(deepTile),
                    y: g.y(deepTile),
                  });
                  const ourScreen = th.worldToScreenCoordinates({
                    x: g.x(borderTile),
                    y: g.y(borderTile),
                  });
                  // Offset 12px further away from our border
                  const dx = deepScreen.x - ourScreen.x;
                  const dy = deepScreen.y - ourScreen.y;
                  const len = Math.sqrt(dx * dx + dy * dy) || 1;
                  const fx = deepScreen.x + (dx / len) * 12;
                  const fy = deepScreen.y + (dy / len) * 12;
                  return {
                    x: fx,
                    y: fy,
                    onScreen:
                      fx > 50 && fx < vw - 50 && fy > 30 && fy < vh - 100,
                    method: "wilderness-border",
                  };
                }
              }
            }
            // Fallback: use CC centroid
            const screen = th.worldToScreenCoordinates({
              x: ccData.centroidX,
              y: ccData.centroidY,
            });
            return {
              x: screen.x,
              y: screen.y,
              onScreen:
                screen.x > 50 &&
                screen.x < vw - 50 &&
                screen.y > 30 &&
                screen.y < vh - 100,
              method: "wilderness-centroid",
            };
          } catch (e) {
            return null;
          }
        },
        wcc,
      );
      if (wildPos) {
        x = wildPos.x;
        y = wildPos.y;
        method = wildPos.method;
      } else {
        ({ x, y } = randomBorder(zone));
        method = "wilderness-fallback";
      }
    }

    // PRIMARY: Use game API for precise targeting (works for ALL neighbor players)
    // Get position RIGHT before clicking to minimize camera drift
    const apiPos =
      target && !target._wildernessCC
        ? await getAttackClickPos(page, target.name)
        : null;
    if (target && target._wildernessCC) {
      // Already handled above — x,y are set
    } else if (apiPos && apiPos.onScreen) {
      x = apiPos.x;
      y = apiPos.y;
      method = `api-${apiPos.method}`;
    } else if (apiPos && !apiPos.onScreen) {
      // Target exists but is off-screen — pan toward it
      const dir = await getDirectionToTarget(page, target.name);
      if (dir !== null) {
        // Show pan debug marker: arrow from center toward pan direction
        const panEndX = zone.cx + Math.cos(dir) * 150;
        const panEndY = zone.cy + Math.sin(dir) * 150;
        await showDebugMarker(
          page,
          panEndX,
          panEndY,
          zone.cx,
          zone.cy,
          `PAN → ${target.name} (off-screen)`,
          "yellow",
        );
        await panInDirection(page, dir, 600);
        didPan = true;
        const apiPos2 = await getAttackClickPos(page, target.name);
        if (apiPos2) {
          x = apiPos2.x + (Math.random() - 0.5) * 20;
          y = apiPos2.y + (Math.random() - 0.5) * 20;
          method = `pan+api-${apiPos2.method}`;
        } else {
          x = zone.cx + Math.cos(dir) * zone.width * 0.4;
          y = zone.cy + Math.sin(dir) * zone.height * 0.4;
          method = "pan+api-edge";
        }
      } else {
        // API found player but can't get direction — use label as fallback
        if (target.labelX && target.labelY) {
          x = target.labelX + (Math.random() - 0.5) * 30;
          y = target.labelY + (Math.random() - 0.5) * 30;
          method = "label";
        } else {
          x = apiPos.x;
          y = apiPos.y;
          method = "api-offscreen";
        }
      }
    } else if (target && target.labelX && target.labelY) {
      // Fallback: label click (works for visible players)
      x = target.labelX + (Math.random() - 0.5) * 30;
      y = target.labelY + (Math.random() - 0.5) * 30;
      method = "label";
    } else {
      ({ x, y } = randomBorder(zone));
      method = "random-fallback";
      log(`RL: Warning — no targeting data for ${target?.name || targetIdx}`);
    }

    const isWilderness = !!(target && target._wildernessCC);

    // Clamp to safe zone to avoid clicking UI elements
    x = Math.max(zone.left, Math.min(zone.right, x));
    y = Math.max(zone.top, Math.min(zone.bottom, y));

    // Check if click would hit ANY UI element — if so, pan away or skip
    const uiCheck = await safeEval(
      page,
      (cx, cy) => {
        const pad = 15;
        const vw = window.innerWidth,
          vh = window.innerHeight;
        // Collect all UI element bounding rects
        const selectors = [
          "leader-board",
          "player-info-bar",
          "attacks-display",
          "game-right-sidebar",
          "build-menu",
          "[class*='player-info']",
          "[class*='bottom-bar']",
          "[class*='hud']",
          ".player-name",
        ];
        for (const sel of selectors) {
          const els = document.querySelectorAll(sel);
          for (const el of els) {
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) continue;
            if (
              cx >= r.left - pad &&
              cx <= r.right + pad &&
              cy >= r.top - pad &&
              cy <= r.bottom + pad
            ) {
              const elCx = (r.left + r.right) / 2;
              const elCy = (r.top + r.bottom) / 2;
              const angle = Math.atan2(elCy - vh / 2, elCx - vw / 2);
              return { blocked: true, element: sel, panAngle: angle + Math.PI };
            }
          }
        }
        // Also check: top 170px is always dangerous (banners, info bars)
        if (cy < 170)
          return { blocked: true, element: "top-zone", panAngle: Math.PI / 2 };
        // Bottom 120px (HUD)
        if (cy > vh - 120)
          return {
            blocked: true,
            element: "bottom-zone",
            panAngle: -Math.PI / 2,
          };
        return { blocked: false };
      },
      x,
      y,
    );

    if (uiCheck && uiCheck.blocked) {
      log(
        `RL: Attack ${target?.name || "?"} at (${Math.round(x)},${Math.round(y)}) behind ${uiCheck.element}`,
      );

      // If blocked by leaderboard, just close it instead of panning
      if (uiCheck.element === "leader-board") {
        log("RL: Closing leaderboard");
        const closed = await safeEval(page, () => {
          // Find the leaderboard toggle button by its icon
          const img = document.querySelector('img[src*="LeaderboardIcon"]');
          if (img) {
            const btn = img.closest("button") || img.parentElement;
            if (btn) {
              btn.click();
              return true;
            }
          }
          // Fallback: try the leader-board element directly
          const lb = document.querySelector("leader-board");
          if (!lb) return false;
          const btn = lb.querySelector(
            "button, [class*='close'], [class*='toggle']",
          );
          if (btn) {
            btn.click();
            return true;
          }
          if (lb.shadowRoot) {
            const sBtn = lb.shadowRoot.querySelector(
              "button, [class*='close'], [class*='toggle']",
            );
            if (sBtn) {
              sBtn.click();
              return true;
            }
          }
          lb.style.display = "none";
          return true;
        });
        if (closed) await sleep(150);
        // After closing, also pan toward target in case it's far off-screen
        const lbDir = target
          ? await getDirectionToTarget(page, target.name)
          : null;
        if (lbDir !== null) {
          await panInDirection(page, lbDir, 600);
          didPan = true;
          await sleep(200);
        }
      } else {
        // Show debug marker and pan for other UI elements
        await showDebugMarker(
          page,
          x,
          y,
          zone.cx,
          zone.cy,
          `BLOCKED by ${uiCheck.element} → pan`,
          "orange",
        );
        await panInDirection(page, uiCheck.panAngle, 500);
        didPan = true;
        await sleep(200);
      }
      // Re-query position after pan
      let apiRetry = null;
      if (isWilderness && target._wildernessCC) {
        apiRetry = await safeEval(
          page,
          async (ccData) => {
            try {
              const sidebar = document.querySelector("game-right-sidebar");
              if (!sidebar?.game) return null;
              const g = sidebar.game;
              const th = document.querySelector("build-menu")?.transformHandler;
              if (!th) return null;
              const me = g.myPlayer?.();
              if (!me || typeof me.borderTiles !== "function") return null;
              const vw = window.innerWidth,
                vh = window.innerHeight;
              let ourBorder = [...(await me.borderTiles()).borderTiles];
              const scored = ourBorder.map((t) => ({
                t,
                d:
                  Math.abs(g.x(t) - ccData.centroidX) +
                  Math.abs(g.y(t) - ccData.centroidY),
              }));
              scored.sort((a, b) => a.d - b.d);
              for (let i = 0; i < Math.min(scored.length, 50); i++) {
                for (const adj of g.neighbors(scored[i].t)) {
                  if (g.ownerID(adj) === 0 && (!g.isLand || g.isLand(adj))) {
                    let deep = adj;
                    for (let d = 0; d < 2; d++) {
                      for (const a2 of g.neighbors(deep)) {
                        if (
                          g.ownerID(a2) === 0 &&
                          (!g.isLand || g.isLand(a2))
                        ) {
                          deep = a2;
                          break;
                        }
                      }
                    }
                    const ds = th.worldToScreenCoordinates({
                      x: g.x(deep),
                      y: g.y(deep),
                    });
                    const os = th.worldToScreenCoordinates({
                      x: g.x(scored[i].t),
                      y: g.y(scored[i].t),
                    });
                    const dx = ds.x - os.x,
                      dy = ds.y - os.y,
                      len = Math.sqrt(dx * dx + dy * dy) || 1;
                    return {
                      x: ds.x + (dx / len) * 12,
                      y: ds.y + (dy / len) * 12,
                      onScreen:
                        ds.x > 50 &&
                        ds.x < vw - 50 &&
                        ds.y > 30 &&
                        ds.y < vh - 100,
                    };
                  }
                }
              }
              return null;
            } catch (e) {
              return null;
            }
          },
          target._wildernessCC,
        );
      } else if (target) {
        apiRetry = await getAttackClickPos(page, target.name);
      }
      if (apiRetry && apiRetry.onScreen) {
        x = apiRetry.x;
        y = apiRetry.y;
        method = "pan-retry";
      } else {
        log(`RL: Target off-screen after pan, will retry next tick`);
        await centerCamera(page);
        await restoreClickZoom(page, atkZoomSteps);
        return;
      }
    }

    // Count outgoing attacks before clicking
    const outBefore =
      (await safeEval(page, () => {
        const d = document.querySelector("attacks-display");
        if (!d) return 0;
        let c = 0;
        if (d.outgoingAttacks) c += d.outgoingAttacks.length;
        if (d.outgoingLandAttacks) c += d.outgoingLandAttacks.length;
        if (d.outgoingBoats) c += d.outgoingBoats.length;
        return c;
      })) || 0;

    // Log what we're clicking and verify it's on our territory's neighbor
    const targetName = target ? target.name : "unknown";
    log(
      `RL: Clicking attack at (${Math.round(x)},${Math.round(y)}) target=${targetName} method=${method} isLand=${target?.isLandNeighbor} isWild=${isWilderness}`,
    );
    await page.mouse.click(x, y);
    await sleep(150);
    await showDebugMarker(
      page,
      x,
      y,
      originX,
      originY,
      `ATK → ${targetName} (${method})`,
      "red",
    );

    // Check if attack actually started
    const outAfter =
      (await safeEval(page, () => {
        const d = document.querySelector("attacks-display");
        if (!d) return 0;
        let c = 0;
        if (d.outgoingAttacks) c += d.outgoingAttacks.length;
        if (d.outgoingLandAttacks) c += d.outgoingLandAttacks.length;
        if (d.outgoingBoats) c += d.outgoingBoats.length;
        return c;
      })) || 0;

    if (outAfter <= outBefore && target) {
      // Attack didn't start — zoom in for precision, retry, zoom back out
      log(`RL: Attack miss on ${targetName}, zooming in to retry`);
      for (let i = 0; i < 4; i++) await zoomStep(page, -100); // zoom in 4 steps
      await sleep(100);
      // Re-focus canvas and re-set troop ratio (zoom may have unfocused)
      await focusCanvas(page);
      const retrySteps = Math.round((troopFraction - currentAttackRatio) / 0.1);
      if (retrySteps !== 0) {
        const key = retrySteps > 0 ? "y" : "t";
        for (let i = 0; i < Math.abs(retrySteps); i++)
          await page.keyboard.press(key);
        currentAttackRatio = troopFraction;
        await sleep(30);
      }
      // For wilderness: re-run wilderness targeting (getAttackClickPos only finds players)
      let retryPos = null;
      if (isWilderness) {
        retryPos = await safeEval(
          page,
          async (ccData) => {
            try {
              const sidebar = document.querySelector("game-right-sidebar");
              if (!sidebar?.game) return null;
              const g = sidebar.game;
              const buildMenu = document.querySelector("build-menu");
              const th = buildMenu?.transformHandler;
              if (!th) return null;
              const me = g.myPlayer?.();
              if (!me || typeof me.borderTiles !== "function") return null;
              const vw = window.innerWidth,
                vh = window.innerHeight;
              let ourBorder = [];
              try {
                ourBorder = [...(await me.borderTiles()).borderTiles];
              } catch (e) {
                return null;
              }
              const ccX = ccData.centroidX,
                ccY = ccData.centroidY;
              const scored = ourBorder.map((t) => ({
                t,
                d: Math.abs(g.x(t) - ccX) + Math.abs(g.y(t) - ccY),
              }));
              scored.sort((a, b) => a.d - b.d);
              for (let i = 0; i < Math.min(scored.length, 50); i++) {
                const borderTile = scored[i].t;
                for (const adj of g.neighbors(borderTile)) {
                  if (g.ownerID(adj) === 0 && (!g.isLand || g.isLand(adj))) {
                    let deepTile = adj;
                    for (let depth = 0; depth < 2; depth++) {
                      for (const adj2 of g.neighbors(deepTile)) {
                        if (
                          g.ownerID(adj2) === 0 &&
                          (!g.isLand || g.isLand(adj2))
                        ) {
                          deepTile = adj2;
                          break;
                        }
                      }
                    }
                    const deepScreen = th.worldToScreenCoordinates({
                      x: g.x(deepTile),
                      y: g.y(deepTile),
                    });
                    const ourScreen = th.worldToScreenCoordinates({
                      x: g.x(borderTile),
                      y: g.y(borderTile),
                    });
                    const dx = deepScreen.x - ourScreen.x,
                      dy = deepScreen.y - ourScreen.y;
                    const len = Math.sqrt(dx * dx + dy * dy) || 1;
                    const fx = deepScreen.x + (dx / len) * 12,
                      fy = deepScreen.y + (dy / len) * 12;
                    return {
                      x: fx,
                      y: fy,
                      onScreen:
                        fx > 50 && fx < vw - 50 && fy > 30 && fy < vh - 100,
                    };
                  }
                }
              }
              return null;
            } catch (e) {
              return null;
            }
          },
          target._wildernessCC,
        );
      } else {
        retryPos = await getAttackClickPos(page, target.name);
      }
      if (retryPos && retryPos.onScreen) {
        // Check all UI elements before retry click
        const retryBlocked = await safeEval(
          page,
          (cx, cy) => {
            const pad = 15;
            const vh = window.innerHeight;
            const selectors = [
              "leader-board",
              "player-info-bar",
              "attacks-display",
              "game-right-sidebar",
              "build-menu",
              "[class*='player-info']",
            ];
            for (const sel of selectors) {
              const el = document.querySelector(sel);
              if (!el) continue;
              const r = el.getBoundingClientRect();
              if (r.width === 0 || r.height === 0) continue;
              if (
                cx >= r.left - pad &&
                cx <= r.right + pad &&
                cy >= r.top - pad &&
                cy <= r.bottom + pad
              )
                return true;
            }
            if (cy < 170 || cy > vh - 120) return true;
            return false;
          },
          retryPos.x,
          retryPos.y,
        );
        if (retryBlocked) {
          log(`RL: Retry target behind UI, skipping`);
        } else {
          await page.mouse.click(retryPos.x, retryPos.y);
          await sleep(150);
          await showDebugMarker(
            page,
            retryPos.x,
            retryPos.y,
            originX,
            originY,
            `ATK retry → ${targetName}`,
            "orange",
          );
          log(
            `RL: Attack retry ${targetName} at (${Math.round(retryPos.x)},${Math.round(retryPos.y)})`,
          );
        }
      }
      // Zoom back out and recenter
      for (let i = 0; i < 4; i++) await zoomStep(page, 100);
      await centerCamera(page);
    } else {
      log(
        `RL: Attack ${targetName} at (${Math.round(x)},${Math.round(y)}) ratio=${troopFraction} method=${method}`,
      );
    }

    // Restore zoom level after attack
    await restoreClickZoom(page, atkZoomSteps);
    // If we panned or zoomed for this attack, recenter so subsequent actions work correctly
    if (didPan || atkZoomSteps > 0) {
      await centerCamera(page);
    }
  } else if (actionType === ACTION_RETREAT) {
    // Click the ❌ button on the most recent outgoing attack in attacks-display
    const retreated = await safeEval(page, () => {
      const display = document.querySelector("attacks-display");
      if (!display) return null;
      // Find all cancel buttons (❌) on outgoing attacks
      const buttons = display.querySelectorAll("button");
      for (const btn of buttons) {
        if (btn.textContent?.trim() === "❌" && !btn.disabled) {
          btn.click();
          return true;
        }
      }
      return false;
    });
    log(`RL: Retreat ${retreated ? "✓" : "no outgoing attacks"}`);
  } else if (BUILD_KEYS[actionType]) {
    const { key, name, minGold } = BUILD_KEYS[actionType];

    // Don't waste time trying to build if we can't afford it
    if (gold < minGold) {
      log(
        `RL: Skip build ${name} — need ${(minGold / 1000).toFixed(0)}K gold, have ${(gold / 1000).toFixed(0)}K. Expanding instead.`,
      );
      // Use raycast to find border and expand
      for (let i = 0; i < 5; i++) {
        const angle = Math.random() * Math.PI * 2;
        const hit = await raycastToBorder(page, angle);
        if (hit) {
          const cx = Math.max(zone.left, Math.min(zone.right, hit.x));
          const cy = Math.max(zone.top, Math.min(zone.bottom, hit.y));
          await page.mouse.click(cx, cy);
          await sleep(50);
        }
      }
      return;
    }

    // Zoom in for click precision
    const buildZoomSteps = await ensureClickZoom(page);
    await centerCamera(page);

    const isBorderBuilding =
      actionType === ACTION_BUILD_DEFENSE ||
      actionType === ACTION_BUILD_SAM ||
      actionType === ACTION_BUILD_PORT;

    // PRIMARY: Use game API to find a good build position (screen coords at current zoom)
    const apiBuildPos = await getBuildClickPos(page, isBorderBuilding);

    // Helper: attempt a build click at (bx, by) — press key, click, check gold
    const tryBuild = async (bx, by, method) => {
      await page.mouse.move(bx, by);
      await sleep(100);
      await focusCanvas(page);
      await page.keyboard.press(key).catch(() => {});
      await sleep(500);
      await page.mouse.click(bx, by);
      await sleep(300);
      await page.keyboard.press("Escape").catch(() => {});
      // Check if gold decreased (= build succeeded)
      const newGold = await safeEval(page, () => {
        const sb = document.querySelector("game-right-sidebar");
        const me = sb?.game?.myPlayer?.();
        return me ? Number(me.gold()) : null;
      });
      const succeeded = newGold !== null && newGold < gold - 1000;
      await showDebugMarker(
        page,
        bx,
        by,
        bx,
        by,
        `BUILD ${name} (${method}${succeeded ? "" : " MISS"})`,
        succeeded ? "lime" : "yellow",
      );
      return succeeded;
    };

    if (apiBuildPos) {
      let built = await tryBuild(apiBuildPos.x, apiBuildPos.y, "api");
      if (!built) {
        // Retry: recenter and recompute position (already zoomed in via ensureClickZoom)
        log(`RL: Build ${name} miss, retrying with fresh coords`);
        await centerCamera(page);
        const retryPos = await getBuildClickPos(page, isBorderBuilding);
        if (retryPos) {
          built = await tryBuild(retryPos.x, retryPos.y, "api-retry");
        }
      }
      lastBuildResult = built ? "success" : "fail";
      lastActionSucceeded = built;
      log(
        `RL: Build ${name} at (${Math.round(apiBuildPos.x)},${Math.round(apiBuildPos.y)}) method=api ${built ? "✓" : "✗"} [gold=${(gold / 1000).toFixed(0)}K]`,
      );
    } else {
      // FALLBACK: Zoom in and pick random spots near center
      const savedScale = currentScale;
      const targetBuildScale = 1.5;
      const zoomInSteps = Math.max(
        0,
        Math.round(Math.log(currentScale / targetBuildScale) / Math.log(1.2)),
      );
      for (let i = 0; i < Math.min(zoomInSteps, 15); i++)
        await zoomStep(page, -100);
      await sleep(200);

      const box2 = await getCanvasBounds(page);
      if (!box2) return;
      const trueCx = box2.x + box2.width / 2;
      const trueCy = box2.y + box2.height / 2;

      const currentUnits = unitPositions || [];
      let spot = null;
      for (let attempt = 0; attempt < 12; attempt++) {
        let candidate;
        if (isBorderBuilding) {
          const angle = Math.random() * Math.PI * 2;
          const dist = 50 + Math.random() * 50;
          candidate = {
            x: trueCx + Math.cos(angle) * dist,
            y: trueCy + Math.sin(angle) * dist,
          };
        } else {
          candidate = {
            x: trueCx + (Math.random() - 0.5) * 100,
            y: trueCy + (Math.random() - 0.5) * 100,
          };
        }
        if (!isNearUnit(candidate.x, candidate.y, currentUnits)) {
          spot = candidate;
          break;
        }
      }
      if (!spot) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 70 + Math.random() * 50;
        spot = {
          x: trueCx + Math.cos(angle) * dist,
          y: trueCy + Math.sin(angle) * dist,
        };
      }

      let built = await tryBuild(spot.x, spot.y, "fallback");
      if (!built) {
        // Retry with a different spot
        log(`RL: Build ${name} fallback miss, retrying different spot`);
        const angle2 = Math.random() * Math.PI * 2;
        const dist2 = 40 + Math.random() * 40;
        const spot2 = {
          x: trueCx + Math.cos(angle2) * dist2,
          y: trueCy + Math.sin(angle2) * dist2,
        };
        built = await tryBuild(spot2.x, spot2.y, "fallback-retry");
      }
      lastBuildResult = built ? "success" : "fail";
      lastActionSucceeded = built;
      log(
        `RL: Build ${name} at (${Math.round(spot.x)},${Math.round(spot.y)}) method=fallback ${built ? "✓" : "✗"} [gold=${(gold / 1000).toFixed(0)}K]`,
      );

      // Zoom back out (fallback's own zoom)
      const zoomOutSteps = Math.max(
        0,
        Math.round(Math.log(currentScale / savedScale) / Math.log(1.2)),
      );
      for (let i = 0; i < Math.min(zoomOutSteps, 15); i++)
        await zoomStep(page, 100);
    }
    // Restore the ensureClickZoom zoom
    await restoreClickZoom(page, buildZoomSteps);
    if (buildZoomSteps > 0) await centerCamera(page);
  } else if (NUKE_KEYS[actionType]) {
    const { key, name, minGold } = NUKE_KEYS[actionType];
    if (gold < minGold) {
      log(`RL: Skip ${name} — need ${(minGold / 1000).toFixed(0)}K gold`);
      return;
    }
    // Nukes: zoom in for precision, press key, then click on enemy territory
    const nukeZoomSteps = 0; // no pre-zoom for nukes
    await page.keyboard.press(key).catch(() => {});
    await sleep(300);

    // Aim toward target player using game API first
    const nukeTargetIdx = action.targetIdx || 0;
    const nukeTarget = neighbors[nukeTargetIdx];
    let nx, ny;
    let nukeMethod = "random";

    // Try game API for target position
    const nukeApiPos = nukeTarget
      ? await getAttackClickPos(page, nukeTarget.name)
      : null;
    if (nukeApiPos) {
      if (nukeApiPos.onScreen) {
        nx = nukeApiPos.x;
        ny = nukeApiPos.y;
        nukeMethod = `api-${nukeApiPos.method}`;
      } else {
        const dir = await getDirectionToTarget(page, nukeTarget.name);
        if (dir !== null) {
          const npx = zone.cx + Math.cos(dir) * 150;
          const npy = zone.cy + Math.sin(dir) * 150;
          await showDebugMarker(
            page,
            npx,
            npy,
            zone.cx,
            zone.cy,
            `PAN → ${nukeTarget.name} (nuke, off-screen)`,
            "magenta",
          );
          await panInDirection(page, dir, 600);
          const pos2 = await getAttackClickPos(page, nukeTarget.name);
          if (pos2) {
            nx = pos2.x;
            ny = pos2.y;
            nukeMethod = `pan+api-${pos2.method}`;
          } else {
            nx = zone.cx + Math.cos(dir) * zone.width * 0.4;
            ny = zone.cy + Math.sin(dir) * zone.height * 0.4;
            nukeMethod = "pan+api-edge";
          }
        } else {
          nx = nukeApiPos.x;
          ny = nukeApiPos.y;
          nukeMethod = "api-offscreen";
        }
      }
    } else {
      // Fallback: cached direction + raycast
      let nukeAngle = null;
      if (nukeTarget && nukeTarget.labelX && nukeTarget.labelY) {
        const dx = nukeTarget.labelX - zone.cx;
        const dy = nukeTarget.labelY - zone.cy;
        if (Math.sqrt(dx * dx + dy * dy) > 10) {
          nukeAngle = Math.atan2(dy, dx);
        }
      }
      if (
        nukeAngle === null &&
        nukeTarget &&
        cachedNeighborDirections.has(nukeTarget.name)
      ) {
        nukeAngle = cachedNeighborDirections.get(nukeTarget.name).angle;
      }
      if (nukeAngle === null && nukeTarget) {
        const dir = await getDirectionToTarget(page, nukeTarget.name);
        if (dir !== null) nukeAngle = dir;
      }

      if (nukeAngle !== null) {
        const hit = await raycastToBorder(page, nukeAngle);
        if (hit) {
          const extra = 60 + Math.random() * 40;
          nx = hit.x + Math.cos(nukeAngle) * extra;
          ny = hit.y + Math.sin(nukeAngle) * extra;
          nukeMethod = "raycast";
        } else {
          nx = zone.cx + Math.cos(nukeAngle) * zone.width * 0.4;
          ny = zone.cy + Math.sin(nukeAngle) * zone.height * 0.4;
          nukeMethod = "edge";
        }
      } else {
        const angle = Math.random() * Math.PI * 2;
        const dist = 150 + Math.random() * 100;
        nx = zone.cx + Math.cos(angle) * dist;
        ny = zone.cy + Math.sin(angle) * dist;
      }
    }

    nx = Math.max(zone.left, Math.min(zone.right, nx));
    ny = Math.max(zone.top, Math.min(zone.bottom, ny));

    await page.mouse.move(nx, ny);
    await sleep(300);
    await page.mouse.click(nx, ny);
    await sleep(200);
    await page.keyboard.press("Escape").catch(() => {});
    const nukeTargetName = nukeTarget ? nukeTarget.name : "unknown";
    log(
      `RL: Launch ${name} at ${nukeTargetName} (${Math.round(nx)},${Math.round(ny)}) method=${nukeMethod}`,
    );
    // Restore zoom and recenter
    await restoreClickZoom(page, nukeZoomSteps);
    if (nukeMethod.startsWith("pan") || nukeZoomSteps > 0) {
      await centerCamera(page);
    }
    return { nukeLaunched: true };
  } else if (actionType === ACTION_UPGRADE) {
    // Right-click near a unit to open radial menu, then click "build" slot,
    // then click the unit type to upgrade it
    unitPositions = unitPositions || [];
    if (unitPositions.length === 0) {
      log("RL: Upgrade — no unit positions");
      return;
    }
    const upgZoomSteps = 0; // no pre-zoom for upgrades
    const upgUnit = unitPositions[0];
    await page.mouse.click(upgUnit.x, upgUnit.y, { button: "right" });
    await sleep(400);
    // Click the "build" slot in the SVG radial menu
    const opened = await safeEval(page, () => {
      const el =
        document.querySelector('path[data-id="build"]') ||
        document.querySelector('g[data-id="build"]');
      if (el) {
        el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        return true;
      }
      return false;
    });
    if (opened) {
      await sleep(300);
      // In the build submenu, clicking near an existing unit upgrades it
      await page.mouse.click(upgUnit.x, upgUnit.y);
    }
    log(
      `RL: Upgrade at (${Math.round(upgUnit.x)},${Math.round(upgUnit.y)}) ${opened ? "✓" : "no menu"}`,
    );
    await restoreClickZoom(page, upgZoomSteps);
    if (upgZoomSteps > 0) await centerCamera(page);
  } else if (actionType === ACTION_DELETE_UNIT) {
    // Right-click near a unit to open radial menu, then click "delete" slot
    unitPositions = unitPositions || [];
    if (unitPositions.length === 0) {
      log("RL: Delete unit — no unit positions");
      return;
    }
    // Double-check cooldown at execution time (mask may be stale)
    const canDel = await safeEval(page, () => {
      const sidebar = document.querySelector("game-right-sidebar");
      const me = sidebar?.game?.myPlayer?.();
      if (!me || typeof me.deleteUnitCooldown !== "function") return false;
      return me.deleteUnitCooldown() === 0;
    });
    if (!canDel) {
      log("RL: Delete unit — cooldown not ready, skipping");
      return;
    }
    const delZoomSteps = 0; // no pre-zoom for deletes
    // Pick last active unit (must be explicitly active)
    const delUnit =
      [...unitPositions].reverse().find((u) => u.active === true) ||
      unitPositions[unitPositions.length - 1];
    await page.mouse.click(delUnit.x, delUnit.y, { button: "right" });
    await sleep(400);
    const deleted = await safeEval(page, () => {
      const el =
        document.querySelector('path[data-id="delete"]') ||
        document.querySelector('g[data-id="delete"]');
      if (el) {
        el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        return true;
      }
      return false;
    });
    log(
      `RL: Delete unit at (${Math.round(delUnit.x)},${Math.round(delUnit.y)}) ${deleted ? "✓" : "no menu"}`,
    );
    await restoreClickZoom(page, delZoomSteps);
    if (delZoomSteps > 0) await centerCamera(page);
    if (deleted) return { deleted: true };
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
    headless: HEADLESS ? "new" : false,
    defaultViewport: HEADLESS ? { width: 1400, height: 900 } : null,
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
          !p.url().includes(new URL(GAME_URL).host) &&
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
  let lastUnitPositions = [];
  let lastGameState = null;
  let lastRecenter = 0;
  lastGold = 0;
  currentScale = 1.8; // reset to game default at game start
  lastTerritoryPct = 0;
  let spawnTime = 0;
  let lastNeighborScan = 0; // Brief zoom-out to refresh neighbor label data
  const playerRelations = new Map(); // name -> Relation (0=Hostile,1=Distrustful,2=Neutral,3=Friendly)
  let nukeCount = 0; // Track nukes launched (increment on launch, decrement on resolve)
  const cachedNeighborDirections = new Map(); // name -> { angle }

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
        // Zoom out a bit on first spawn attempt so we can see land
        if (spawnAttempts === 0) {
          await focusCanvas(page);
          for (let i = 0; i < 5; i++) await zoomStep(page, 100); // zoom out slightly
          await sleep(200);
        }
        const box = await getCanvasBounds(page);
        if (box) {
          const cx = box.x + box.width * (0.3 + Math.random() * 0.4);
          const cy = box.y + box.height * (0.2 + Math.random() * 0.4);
          log(`Spawn click at (${Math.round(cx)},${Math.round(cy)})`);
          await page.mouse.click(cx, cy);
        }
        spawnAttempts++;
        await sleep(2000);

        // Check if we spawned: the "Choose a starting location" banner must be GONE
        // AND we must have actual tiles. Both conditions required to avoid false positives.
        const spawnBannerGone = await safeEval(page, () => {
          // Check for the spawn location prompt — various possible selectors
          const body = document.body?.innerText || "";
          if (body.includes("Choose a starting location")) return false;
          if (body.includes("choose a starting location")) return false;
          // Also check for common spawn UI elements
          const spawnUI = document.querySelector(
            ".spawn-overlay, .spawn-prompt, [class*='spawn'], [class*='starting-location']",
          );
          if (spawnUI && spawnUI.offsetParent !== null) return false;
          return true;
        });
        const hudState = await extractGameState(page, BOT_NAME);
        // Require BOTH: spawn banner gone AND actual tiles owned
        const didSpawn =
          spawnBannerGone &&
          hudState &&
          (hudState.myGold > 100 || hudState.myTiles > 10);
        if (!didSpawn) {
          log(
            `Spawn check: bannerGone=${spawnBannerGone}, tiles=${hudState?.myTiles || 0}, gold=${hudState?.myGold || 0}`,
          );
        }

        if (didSpawn) {
          spawned = true;
          spawnTime = Date.now();
          log(
            `Spawned! troops=${hudState?.myTroops || "?"}, territory=${((hudState?.territoryPct || 0) * 100).toFixed(1)}%`,
          );

          // Center camera, then zoom in until we can see our own label
          await centerCamera(page);
          await sleep(500); // extra wait for center animation to settle

          // Zoom in until our label is visible on screen
          for (let attempt = 0; attempt < 25; attempt++) {
            const canSeeLabel = await safeEval(
              page,
              (botName) => {
                const spans = document.querySelectorAll(".player-name-span");
                for (const s of spans) {
                  // Match by includes (handles clan tag prefix like "[XXDAR] xXDark...")
                  if (s.textContent?.includes(botName)) {
                    const container = s.closest(".player-name")?.parentElement;
                    if (!container) continue;
                    const r = container.getBoundingClientRect();
                    const vw = window.innerWidth,
                      vh = window.innerHeight;
                    return (
                      r.width > 0 &&
                      r.left > 30 &&
                      r.right < vw - 30 &&
                      r.top > 30 &&
                      r.bottom < vh - 80
                    );
                  }
                }
                return false;
              },
              BOT_NAME,
            );
            if (canSeeLabel) {
              log(`Spawn zoom: found our label after ${attempt} zoom steps`);
              break;
            }
            await zoomStep(page, -100); // zoom in
            await sleep(50);
          }
          // One final re-center after all zooming
          await centerCamera(page);
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

        // Sync actual scale from game
        const realScale2 = await safeEval(page, () => {
          const bm = document.querySelector("build-menu");
          return bm?.transformHandler?.scale ?? null;
        });
        if (realScale2) currentScale = realScale2;

        // Zoom targets: balance seeing our territory and neighbors
        const tPct = lastTerritoryPct;
        let targetScale;
        if (tPct < 0.0005) targetScale = 10.0;
        else if (tPct < 0.001) targetScale = 6.0;
        else if (tPct < 0.003) targetScale = 4.0;
        else if (tPct < 0.005) targetScale = 3.0;
        else if (tPct < 0.01) targetScale = 2.5;
        else if (tPct < 0.03) targetScale = 2.0;
        else if (tPct < 0.05) targetScale = 1.8;
        else targetScale = 1.5;

        const scaleRatio = currentScale / targetScale;
        if (scaleRatio > 1.3 || scaleRatio < 0.77) {
          const stepsNeeded = Math.round(
            Math.log(targetScale / currentScale) / Math.log(1.2),
          );
          const clamped = Math.max(-10, Math.min(10, stepsNeeded));
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
      // We get neighbor data from the game API now, so we just need directions.
      // Zoom out, grab label positions, zoom back to target scale.
      if (
        Date.now() - lastNeighborScan > 15000 &&
        Date.now() - spawnTime > 15000
      ) {
        lastNeighborScan = Date.now();
        // Sync actual scale from game
        const realScale = await safeEval(page, () => {
          const bm = document.querySelector("build-menu");
          return bm?.transformHandler?.scale ?? null;
        });
        if (realScale) currentScale = realScale;

        if (currentScale > 2.5) {
          const tPct = lastTerritoryPct;
          let targetScale;
          if (tPct < 0.0005) targetScale = 10.0;
          else if (tPct < 0.001) targetScale = 6.0;
          else if (tPct < 0.003) targetScale = 4.0;
          else if (tPct < 0.005) targetScale = 3.0;
          else if (tPct < 0.01) targetScale = 2.5;
          else if (tPct < 0.03) targetScale = 2.0;
          else if (tPct < 0.05) targetScale = 1.8;
          else targetScale = 1.5;

          // Zoom out to ~1.5 for label visibility
          const outSteps = Math.round(
            Math.log(currentScale / 1.5) / Math.log(1.2),
          );
          const clamped = Math.min(outSteps, 8);
          for (let i = 0; i < clamped; i++) await zoomStep(page, 100);
          await sleep(600);

          // Extract neighbor directions while zoomed out
          const zoomedOutState = await extractGameState(page, BOT_NAME);
          if (zoomedOutState && zoomedOutState.myLabelX) {
            const ox = zoomedOutState.myLabelX;
            const oy = zoomedOutState.myLabelY;
            cachedNeighborDirections.clear();
            for (const n of zoomedOutState.neighbors) {
              if (n.labelX && n.labelY) {
                const dx = n.labelX - ox;
                const dy = n.labelY - oy;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > 5) {
                  cachedNeighborDirections.set(n.name, {
                    angle: Math.atan2(dy, dx),
                  });
                }
              }
            }
            log(
              `Neighbor scan: cached directions for ${cachedNeighborDirections.size} players`,
            );
          }

          // Zoom back to target scale (not previous scale)
          const nowScale = await safeEval(page, () => {
            const bm = document.querySelector("build-menu");
            return bm?.transformHandler?.scale ?? null;
          });
          if (nowScale) currentScale = nowScale;
          const inSteps = Math.round(
            Math.log(targetScale / currentScale) / Math.log(1.2),
          );
          const clampedIn = Math.max(-12, Math.min(12, inSteps));
          if (clampedIn > 0) {
            for (let i = 0; i < clampedIn; i++) await zoomStep(page, -100);
          }
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

          lastGold = gameState.myGold || 0;
          lastTerritoryPct = gameState.territoryPct || 0;
          const g = lastGold;
          const neighbors = gameState.neighbors || [];
          const hasT = gameState.myTroops > 10;
          const hasLandNeighbor = neighbors.some(
            (n) => n.isLandNeighbor && !n.isAllied,
          );
          const hasSeaNeighbor = neighbors.some(
            (n) =>
              !n.isLandNeighbor &&
              !n.isAllied &&
              !n.id?.startsWith("wilderness_"),
          );
          const hasOut = (gameState.outgoingAttacks || 0) > 0;
          const hasUnits = gameState.hasUnits || false;
          const hasSilo = gameState.hasSilo || false;
          const hasPort = gameState.hasPort || false;
          const numWarships = gameState.numWarships || 0;

          // Use game API costs (accounts for scaling) to determine affordability
          const costs = gameState._buildCosts || {};
          const canBuildAPI = gameState.canBuildFromAPI || {};
          const canAfford = (type) => {
            const cost = costs[type];
            return cost !== undefined ? g >= cost : false;
          };

          // Action mask — check affordability using real costs from API
          gameState.actionMask = [
            true, // 0: NOOP
            hasLandNeighbor && hasT, // 1: ATTACK (land neighbors only)
            hasSeaNeighbor && hasT && (gameState.hasCoast || hasPort), // 2: BOAT_ATTACK (needs coast access)
            hasOut, // 3: RETREAT
            canAfford("City"), // 4: BUILD_CITY
            canAfford("Factory"), // 5: BUILD_FACTORY
            canAfford("Defense Post"), // 6: BUILD_DEFENSE
            canAfford("Port") && canBuildAPI["Port"] !== false, // 7: BUILD_PORT (needs coast)
            canAfford("SAM Launcher"), // 8: BUILD_SAM
            canAfford("Missile Silo"), // 9: BUILD_SILO
            canAfford("Warship") && hasPort, // 10: BUILD_WARSHIP
            hasSilo && neighbors.length > 0 && canAfford("Atom Bomb"), // 11: LAUNCH_ATOM
            hasSilo && neighbors.length > 0 && canAfford("Hydrogen Bomb"), // 12: LAUNCH_HBOMB
            hasSilo && neighbors.length > 0 && canAfford("MIRV"), // 13: LAUNCH_MIRV
            numWarships > 0 && neighbors.length > 0, // 14: MOVE_WARSHIP
            hasUnits && g >= 50_000, // 15: UPGRADE
            hasUnits &&
              gameState.canDeleteUnit === true &&
              (gameState.unitPositions || []).some((u) => u.active === true), // 16: DELETE_UNIT (strict: cooldown must be 0, unit must be explicitly active)
          ];
          // Add build feedback (tracked from previous action)
          gameState.lastBuildResult = lastBuildResult;
          gameState.lastActionSucceeded = lastActionSucceeded;
          lastBuildResult = "none"; // reset after sending

          const action = await queryPolicy(gameState);
          if (action) {
            currentAction = action;
            lastNeighbors = gameState.neighbors || [];
            lastUnitPositions = gameState.unitPositions || [];
            lastGameState = gameState;
            if (tick % 10 === 0) {
              log(
                `RL action: type=${action.actionType} target=${action.targetIdx} troops=${action.troopFraction} gold=${(lastGold / 1000).toFixed(0)}K`,
              );
            }
            // Update real-time debug overlay with obs vector + action
            const ACTION_NAMES = [
              "NOOP",
              "ATTACK",
              "BOAT_ATK",
              "RETREAT",
              "CITY",
              "FACTORY",
              "DEFENSE",
              "PORT",
              "SAM",
              "SILO",
              "WARSHIP",
              "ATOM",
              "HBOMB",
              "MIRV",
              "MV_SHIP",
              "UPGRADE",
              "DELETE",
            ];
            const mask = gameState.actionMask || [];
            const neighbors = gameState.neighbors || [];
            const targetName =
              action.targetIdx < neighbors.length
                ? neighbors[action.targetIdx]?.name || `idx${action.targetIdx}`
                : `idx${action.targetIdx}`;
            const debugData = {
              // Player stats (raw)
              tiles: gameState.myTiles || 0,
              troops: gameState.myTroops || 0,
              gold: gameState.myGold || 0,
              pct: ((gameState.territoryPct || 0) * 100).toFixed(2),
              tick: gameState.tick || 0,
              inAtk: gameState.incomingAttacks || 0,
              outAtk: gameState.outgoingAttacks || 0,
              units: (gameState.units || []).length,
              silo: gameState.hasSilo,
              port: gameState.hasPort,
              sam: gameState.hasSAM,
              ships: gameState.numWarships || 0,
              nukes: gameState.numNukes || 0,
              coast: gameState.hasCoast || false,
              canDel: gameState.canDeleteUnit || false,
              // Neighbors summary
              nCount: neighbors.length,
              neighbors: neighbors.slice(0, 16).map((n, i) => ({
                i,
                name: (n.name || "?").slice(0, 12),
                tiles: n.tiles || 0,
                troops: n.troops || 0,
                rel: n.relation || 0,
                land: n.isLandNeighbor ? "L" : "S",
                ally: n.isAllied ? "A" : "",
              })),
              // Action mask
              maskStr: mask
                .map((v, i) => (v ? ACTION_NAMES[i] || i : null))
                .filter(Boolean)
                .join(", "),
              // Current action
              actionName: ACTION_NAMES[action.actionType] || action.actionType,
              targetIdx: action.targetIdx,
              targetName,
              troopFrac: action.troopFraction,
              // Build costs
              costs: gameState._buildCosts || {},
              // Feedback
              lastBuild: gameState.lastBuildResult || "none",
              lastOK: gameState.lastActionSucceeded || false,
            };
            await safeEval(
              page,
              (d) => {
                let panel = document.getElementById("rl-obs-debug");
                if (!panel) {
                  panel = document.createElement("div");
                  panel.id = "rl-obs-debug";
                  panel.style.cssText = `
                  position:fixed; top:10px; right:10px; width:340px;
                  background:rgba(0,0,0,0.85); color:#0f0;
                  font:11px/1.4 monospace; border-radius:6px;
                  z-index:100000; pointer-events:auto; user-select:text;
                  border:1px solid #0f04; scrollbar-width:thin;
                `;
                  document.body.appendChild(panel);
                }
                // Preserve collapsed state
                const wasCollapsed = panel.dataset.collapsed === "1";
                const fmt = (n) =>
                  n >= 1e6
                    ? (n / 1e6).toFixed(1) + "M"
                    : n >= 1e3
                      ? (n / 1e3).toFixed(1) + "K"
                      : String(n);
                const relStr = (r) =>
                  ["hostile", "neutral", "friendly", "allied"][r] || "?";
                // Header bar (always visible, clickable to toggle)
                let html = `<div id="rl-obs-header" style="color:#ff0;font-weight:bold;padding:8px;cursor:pointer;display:flex;justify-content:space-between;align-items:center">
                  <span>RL OBS DEBUG</span>
                  <span style="color:#ff0;font-size:13px"><b>${d.actionName}</b>→${d.targetName}</span>
                  <span id="rl-obs-toggle" style="font-size:14px">${wasCollapsed ? "▶" : "▼"}</span>
                </div>`;
                // Collapsible body
                html += `<div id="rl-obs-body" style="padding:0 8px 8px 8px;max-height:80vh;overflow-y:auto;${wasCollapsed ? "display:none" : ""}">`;
                // Player stats
                html += `<div style="color:#aaf">── Player ──</div>`;
                html += `<div>tiles: ${fmt(d.tiles)} | pct: ${d.pct}%</div>`;
                html += `<div>troops: ${fmt(d.troops)} | gold: ${fmt(d.gold)}</div>`;
                html += `<div>tick: ${d.tick} | in: ${d.inAtk} out: ${d.outAtk}</div>`;
                html += `<div>units: ${d.units} | silo:${d.silo ? 1 : 0} port:${d.port ? 1 : 0} sam:${d.sam ? 1 : 0} ships:${d.ships} nukes:${d.nukes}</div>`;
                html += `<div>coast:${d.coast ? 1 : 0} canDel:${d.canDel ? 1 : 0}</div>`;
                // Action
                html += `<div style="color:#ff0;margin-top:4px">── Action ──</div>`;
                html += `<div style="color:#ff0;font-size:13px"><b>${d.actionName}</b> → ${d.targetName} (${(d.troopFrac * 100).toFixed(0)}%)</div>`;
                html += `<div style="color:#888">lastBuild: ${d.lastBuild} | lastOK: ${d.lastOK ? 1 : 0}</div>`;
                // Mask
                html += `<div style="color:#aaf;margin-top:4px">── Mask ──</div>`;
                html += `<div style="color:#8f8;word-break:break-all">${d.maskStr || "NONE"}</div>`;
                // Neighbors
                html += `<div style="color:#aaf;margin-top:4px">── Neighbors (${d.nCount}) ──</div>`;
                for (const n of d.neighbors) {
                  const hi =
                    n.i === d.targetIdx ? "color:#ff0;font-weight:bold" : "";
                  html += `<div style="${hi}">${n.i}: ${n.name} ${n.land}${n.ally} t:${fmt(n.tiles)} tr:${fmt(n.troops)} ${relStr(n.rel)}</div>`;
                }
                // Costs
                html += `<div style="color:#aaf;margin-top:4px">── Costs ──</div>`;
                const costEntries = Object.entries(d.costs).filter(
                  ([, v]) => v > 0,
                );
                if (costEntries.length) {
                  html += `<div>${costEntries.map(([k, v]) => `${k}:${fmt(v)}`).join(" | ")}</div>`;
                } else {
                  html += `<div style="color:#888">no cost data</div>`;
                }
                html += `</div>`; // close body
                panel.innerHTML = html;
                // Attach toggle listener
                document
                  .getElementById("rl-obs-header")
                  .addEventListener("click", () => {
                    const body = document.getElementById("rl-obs-body");
                    const toggle = document.getElementById("rl-obs-toggle");
                    const collapsed = body.style.display !== "none";
                    body.style.display = collapsed ? "none" : "";
                    toggle.textContent = collapsed ? "▶" : "▼";
                    panel.dataset.collapsed = collapsed ? "1" : "0";
                  });
              },
              debugData,
            );
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
        lastUnitPositions,
        lastGameState,
        cachedNeighborDirections,
      );
      if (actionResult?.nukeLaunched) nukeCount++;
      // Clear action after execution — matches training where 1 action = 1 step.
      // Repeating attacks wastes troops and doesn't match how the model was trained.
      if (currentAction && currentAction.actionType !== ACTION_NOOP) {
        currentAction = null;
      }

      // ── Accept alliance requests ──
      if (tick % 10 === 0) {
        const allyName = await safeEval(page, () => {
          for (const el of document.querySelectorAll("*")) {
            if (el.textContent?.trim() === "Accept") {
              const r = el.getBoundingClientRect();
              if (r.width > 20 && r.width < 200) {
                // Find the requester's name in a nearby element
                const container =
                  el.closest("[class*='alliance'], [class*='request'], div") ||
                  el.parentElement?.parentElement;
                const nameEl = container?.querySelector(
                  ".truncate, [class*='name']",
                );
                const name = nameEl?.textContent?.trim() || null;
                el.click();
                return name;
              }
            }
          }
          return null;
        });
        if (allyName) {
          playerRelations.set(allyName, 3); // Friendly
          log(`Alliance accepted with ${allyName} → relation=3`);
        }
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
          `[${elapsed.toFixed(0)}s] ${pct}% terr, troops=${troops}, gold=${(lastGold / 1000).toFixed(0)}K, tiles=${gs?.myTiles || 0}, tick=${gs?.tick || 0}, units=${units}, ${attacks}, ${bldgs}, neighbors=${nNeighbors}(${visNeighbors}vis), scale=${currentScale.toFixed(1)}, dirs=${cachedNeighborDirections.size}`,
        );
        if (gs?._borderDebug) {
          const bd = gs._borderDebug;
          const wildCount = bd.wildernessCCCount || 0;
          const mask = gs.actionMask || [];
          const maskStr = mask
            .map((v, i) => (v ? i : null))
            .filter((v) => v !== null)
            .join(",");
          log(
            `  border: myBorder=${bd.myBorderLen} landN=[${bd.landNames.slice(0, 3)}] apiN=${bd.apiNeighborCount} wildCCs=${wildCount} mask=[${maskStr}]`,
          );
          if (gs._buildDebug) {
            log(`  build: ${JSON.stringify(gs._buildDebug.slice(0, 5))}`);
          }
          if (gs._buildError) {
            log(`  buildErr: ${gs._buildError}`);
          }
          if (gs.canBuildFromAPI) {
            log(`  canBuild: ${JSON.stringify(gs.canBuildFromAPI)}`);
          }
        }
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
