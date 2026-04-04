#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RL_DIR="$SCRIPT_DIR/rl"
VENV="$RL_DIR/.venv/bin/activate"
MODEL="$RL_DIR/checkpoints/best_model.pt"
PORT=8765
GAME_PORT=3000
BOT_NAME="${1:-xXDarkLord42Xx}"

# Check prerequisites
if [ ! -f "$VENV" ]; then
  echo "Error: Python venv not found. Run:"
  echo "  cd rl && uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install torch numpy gymnasium huggingface_hub"
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "Error: Model not found at $MODEL. Run:"
  echo "  source rl/.venv/bin/activate && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('mischievers/openfront-rl-agent', 'best_model.pt', local_dir='rl/checkpoints', force_download=True)\""
  exit 1
fi

cd "$SCRIPT_DIR"

if [ ! -d "node_modules" ]; then
  echo "Installing npm dependencies..."
  npm ci --ignore-scripts
fi

# Build the frontend if static/index.html is missing or Main.ts is newer.
# The build bakes window.__rl__ into the JS bundle served by the game server.
if [ ! -f "static/index.html" ] || [ "src/client/Main.ts" -nt "static/index.html" ]; then
  echo "Building frontend (vite)..."
  npx vite build --mode development
fi

# Kill background processes on exit
POLICY_PID=""
GAME_PID=""
cleanup() {
  [ -n "$POLICY_PID" ] && kill "$POLICY_PID" 2>/dev/null || true
  [ -n "$GAME_PID"   ] && kill "$GAME_PID"   2>/dev/null || true
}
trap cleanup EXIT

# Start game server (serves built static files + handles WebSocket game logic)
echo "Starting game server on port $GAME_PORT..."
GAME_ENV=dev SKIP_BROWSER_OPEN=true npm run start:server-dev &
GAME_PID=$!

echo -n "Waiting for game server"
for i in $(seq 1 40); do
  if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$GAME_PORT" 2>/dev/null | grep -qE "^[23]"; then
    break
  fi
  echo -n "."
  sleep 1
done
echo " ready."

# Start policy server in background
echo "Starting policy server on port $PORT..."
source "$VENV"
#python "$RL_DIR/play.py" --model "$MODEL" --mode server --port "$PORT" &
python "$RL_DIR/rules_server.py" --port "$PORT" &
POLICY_PID=$!

echo -n "Waiting for policy server"
for i in $(seq 1 20); do
  if curl -s -X POST "http://localhost:$PORT" -H "Content-Type: application/json" -d '{}' -o /dev/null 2>/dev/null; then
    break
  fi
  echo -n "."
  sleep 0.5
done
echo " ready."

# Run Puppeteer bot against the local game server (Ctrl+C stops everything)
echo "Starting Puppeteer bot as \"$BOT_NAME\"..."
cd "$RL_DIR"
node bot-rl.mjs "$BOT_NAME" "http://localhost:$PORT" "http://localhost:$GAME_PORT"
