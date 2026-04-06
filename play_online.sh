#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RL_DIR="$SCRIPT_DIR/rl"
VENV="$RL_DIR/.venv/bin/activate"
MODEL="$RL_DIR/checkpoints/best_model.pt"
PORT=8765
BOT_NAME="${1:-xXDarkLord42Xx}"
GAME_URL="${2:-https://openfront.io}"

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

# Kill policy server on exit
cleanup() {
    if [ -n "$POLICY_PID" ]; then
        echo "Stopping policy server (pid $POLICY_PID)..."
        kill "$POLICY_PID" 2> /dev/null || true
    fi
}
trap cleanup EXIT

# Start policy server in background
echo "Starting policy server on port $PORT..."
source "$VENV"
python "$RL_DIR/play.py" --model "$MODEL" --mode server --port "$PORT" &
POLICY_PID=$!

# Wait for it to be ready
echo -n "Waiting for policy server"
for i in $(seq 1 20); do
    if curl -s -o /dev/null "http://localhost:$PORT" 2> /dev/null \
        || kill -0 "$POLICY_PID" 2> /dev/null && sleep 0.5 \
        && curl -s -X POST "http://localhost:$PORT" -H "Content-Type: application/json" -d '{}' -o /dev/null 2> /dev/null; then
        break
    fi
    echo -n "."
    sleep 0.5
done
echo " ready."

# Run Puppeteer bot (foreground — Ctrl+C stops everything)
echo "Starting Puppeteer bot as \"$BOT_NAME\" on $GAME_URL..."
cd "$RL_DIR"
node bot-rl.mjs "$BOT_NAME" "http://localhost:$PORT" "$GAME_URL"
