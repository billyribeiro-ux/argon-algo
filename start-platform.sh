#!/usr/bin/env bash
# ==============================================================================
# ARGON-4 QUANT INTELLIGENCE PLATFORM — ONE-COMMAND MAC STUDIO LAUNCHER
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "[ARGON-4] Shutting down live execution daemon & Svelte 5 command terminal..."
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    echo "[ARGON-4] Clean shutdown complete."
}
trap cleanup SIGINT SIGTERM EXIT

echo "=============================================================================="
echo " ARGON-4 QUANT INTELLIGENCE PLATFORM — MAC STUDIO RELEASE BUILD & LAUNCH"
echo "=============================================================================="

# 1. Build live-execution-server in optimized release mode
echo "[1/4] Building live-execution-server (--release)..."
cargo build --release -p live-execution-server

# Free ports 8080 and 5173 if an old process is lingering
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# 2. Launch Rust live-execution-server in the background
echo "[2/4] Starting live-execution-server on http://127.0.0.1:8080 ..."
./target/release/live-execution-server &
BACKEND_PID=$!

# 3. Wait for http://127.0.0.1:8080/api/v1/health to report ready
echo "[3/4] Waiting for Yahoo Finance / CBOE / FINRA backfill & /api/v1/health ..."
MAX_WAIT_SECS=60
ELAPSED=0
until curl -fsS "http://127.0.0.1:8080/api/v1/health" >/dev/null 2>&1; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[ERROR] live-execution-server exited unexpectedly."
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [[ $ELAPSED -ge $MAX_WAIT_SECS ]]; then
        echo "[ERROR] Timed out waiting for http://127.0.0.1:8080/api/v1/health"
        exit 1
    fi
done

echo "[OK] Rust Execution & Market Data Bus ready on http://127.0.0.1:8080"
echo ""
echo "=============================================================================="
echo " CLICKABLE TERMINAL & MULTI-MONITOR POP-OUT URLS:"
echo "   • Main Command Cockpit (All-in-One):   http://127.0.0.1:5173/"
echo "   • Window 1 (Blueprint & 2x2 Quad):     http://127.0.0.1:5173/blueprint"
echo "   • Window 2 (Positions & Trade Ledger): http://127.0.0.1:5173/positions-and-ledger"
echo "   • Window 3 (Macro & Adapter Switcher): http://127.0.0.1:5173/macro-and-adapters"
echo "   • Rust REST/WS Health Telemetry:       http://127.0.0.1:8080/api/v1/health"
echo "=============================================================================="
echo ""

# 4. Launch Svelte 5 Web Terminal on http://127.0.0.1:5173
echo "[4/4] Starting Svelte 5 Web Terminal on http://127.0.0.1:5173 ..."
cd "$ROOT_DIR/web-terminal"

AGY_NODE="/Applications/Antigravity IDE.app/Contents/Frameworks/Antigravity IDE Helper (Plugin).app/Contents/MacOS/Antigravity IDE Helper (Plugin)"

if command -v pnpm >/dev/null 2>&1; then
    pnpm run dev --host 127.0.0.1 --port 5173 &
    FRONTEND_PID=$!
elif command -v npm >/dev/null 2>&1; then
    npm run dev -- --host 127.0.0.1 --port 5173 &
    FRONTEND_PID=$!
elif command -v node >/dev/null 2>&1; then
    node ./node_modules/vite/bin/vite.js dev --host 127.0.0.1 --port 5173 &
    FRONTEND_PID=$!
elif [[ -x "$AGY_NODE" ]]; then
    ELECTRON_RUN_AS_NODE=1 "$AGY_NODE" ./node_modules/vite/bin/vite.js dev --host 127.0.0.1 --port 5173 &
    FRONTEND_PID=$!
else
    echo "[ERROR] No Node.js / pnpm / npm runtime found in PATH."
    exit 1
fi

wait "$BACKEND_PID" "$FRONTEND_PID"

