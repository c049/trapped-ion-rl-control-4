#!/usr/bin/env bash
set -euo pipefail

# One-click launcher: starts training server + client, logs output,
# and auto-generates plots when training finishes.
# Run this inside an activated environment (venv/conda) on a compute node.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +"%Y%m%d_%H%M%S")"
SERVER_LOG="${LOG_DIR}/server_${TS}.log"
CLIENT_LOG="${LOG_DIR}/client_${TS}.log"
RUNNER_LOG="${LOG_DIR}/runner_${TS}.log"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "Logs:"
echo "  server -> ${SERVER_LOG}"
echo "  client -> ${CLIENT_LOG}"
echo "Plots will be generated into ${ROOT_DIR}/outputs/"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Training server (background)
python -u "${ROOT_DIR}/trapped_ion_cat_training_server.py" 2>&1 | tee -a "${SERVER_LOG}" &
SERVER_PID=$!
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-15}"
sleep "${SERVER_STARTUP_WAIT}"

# Client (foreground)
python -u "${ROOT_DIR}/trapped_ion_cat_client.py" 2>&1 | tee -a "${CLIENT_LOG}"

# Post-processing
python -u "${ROOT_DIR}/parse_trapped_ion_cat_data.py" 2>&1 | tee -a "${RUNNER_LOG}"
python -u "${ROOT_DIR}/plot_trapped_ion_cat_pulses.py" 2>&1 | tee -a "${RUNNER_LOG}"
