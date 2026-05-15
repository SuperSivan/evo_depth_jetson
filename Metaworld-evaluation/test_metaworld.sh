#!/usr/bin/env bash
# MetaWorld MT50 evaluation — EvoDepth websocket client.
# Usage: ./test_metaworld.sh [log_dir] [server_url] [extra_client_args...]
#   log_dir:    optional; passed as --log_dir (default: see metaworld_eval.yaml)
#   server_url: optional; e.g. ws://0.0.0.0:9000
# Examples:
#   ./test_metaworld.sh
#   ./test_metaworld.sh ./my_logs ws://127.0.0.1:9000
#   ./test_metaworld.sh ./my_logs ws://127.0.0.1:9000 --horizon 17 --target_level easy

set -euo pipefail

LOG_DIR=${1:-}
SERVER_URL=${2:-}
if [[ $# -ge 2 ]]; then
  EXTRA=("${@:3}")
else
  EXTRA=()
fi

CMD=(python metaworld_client.py)
[[ -n "${LOG_DIR}" ]] && CMD+=(--log_dir "${LOG_DIR}")
[[ -n "${SERVER_URL}" ]] && CMD+=(--server_url "${SERVER_URL}")
CMD+=("${EXTRA[@]}")

echo "=============================="
echo "Running: ${CMD[*]}"
echo "=============================="
exec "${CMD[@]}"
