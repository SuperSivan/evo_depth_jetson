#!/bin/bash
# LIBERO evaluation - pass log path, task, server_url.
# Usage: ./test_libero.sh [log_path] [task] [server_url]
#   log_path: dir to save log.txt and videos (optional, default uses ckpt_name)
#   task: libero_spatial | libero_goal | libero_object | libero_10  (default: libero_spatial)
#   server_url: e.g. ws://0.0.0.0:9000  (default: ws://0.0.0.0:9000)

LOG_PATH=$1
TASK=${2:-libero_spatial}
SERVER_URL=${3:-ws://0.0.0.0:9000}

echo "=============================="
echo "Task: $TASK | Server: $SERVER_URL"
echo "=============================="
export PYTHONPATH=$PYTHONPATH:./LIBERO/
if [ -n "$LOG_PATH" ]; then
    python libero_client.py --task_suites "$TASK" --server_url "$SERVER_URL" --log_file "${LOG_PATH}/log.txt" --video_log_dir "${LOG_PATH}/videos"
else
    python libero_client.py --task_suites "$TASK" --server_url "$SERVER_URL"
fi