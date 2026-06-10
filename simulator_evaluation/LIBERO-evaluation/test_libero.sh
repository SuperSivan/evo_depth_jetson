#!/bin/bash
# LIBERO evaluation - pass log path, task, server_url.
# Usage: ./test_libero.sh [log_path] [task] [server_url] [stream_port]
#   log_path: dir to save log.txt and videos (optional, default uses ckpt_name)
#   task: libero_spatial | libero_goal | libero_object | libero_10  (default: libero_spatial)
#   server_url: e.g. ws://0.0.0.0:9000  (default: ws://0.0.0.0:9000)
#   stream_port: e.g. 8080 to enable live browser stream (optional)

LOG_PATH=$1
TASK=${2:-libero_spatial}
SERVER_URL=${3:-ws://0.0.0.0:9000}
STREAM_PORT=$4

echo "=============================="
echo "Task: $TASK | Server: $SERVER_URL"
echo "=============================="
export PYTHONPATH=$PYTHONPATH:./LIBERO/
STREAM_ARGS=""
if [ -n "$STREAM_PORT" ]; then
    STREAM_ARGS="--stream_port $STREAM_PORT"
    echo "Web stream: http://127.0.0.1:${STREAM_PORT}/"
fi

if [ -n "$LOG_PATH" ]; then
    python libero_client.py --task_suites "$TASK" --server_url "$SERVER_URL" --log_file "${LOG_PATH}/log.txt" --video_log_dir "${LOG_PATH}/videos" $STREAM_ARGS
else
    python libero_client.py --task_suites "$TASK" --server_url "$SERVER_URL" $STREAM_ARGS
fi