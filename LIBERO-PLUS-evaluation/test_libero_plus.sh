#!/bin/bash
# LIBERO-PLUS evaluation - pass log path, task, and filter category.
# Usage: ./test_libero_plus.sh <log_path> [task] [filter_category]
#   log_path: dir to save log.txt and videos (required)
#   task: libero_spatial | libero_goal | libero_object | libero_10  (default: libero_spatial)
#   filter_category: perturbation category to evaluate, comma-separated; empty string means all tasks (default: background)

LOG_PATH=${1:?Usage: $0 <log_path> [task] [filter_category]}
TASK=${2:-libero_spatial}
FILTER_CATEGORY=${3:-background}
export PYTHONPATH=$PYTHONPATH:./LIBERO-plus/
echo "=============================="
echo "Log path: $LOG_PATH | Task: $TASK | Filter: $FILTER_CATEGORY"
echo "=============================="

python libero_plus_client.py \
    --task_suites "$TASK" \
    --filter_category "$FILTER_CATEGORY" \
    --log_file "${LOG_PATH}/log.txt" \
    --video_log_dir "${LOG_PATH}/videos"
