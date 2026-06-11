#!/usr/bin/env bash
set -euo pipefail

# 一键进入 Jetson 满性能并启动 Evo_depth benchmark
# 用法示例：
#   bash benchmark_max_perf.sh -- --num_warmup 5 --num_iterations 100 --input_mode random
#   bash benchmark_max_perf.sh --tegrastats-log ./logs/tegrastats.log -- --input_mode real --image_paths a.jpg b.jpg c.jpg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/Evo_depth/scripts/benchmark_evo_depth.py"

MODE_ID=""
PREFERRED_MODE_NAME="MAXN_SUPER"
TEGRASTATS_LOG=""
TEGRASTATS_INTERVAL=1000
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmark_max_perf.sh [options] -- [benchmark_args...]

Options:
  --mode-id <id>                 指定 nvpmodel 模式号（默认自动选择 MAXN_SUPER/MAXN）
  --mode-name <name>             优先模式名（默认 MAXN_SUPER）
  --python <bin>                 Python 可执行文件（默认 python，可用 PYTHON_BIN 覆盖）
  --tegrastats-log <path>        记录 tegrastats 输出到文件（可选）
  --tegrastats-interval <ms>     tegrastats 采样间隔毫秒（默认 1000）
  -h, --help                     显示帮助

benchmark_args:
  透传给 Evo_depth/scripts/benchmark_evo_depth.py 的参数，
  如 --num_warmup --num_iterations --input_mode --image_paths 等。
EOF
}

log() { echo "[benchmark_max_perf] $*"; }
warn() { echo "[benchmark_max_perf][WARN] $*"; }

parse_mode_id_by_name() {
  local name="$1"
  local out
  out="$(sudo nvpmodel -q --verbose 2>/dev/null || true)"
  if [[ -z "${out}" ]]; then
    return 1
  fi

  # 典型行：POWER_MODEL: ID=2 NAME=MAXN_SUPER
  local id
  id="$(echo "${out}" | sed -nE "s/.*POWER_MODEL: ID=([0-9]+) NAME=${name}.*/\\1/p" | head -n1)"
  if [[ -n "${id}" ]]; then
    echo "${id}"
    return 0
  fi
  return 1
}

detect_current_mode_id() {
  local out
  out="$(sudo nvpmodel -q --verbose 2>/dev/null || true)"
  # 兼容当前机器输出：Current mode...下一行仅数字
  local cur
  cur="$(echo "${out}" | awk '/Current mode:/ {getline; if ($1 ~ /^[0-9]+$/) {print $1; exit}}')"
  if [[ -n "${cur}" ]]; then
    echo "${cur}"
    return 0
  fi
  return 1
}

cleanup() {
  if [[ -n "${TEGRA_PID:-}" ]]; then
    log "停止 tegrastats (pid=${TEGRA_PID})"
    kill "${TEGRA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode-id)
      MODE_ID="$2"
      shift 2
      ;;
    --mode-name)
      PREFERRED_MODE_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --tegrastats-log)
      TEGRASTATS_LOG="$2"
      shift 2
      ;;
    --tegrastats-interval)
      TEGRASTATS_INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      BENCH_ARGS=("$@")
      break
      ;;
    *)
      warn "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "Benchmark script not found: ${BENCH_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${MODE_ID}" ]]; then
  if MODE_ID="$(parse_mode_id_by_name "${PREFERRED_MODE_NAME}")"; then
    log "检测到 ${PREFERRED_MODE_NAME} 模式，ID=${MODE_ID}"
  elif MODE_ID="$(parse_mode_id_by_name "MAXN")"; then
    log "未找到 ${PREFERRED_MODE_NAME}，回退到 MAXN，ID=${MODE_ID}"
  elif MODE_ID="$(detect_current_mode_id)"; then
    warn "无法解析 MAXN 模式，保留当前模式 ID=${MODE_ID}"
  else
    warn "无法自动识别模式，回退到 ID=0"
    MODE_ID="0"
  fi
fi

log "设置 nvpmodel 模式: ${MODE_ID}"
sudo nvpmodel -m "${MODE_ID}"

log "锁定 Jetson 时钟"
sudo jetson_clocks

log "当前性能状态"
sudo jetson_clocks --show

if [[ -n "${TEGRASTATS_LOG}" ]]; then
  mkdir -p "$(dirname "${TEGRASTATS_LOG}")"
  log "启动 tegrastats 记录: ${TEGRASTATS_LOG} (interval=${TEGRASTATS_INTERVAL}ms)"
  tegrastats --interval "${TEGRASTATS_INTERVAL}" >"${TEGRASTATS_LOG}" 2>&1 &
  TEGRA_PID=$!
fi

log "启动 benchmark"
"${PYTHON_BIN}" "${BENCH_SCRIPT}" "${BENCH_ARGS[@]}"

log "benchmark 运行完成"
