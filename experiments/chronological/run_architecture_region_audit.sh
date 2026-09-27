#!/usr/bin/env bash
# Saved v12a statistics only. Invoke with bash from an allocated server node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12b_regions_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12b_regions_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12b_regions_01；重跑请改为 _02。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
LOG="$JOB/run.log"

if [[ "$ACTION" == _worker ]]; then
  PYTHON_BIN=$3
  SOURCE=$4
  trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
  export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
  printf 'Started: %s\nHost: %s\nWorker PID: %s\nPython: %s\nSource: %s\n' \
    "$(date -Is)" "$(hostname)" "$$" "$PYTHON_BIN" "$SOURCE"
  git log -1 --format='commit=%H'
  "$PYTHON_BIN" -c 'import sys, numpy; print("python:", sys.version, "numpy:", numpy.__version__, flush=True)'
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_architecture_regions.py' -v
  time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_architecture_regions.py" run \
    --data-dir ../data/chronological/Contra_Costa_v8_dev \
    --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01 \
    --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01 \
    --v12a-dir "$SOURCE" --output "$OUT"
  exit 0
fi

case "$ACTION" in
  start)
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    "$PYTHON_BIN" -c 'import numpy'
    SOURCE=${V12A_SOURCE:-experiments/chronological_runs/contra_v12a_architecture_01}
    if [[ ! -f "$SOURCE/summary.json" ]]; then
      printf '缺少 v12a 汇总：%s/summary.json\n' "$SOURCE" >&2
      exit 1
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" ]]; then
        printf '已存在，保留现场：%s\n请使用新的运行名，例如 contra_v12b_regions_02。\n' "$TARGET" >&2
        exit 1
      fi
    done
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    nohup bash "$SCRIPT_DIR/run_architecture_region_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$SOURCE" \
      > "$LOG" 2>&1 < /dev/null &
    RUN_PID=$!
    printf '%s\n' "$RUN_PID" > "$JOB/pid"
    printf 'v12b started on %s, worker PID %s\nLog: %s\n' "$(hostname)" "$RUN_PID" "$LOG"
    printf '查看进度：bash experiments/chronological/run_architecture_region_audit.sh status %s\n' "$RUN_NAME"
    printf '实时日志：tail -f "%s"\n' "$LOG"
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then
      printf '没有找到运行记录：%s\n' "$JOB" >&2
      exit 1
    fi
    RUN_HOST=$(< "$JOB/host")
    printf 'Run host: %s\nCurrent host: %s\n' "$RUN_HOST" "$(hostname)"
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ "$RUN_HOST" == "$(hostname)" && -f "$JOB/pid" ]]; then
      if ! ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")"; then
        echo '没有找到本机进程且没有退出码；请保留日志，检查平台任务状态。'
      fi
    else
      echo '该任务运行在另一主机；不要用当前主机的 PID 查询判断它是否结束。'
    fi
    if [[ -f "${OUT}.partial/progress.json" ]]; then
      cat "${OUT}.partial/progress.json"
    fi
    tail -n 20 "$LOG"
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    fi
    ;;
  report)
    python "$SCRIPT_DIR/audit_architecture_regions.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_architecture_region_audit.sh {start|status|report} [contra_v12b_regions_01]'
    echo '仅用 CPU/NumPy 分析已保存的 v12a 统计；先测试再运行。无需 PyTorch/GPU。'
    ;;
  *)
    echo "未知操作：$ACTION" >&2
    exit 2
    ;;
esac
