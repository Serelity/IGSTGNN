#!/usr/bin/env bash
# Invoke with bash. Start reserves a unique job directory before launching nohup.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

ACTION=${1:-help}
RUN_NAME=${2:-contra_v12a_architecture_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12a_architecture_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12a_architecture_01；重跑请改为 _02。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
LOG="$JOB/run.log"
DEVICE=${V12A_DEVICE:-cuda:0}

if [[ "$ACTION" == _worker ]]; then
  PYTHON_BIN=$3
  DEVICE=$4
  trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Python workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
  printf 'Started: %s\nHost: %s\nWorker PID: %s\nPython: %s\nDevice: %s\n' \
    "$(date -Is)" "$(hostname)" "$$" "$PYTHON_BIN" "$DEVICE"
  git log -1 --format='commit=%H'
  "$PYTHON_BIN" -c 'import sys, numpy, torch; print("python:", sys.version, "numpy:", numpy.__version__, "torch:", torch.__version__, flush=True)'
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_architecture_mechanisms.py' -v
  ARGS=(
    --data-dir ../data/chronological/Contra_Costa_v8_dev
    --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
    --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
    --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
    --device "$DEVICE"
  )
  echo 'Starting two-sample-per-cohort engineering check.'
  "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_architecture_mechanisms.py" "${ARGS[@]}" \
    --output "$JOB/check" --check
  echo 'Engineering check passed. Starting full TRAIN-ONLY mechanism audit.'
  time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_architecture_mechanisms.py" "${ARGS[@]}" --output "$OUT"
  "$PYTHON_BIN" "$SCRIPT_DIR/report_architecture_mechanisms.py" "$OUT/summary.json"
  exit 0
fi

case "$ACTION" in
  start)
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    "$PYTHON_BIN" -c 'import sys, torch; d=sys.argv[1]; assert not d.startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable: use an allocated GPU node and activate igstgnn"' "$DEVICE"
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" ]]; then
        printf '已存在，保留现场：%s\n请使用新的运行名，例如 contra_v12a_architecture_02。\n' "$TARGET" >&2
        exit 1
      fi
    done
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    nohup bash "$SCRIPT_DIR/run_architecture_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" \
      > "$LOG" 2>&1 < /dev/null &
    RUN_PID=$!
    printf '%s\n' "$RUN_PID" > "$JOB/pid"
    printf 'v12a started on %s, worker PID %s\nLog: %s\n' "$(hostname)" "$RUN_PID" "$LOG"
    printf '查看进度：bash experiments/chronological/run_architecture_audit.sh status %s\n' "$RUN_NAME"
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
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
      echo '如只显示表头且没有退出码，任务可能已被平台终止；保留日志后检查平台任务状态。'
    else
      echo '该任务运行在另一主机；不要用当前主机的 PID 查询判断它是否结束。'
    fi
    tail -n 18 "$LOG"
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    fi
    ;;
  report)
    python "$SCRIPT_DIR/report_architecture_mechanisms.py" "$OUT/summary.json"
    ;;
  *)
    echo '用法：bash experiments/chronological/run_architecture_audit.sh {start|status|report} [contra_v12a_architecture_01]'
    echo 'start 自动执行单元测试、小样本工程检查，通过后运行完整训练集审计。默认 cuda:0。'
    ;;
esac
