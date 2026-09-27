#!/usr/bin/env bash
# Gate-only GPU training, with a real-checkpoint engineering check first.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12c_strength_gate_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12c_strength_gate_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12c_strength_gate_01；重跑请改为 _02。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
LOG="$JOB/run.log"

if [[ "$ACTION" == _worker ]]; then
  PYTHON_BIN=$3
  DEVICE=$4
  RECOVERY_SOURCE=${5:-}
  trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
  printf 'Started: %s\nHost: %s\nWorker PID: %s\nPython: %s\nDevice: %s\n' \
    "$(date -Is)" "$(hostname)" "$$" "$PYTHON_BIN" "$DEVICE"
  git log -1 --format='commit=%H'
  "$PYTHON_BIN" -c 'import sys,numpy,torch; print("python:",sys.version,"numpy:",numpy.__version__,"torch:",torch.__version__,flush=True)'
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_incident_strength_gate*.py' -v
  ARGS=(
    --data-dir ../data/chronological/Contra_Costa_v8_dev
    --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
    --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
    --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
    --device "$DEVICE"
  )
  echo 'Checking identity initialization, real-data gradients and selection/report workflow.'
  "$PYTHON_BIN" -u "$SCRIPT_DIR/train_incident_strength_gate.py" run "${ARGS[@]}" --output "$JOB/check" --check
  echo 'Engineering check passed. Starting fixed-budget scalar/node adapters, three seeds.'
  if [[ -n "$RECOVERY_SOURCE" ]]; then
    ARGS+=(--resume-from "$RECOVERY_SOURCE")
    printf 'Recovery source (read-only): %s\n' "$RECOVERY_SOURCE"
  fi
  time "$PYTHON_BIN" -u "$SCRIPT_DIR/train_incident_strength_gate.py" run "${ARGS[@]}" --output "$OUT"
  exit 0
fi

case "$ACTION" in
  start|resume)
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12C_DEVICE:-cuda:0}
    "$PYTHON_BIN" -c 'import sys,torch; d=sys.argv[1]; assert not d.startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable: activate igstgnn on an allocated GPU node"' "$DEVICE"
    RECOVERY_SOURCE=''
    if [[ "$ACTION" == resume ]]; then
      SOURCE_NAME=${3:-}
      if [[ ! "$SOURCE_NAME" =~ ^contra_v12c_strength_gate_[A-Za-z0-9_-]+$ || "$SOURCE_NAME" == "$RUN_NAME" ]]; then
        echo '恢复用法：resume 新运行名 原运行名（两者必须不同）。' >&2
        exit 2
      fi
      RECOVERY_SOURCE="$REPO_DIR/experiments/chronological_runs/${SOURCE_NAME}.partial"
      if [[ ! -f "$RECOVERY_SOURCE/effective_plan.json" ]]; then
        printf '缺少原运行恢复清单：%s/effective_plan.json\n' "$RECOVERY_SOURCE" >&2
        exit 1
      fi
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" ]]; then
        printf '已存在，保留现场：%s\n请使用新运行名，例如 contra_v12c_strength_gate_02。\n' "$TARGET" >&2
        exit 1
      fi
    done
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    nohup bash "$SCRIPT_DIR/run_incident_strength_gate.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$RECOVERY_SOURCE" \
      > "$LOG" 2>&1 < /dev/null &
    RUN_PID=$!
    printf '%s\n' "$RUN_PID" > "$JOB/pid"
    printf 'v12c started on %s, worker PID %s\nLog: %s\n' "$(hostname)" "$RUN_PID" "$LOG"
    printf '查看进度：bash experiments/chronological/run_incident_strength_gate.sh status %s\n' "$RUN_NAME"
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
        echo '本机进程不存在且未记录退出码；请保留日志并检查平台任务状态。'
      fi
    else
      echo '任务在另一台主机；不要用本机 PID 查询判断它是否结束。'
    fi
    tail -n 20 "$LOG"
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    elif [[ -d "${OUT}.partial" ]]; then
      printf '运行未完成；已保存数据：%s.partial\n使用 report 查看部分进度。\n' "$OUT"
    fi
    ;;
  report)
    python "$SCRIPT_DIR/train_incident_strength_gate.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_incident_strength_gate.sh {start|status|report} [contra_v12c_strength_gate_01]'
    echo '恢复：bash experiments/chronological/run_incident_strength_gate.sh resume contra_v12c_strength_gate_02 contra_v12c_strength_gate_01'
    echo '默认 cuda:0；先单元测试和小样本梯度检查，再执行两种门控、三个种子、每次 12 epoch。'
    ;;
  *)
    echo "未知操作：$ACTION" >&2
    exit 2
    ;;
esac
