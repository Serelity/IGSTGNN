#!/usr/bin/env bash
# Foreground paired node-gate experiment with checked recovery into a new run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12e_regional_gate_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12e_regional_gate_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12e_regional_gate_01。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run|resume)
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12E_DEVICE:-cuda:0}
    "$PYTHON_BIN" -c 'import sys,torch; assert not sys.argv[1].startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable: activate igstgnn on an allocated GPU"' "$DEVICE"
    SOURCE=''
    if [[ "$ACTION" == resume ]]; then
      SOURCE_NAME=${3:-}
      if [[ ! "$SOURCE_NAME" =~ ^contra_v12e_regional_gate_[A-Za-z0-9_-]+$ || "$SOURCE_NAME" == "$RUN_NAME" ]]; then
        echo '恢复用法：resume 新运行名 原运行名；两者须不同。' >&2
        exit 2
      fi
      SOURCE="$REPO_DIR/experiments/chronological_runs/${SOURCE_NAME}.partial"
      if [[ ! -f "$SOURCE/run_identity.json" ]]; then
        echo "缺少恢复身份文件：$SOURCE/run_identity.json" >&2
        exit 1
      fi
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" ]]; then
        printf '已存在，保留现场并使用新运行名：%s\n' "$TARGET" >&2
        exit 1
      fi
    done
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_regional_gate_objective.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$SOURCE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    SOURCE=${5:-}
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$DEVICE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_regional_gate_objective*.py' -v
    ARGS=(
      --data-dir ../data/chronological/Contra_Costa_v8_dev
      --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
      --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
      --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
      --device "$DEVICE"
    )
    "$PYTHON_BIN" -u "$SCRIPT_DIR/train_regional_gate_objective.py" run "${ARGS[@]}" --output "$JOB/check" --check
    echo 'Engineering check passed; starting paired global/regional node gates, three seeds.'
    if [[ -n "$SOURCE" ]]; then ARGS+=(--resume-from "$SOURCE"); fi
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/train_regional_gate_objective.py" run "${ARGS[@]}" --output "$OUT"
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then echo "没有运行记录：$JOB" >&2; exit 1; fi
    printf 'Run host: %s\nCurrent host: %s\n' "$(< "$JOB/host")" "$(hostname)"
    if [[ -f "$JOB/slurm_job_id" ]]; then printf 'Slurm job: %s\n' "$(< "$JOB/slurm_job_id")"; fi
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    fi
    if [[ -f "$JOB/run.log" ]]; then tail -n 20 "$JOB/run.log"; fi
    if [[ -f "$OUT/summary.json" ]]; then printf 'Summary: %s\n' "$OUT/summary.json"; fi
    ;;
  report)
    python "$SCRIPT_DIR/train_regional_gate_objective.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_regional_gate_objective.sh {run|status|report} [contra_v12e_regional_gate_01]'
    echo '恢复：bash experiments/chronological/run_regional_gate_objective.sh resume contra_v12e_regional_gate_02 contra_v12e_regional_gate_01'
    echo 'run/resume 均为前台入口；先工程检查，再六组节点门控拟合。'
    ;;
  *) echo "未知操作：$ACTION" >&2; exit 2 ;;
esac
