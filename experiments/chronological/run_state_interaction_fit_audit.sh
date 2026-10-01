#!/usr/bin/env bash
# Foreground inference-only full-fit diagnostic of completed v12f models.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12i_full_fit_audit_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12i_full_fit_audit_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12i_full_fit_audit_01。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run)
    SOURCE_NAME=${3:-contra_v12f_state_interaction_01}
    if [[ ! "$SOURCE_NAME" =~ ^contra_v12f_state_interaction_[A-Za-z0-9_-]+$ ]]; then
      echo '来源运行名须形如 contra_v12f_state_interaction_01，不能使用 partial。' >&2
      exit 2
    fi
    SOURCE="$REPO_DIR/experiments/chronological_runs/$SOURCE_NAME"
    if [[ ! -d "$SOURCE" || -L "$SOURCE" ]]; then
      printf '缺少完整 v12f 来源目录，或来源为符号链接：%s\n' "$SOURCE" >&2
      exit 1
    fi
    SOURCE_FILES=(summary.json run_identity.json eligibility.json)
    for ARM in strength state_vector interaction_vector; do
      for SEED in 2025 2026 2027; do
        SOURCE_FILES+=("${ARM}_s${SEED}/selected_gate.pt")
      done
    done
    for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
      if [[ ! -f "$SOURCE/$SOURCE_FILE" ]]; then
        printf '缺少完整 v12f 结果：%s\n' "$SOURCE/$SOURCE_FILE" >&2
        exit 1
      fi
    done
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf '已存在，保留现场并使用新运行名：%s\n' "$TARGET" >&2
        exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12I_DEVICE:-cuda:0}
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_state_interaction_fit_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$SOURCE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    SOURCE=$5
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: %s\nSource: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$DEVICE" "$SOURCE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import sys,numpy,torch; assert not sys.argv[1].startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable: activate igstgnn on an allocated GPU"; print("NumPy:", numpy.__version__, "Torch:", torch.__version__, "Device:", sys.argv[1])' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_state_interaction_fit*.py' -v
    echo 'Preflight passed; evaluating A and all nine saved selected adapters. No training or epoch reselection.'
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_state_interaction_fit.py" run \
      --data-dir ../data/chronological/Contra_Costa_v8_dev \
      --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01 \
      --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01 \
      --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt \
      --source-dir "$SOURCE" \
      --output "$OUT" \
      --device "$DEVICE"
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
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    else
      printf '尚无完整 summary.json；请根据退出码和日志确认运行状态。\n'
      if [[ -d "${OUT}.partial" ]]; then printf 'Partial: %s\n' "${OUT}.partial"; fi
    fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then
      printf '尚无完整结果：%s\n先用 status 查看退出码和日志；中断后请使用新运行名重新评估。\n' "$OUT/summary.json" >&2
      exit 1
    fi
    python "$SCRIPT_DIR/audit_state_interaction_fit.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_state_interaction_fit_audit.sh {run|status|report} [contra_v12i_full_fit_audit_01]'
    echo '指定来源：run contra_v12i_full_fit_audit_02 contra_v12f_state_interaction_01'
    echo 'run 为前台固定模型推理入口，默认 cuda:0（可用 V12I_DEVICE 指定）；不训练、不重新选择 epoch。'
    echo '完整 v12f 来源保持只读；中断后使用新运行名重新评估，没有恢复或 check 训练入口。'
    ;;
  *) echo "未知操作：$ACTION" >&2; exit 2 ;;
esac
