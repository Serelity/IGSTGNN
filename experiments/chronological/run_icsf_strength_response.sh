#!/usr/bin/env bash
# Foreground platform workflow: engineering check before full fixed-weight diagnosis.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12d_strength_response_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12d_strength_response_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12d_strength_response_01。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run)
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12D_DEVICE:-cuda:0}
    "$PYTHON_BIN" -c 'import sys,torch; assert not sys.argv[1].startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable: use an allocated GPU and igstgnn environment"' "$DEVICE"
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" ]]; then
        printf '已存在，保留现场并使用新运行名：%s\n' "$TARGET" >&2
        exit 1
      fi
    done
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_icsf_strength_response.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$DEVICE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_icsf_strength_response*.py' -v
    ARGS=(
      --data-dir ../data/chronological/Contra_Costa_v8_dev
      --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
      --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
      --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
      --device "$DEVICE"
    )
    "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_icsf_strength_response.py" run "${ARGS[@]}" --output "$JOB/check" --check
    echo 'Engineering check passed; starting all frozen scan points and fit-only gradients.'
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_icsf_strength_response.py" run "${ARGS[@]}" --output "$OUT"
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
    if [[ -f "$JOB/run.log" ]]; then tail -n 15 "$JOB/run.log"; fi
    if [[ -f "$OUT/summary.json" ]]; then printf 'Summary: %s\n' "$OUT/summary.json"; fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then
      echo '尚无完整报告；请用 status 查看日志，保留 .partial。' >&2
      exit 1
    fi
    python "$SCRIPT_DIR/audit_icsf_strength_response.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_icsf_strength_response.sh {run|status|report} [contra_v12d_strength_response_01]'
    echo 'run 为前台平台任务入口，默认 CUDA；先测试和真实 checkpoint 检查，再完成固定权重诊断。'
    ;;
  *) echo "未知操作：$ACTION" >&2; exit 2 ;;
esac
