#!/usr/bin/env bash
# Foreground v12k training; each trajectory supplies two selected endpoints.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12k_objective_alignment_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12k_objective_alignment_[A-Za-z0-9_-]+$ ]]; then
  echo '运行名须形如 contra_v12k_objective_alignment_01。' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run|resume)
    SOURCE=''
    if [[ "$ACTION" == resume ]]; then
      SOURCE_NAME=${3:-}
      if [[ ! "$SOURCE_NAME" =~ ^contra_v12k_objective_alignment_[A-Za-z0-9_-]+$ || "$SOURCE_NAME" == "$RUN_NAME" ]]; then
        echo '恢复用法：resume 新运行名 原运行名；两者须不同。' >&2
        exit 2
      fi
      SOURCE="$REPO_DIR/experiments/chronological_runs/${SOURCE_NAME}.partial"
      if [[ -L "$SOURCE" || ! -f "$SOURCE/run_identity.json" || -L "$SOURCE/run_identity.json" ]]; then
        echo "缺少正常恢复身份文件，或来源为符号链接：$SOURCE/run_identity.json" >&2
        exit 1
      fi
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf '已存在，保留现场并使用新运行名：%s\n' "$TARGET" >&2
        exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12K_DEVICE:-cuda:0}
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_vector_objective_alignment.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$SOURCE" 2>&1 | tee "$JOB/run.log"
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
    "$PYTHON_BIN" -c 'import sys,torch,numpy; print("Torch:",torch.__version__,"NumPy:",numpy.__version__); device=torch.device(sys.argv[1]); torch.empty(1, device=device); assert device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable: use an allocated GPU and activate igstgnn"' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_vector_objective_alignment*.py' -v
    ARGS=(
      --data-dir ../data/chronological/Contra_Costa_v8_dev
      --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
      --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
      --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
      --device "$DEVICE"
    )
    "$PYTHON_BIN" -u "$SCRIPT_DIR/train_vector_objective_alignment.py" run "${ARGS[@]}" --output "$JOB/check" --check
    echo 'Engineering check passed; starting twelve fits, two selectors per trajectory (24 endpoints, 144 epochs).'
    if [[ -n "$SOURCE" ]]; then ARGS+=(--resume-from "$SOURCE"); fi
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/train_vector_objective_alignment.py" run "${ARGS[@]}" --output "$OUT"
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
      echo '尚无完整 summary.json；请根据退出码和日志确认运行状态。'
      if [[ -d "${OUT}.partial" ]]; then printf 'Partial: %s\n' "${OUT}.partial"; fi
    fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then
      printf '尚无完整结果：%s\n先用 status 查看日志；中断后恢复到新运行名。\n' "$OUT/summary.json" >&2
      exit 1
    fi
    python "$SCRIPT_DIR/train_vector_objective_alignment.py" report "$OUT/summary.json"
    ;;
  help)
    echo '用法：bash experiments/chronological/run_vector_objective_alignment.sh {run|status|report} [contra_v12k_objective_alignment_01]'
    echo '恢复：bash experiments/chronological/run_vector_objective_alignment.sh resume contra_v12k_objective_alignment_02 contra_v12k_objective_alignment_01'
    echo 'run/resume 为前台入口；先工程检查，再十二次拟合。两种选择规则共用每条训练轨迹。'
    ;;
  *) echo "未知操作：$ACTION" >&2; exit 2 ;;
esac
