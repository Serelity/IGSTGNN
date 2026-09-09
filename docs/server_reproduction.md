# Server Reproduction

This page documents the earlier released-code baseline at `558ab01`. On the
`fix/paper-alignment` branch, follow [Paper Alignment](paper_alignment.md): the
model and sensor inputs changed, run directories are labeled `paper_aligned_v1`,
and existing checkpoints must not be used to resume the corrected model.

These presets finish the remaining single-seed baselines without changing the
IGSTGNN model, data split, loss, optimizer, learning-rate schedule, or curriculum.
Keep the existing Alameda result as a reference.

| Dataset | Batch size | Seed | Max epochs | Patience |
| --- | ---: | ---: | ---: | ---: |
| Alameda | 48 | 2025 | 100 | 20 |
| Contra_Costa | 48 | 2025 | 100 | 20 |
| Orange | 24 | 2025 | 100 | 20 |

Incident information and sensor metadata are enabled. Warm-up remains 30 epochs
and the curriculum interval remains 3 epochs. Other hyperparameters retain the
existing defaults in `experiments/IGSTGNN/main.py`.

## Submit Contra_Costa First

Use a login/remote terminal. Reuse the working `igstgnn` environment; do not
reinstall dependencies or train on the login node.

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin main
conda activate igstgnn
sinfo -p gpu_v100 -o "%P %l %G"
sbatch --export=ALL --job-name=igstgnn_Contra_Costa experiments/IGSTGNN/run.slurm Contra_Costa
```

The script inherits the active Conda environment at submission time and uses its
Python executable on the allocated node. An inactive/missing environment is a
hard error. The default request is one V100, 3 CPU cores, 32 GiB host memory,
and an 8-hour wall-time limit. The platform may adjust CPU allocation per GPU.
The queue and submission format follow the SEU GPU manual, pages 19 and 32.
Do not use `normal_test`: the manual says its tasks are cleared every 30 minutes.

The data must already be reachable as `data/xtraffic/Contra_Costa/` (and later
`data/xtraffic/Orange/`). Existing working data-directory links need no changes.
The launcher reuses complete split files, creates them when all are absent, and
stops if only some are present. Split creation runs inside the batch allocation,
so do not start a separate large preprocessing process on the login node.

## Check the Run

`sbatch` prints a job ID. Use that exact ID in the commands below; replace
`JOB_ID` rather than cancelling all of your jobs.

```bash
squeue -u "$(whoami)"
tail -f slurm-igstgnn_Contra_Costa-JOB_ID.out
tail -n 80 slurm-igstgnn_Contra_Costa-JOB_ID.err
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

Training progress bars are written to `.err`, which does not by itself indicate
failure. Check for tracebacks, the scheduler exit status, and the final
`Average Test MAE` line. Disconnecting your terminal does not stop a batch job;
Ctrl+C on `tail -f` only stops log viewing. To stop the job itself:

```bash
scancel JOB_ID
```

Keep the `.out` and `.err` files: `.out` includes the environment and source
revision before training, as well as the model's log-directory path. Model logs
and the best checkpoint remain in the existing location:

```text
experiments/igstgnn/Contra_Costa_igstgnn_s2025_<timestamp>/
    record_igstgnn_s2025_<timestamp>.log
    final_model_s2025.pt
```

The checkpoint contains the validation-best model weights, not a full optimizer
resume state. A timeout is not an exact-resume checkpoint. Review the metrics and
training curve before moving on to Orange; a successful exit alone is not a
successful numerical reproduction.

## Submit Orange After Reviewing Contra_Costa

Orange has more nodes and batches, so do not assume Alameda's measured runtime
applies. The command below requests a conservative 36-hour ceiling, not a
measured runtime or a requirement to train for 36 hours. Check the queue's
current time limit with `sinfo` first. If it does not allow 36 hours, confirm a
suitable allocation before starting instead of knowingly cutting training short.

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
sinfo -p gpu_v100 -o "%P %l %G"
sbatch --export=ALL --job-name=igstgnn_Orange --time=1-12:00:00 experiments/IGSTGNN/run.slurm Orange
```

Use `slurm-igstgnn_Orange-JOB_ID.out` / `.err` to monitor this job. The model
directory uses the same pattern as above with `Orange` in place of `Contra_Costa`.
Do not update code or dependencies while the baseline jobs are queued or running.
Submit the cities separately rather than concurrently or as an automatic chain.

The allocation is released when the process exits, including early stopping or
failure. Requested wall time is a safety ceiling; actual charges depend on the
platform's allocation duration and billing policy. No local or remote GPU job
is started by preparing or downloading these scripts.

## Orange First-Batch CUDA OOM

The original graph convolution broadcasts each sample's dense graph across
time steps before matrix multiplication. On PyTorch 2.3.1, this can materialize
and retain repeated graph storage for backward. For Orange at batch 24, a
single FP32 `[24, 6, 990, 2970]` expansion occupies about 1.58 GiB.

The graph convolution now uses an equivalent `einsum` contraction without
copying the graph across time. This does not change model parameters, checkpoint
keys, batch size, data splits, loss, or training schedules. Floating-point
results need not be bit-identical; tests compare outputs and gradients against
the original expression, including a small five-layer model with incident and
sensor inputs. A regression test rejects retained time-expanded graph storage.

These local tests do not establish full-dataset V100 memory usage or final
forecasting accuracy. Keep completed Alameda and Contra_Costa results; update
the code before submitting a new Orange-only task:

```bash
git pull --ff-only origin main
```

For an existing platform submission workflow, request one V100 and use this
startup script inside the allocated GPU task:

```bash
set -e
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
set -o pipefail
bash experiments/IGSTGNN/run.sh Orange 2>&1 | tee "Orange_$(date +%Y%m%d_%H%M%S).log"
```

The shared-directory log persists after the task exits and includes the full
CUDA error if another failure occurs. `pipefail` preserves a training failure
as a nonzero task exit status even when `tee` succeeds. Existing complete split
files are reused. This shell script relies on the platform task for its lifetime;
it does not detach training from an ordinary SSH session.
