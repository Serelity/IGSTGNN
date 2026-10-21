# PASE-Net paper experiments: Sections 5.3 and 5.4

This entry uses the GW16 STATE_EXCHANGE architecture and original recipe. It
does not incorporate GW17 or change any previous main-result checkpoint. New
runs start from scratch in a separate directory. The FULL constructor and
forward are identical to GW16; the other arms change the factors below.

## Fixed experiment matrix

| Section | Arm | Change |
|---|---|---|
| 5.3 and 5.4 | FULL | PASE-Net, shared full-model reference |
| 5.3 | CONCAT_MLP | Replace coefficient generation with 128→16→2 GELU MLP; retain tanh and regional centering |
| 5.3 | REGIONAL_SOFTMAX | Replace tanh with regional softmax scaled by region size, then center |
| 5.4 | NO_EXCHANGE | Disable both exchange directions |
| 5.4 | UP_ONLY | Keep local-to-regional exchange only |
| 5.4 | DOWN_ONLY | Keep regional-to-local exchange only |
| 5.4 | ORIGIN_EXCHANGE | Read C0/L0 instead of current C/L for coefficients; current message values and U_h stay unchanged |
| 5.4 | NO_POSITION_BRANCH | Replace the exchange coefficient generator's explicit position projection with a learned shared vector; retain position elsewhere |
| 5.4 | NO_INTERIOR_CENTER | Remove only the projection after each local-state update; retain initial, downward-message and readout centering |

The coefficient generators contain 2064 (FULL/REGIONAL_SOFTMAX), 2098
(CONCAT_MLP), and 1568 (NO_POSITION_BRANCH) parameters. Reports also count full
registered, trainable and structurally active parameters. Disabled parameters
are not treated as active equal-capacity computation.

Each arm receives 100 complete joint-training epochs, batch 48, the GW16
optimizer/loss and learning-rate schedule, without early stopping. BEST is
selected using validation pooled MAE and includes stage zero. The final report
uses the original per-horizon metric implementation and equal mean of 12 leads;
it also records the pooled selection metric separately. Common initial tensors
and epoch sample orders match within each seed.

The initial run uses seed 2025. The predeclared repetitions are 2025, 2026 and
2027 for the entire matrix, not selected winning variants. This implementation
trains all nine arms, including FULL; it does not automatically import an older
FULL run. NO_INCIDENT/NO_REGION_LOSS and rank/sharing efficiency experiments are
plans, not implemented arms in this entry.

## Server setup and run

Use the existing `igstgnn` environment on a GPU node. Repository requirements
recommend Python 3.10/3.11 and specify torch 2.3.1. CUDA is not pinned. The
launcher records actual installed Python/PyTorch/CUDA, driver and GPU versions;
it never installs or upgrades packages.

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git clone --single-branch --branch research/pase-paper-experiments \
  https://github.com/Serelity/IGSTGNN.git PASE-paper-code
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh
```

The script resolves its own code directory and uses the existing dataset at
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa`.
The scheduler's working directory may remain `IGSTGNN-code`; its startup command
is the final Bash line above. Arms run sequentially on one GPU. No server
plotting package is required.

Useful overrides, set before the same launcher:

```bash
export PASE_DEVICE=cuda:0
export PASE_THREADS=3
export PASE_SEEDS="2025"                       # first complete matrix
export PASE_RUN_ROOT="$(pwd -P)/pase_paper_cc_core"  # optional stable result path
```

For the three predeclared repetitions, set `PASE_SEEDS="2025 2026 2027"` and
keep the same run root. Completed 2025 arms are reused, and missing seeds are
trained. The evaluation directory includes the seed set. If using a different
reporting arm subset, set a fresh `PASE_EVAL_DIR` rather than overwriting a
frozen cache.

## Interruption, environment and reports

Rerun the same command to resume from the last atomically committed epoch,
including optimizer, scheduler and random state. An interrupted, uncommitted
epoch is replayed. Keep the code, data, device/runtime and run root unchanged.
Do not change source files or pull new code into an active numerical run.

```bash
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh resume
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh env
```

For environment inspection on a login node without a GPU, use
`PASE_DEVICE=cpu bash PASE-paper-code/scripts/run_pase_paper_experiments.sh env`.
The CUDA build version will still be recorded, but confirm the actual GPU and
driver record from the allocated GPU node for the manuscript.

Default outputs (when PASE_RUN_ROOT is unset):

```text
PASE-paper-code/experiments/chronological_runs/pase_paper_cc_core/
  _environment/environment_*.json
  s2025/identity.json
  s2025/summary.json
  s2025/record.log
  s2025/<ARM>/history.json
  evaluation_val_s2025/report.md
  evaluation_val_s2025/parameters.csv
  evaluation_val_s2025/per_horizon.csv
```

The default run evaluates validation only. Use `evaluate` to finish an
interrupted validation evaluation; authenticated completed caches are reused
without new model forwards. The explicit `test` phase is permitted only after
all nine arms have completed 100 epochs for every requested seed. It freezes
all BEST roles before accessing test. Test results cannot select additional
variants, training budgets or checkpoints; these mechanism comparisons do not
constitute a new independent held-out confirmation.

```bash
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh evaluate
# Only after the full predeclared matrix is complete:
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh test
```

## Engineering checks

```bash
bash PASE-paper-code/scripts/run_pase_paper_experiments.sh check
```

The targeted CPU checks cover exact FULL equivalence, meaningful interventions,
parameter counts, transactional resume and evaluation contracts. GitHub Actions
also checks Python and Bash syntax. These checks are not GPU training results.
Optional direct `train --engineering` runs are explicitly marked and cannot be
reported as the formal 100-epoch experiment or evaluated on test.
