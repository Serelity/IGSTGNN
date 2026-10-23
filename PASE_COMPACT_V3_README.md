# PASE-Net: compact MLP version (v3)

The current PASE model removes only the exchange scorer's explicit position
input. Position remains in historical encoding, origin conditions and the fixed
regional partition. Persistent local centering, incident inputs, synchronous
bidirectional updates, shared transitions and losses remain unchanged.

The scorer is `[condition, lifted normalized regional state, normalized local
state] -> Linear(96,16) -> GELU -> Linear(16,2) -> tanh -> regional centering`.
On Contra Costa it has 1,586 scorer parameters and 73,357 total parameters.
This is actual column removal, rather than retaining 512 weights on zero inputs.

## Only two new training runs

| Role | Source / budget |
|---|---|
| FULL | Existing MLP v2 `NO_POSITION_INPUT` validation BEST, fixed 100-epoch source |
| UP_ONLY | Compact scorer, only local to region messages; 100 fresh epochs |
| DOWN_ONLY | Compact scorer, only region to local messages; 100 fresh epochs |

Seed is 2025. No other datasets, seeds, centering variants or mechanism searches
are launched. Each one-way arm starts from the same seed-2025 scratch tensors
with the zero-position columns removed; neither starts from the trained FULL.
Training and BEST selection follow the existing recipe, including stage zero.
The shared tensors, sample order, and following construction RNG are preserved.

FULL comes from the completed `NO_POSITION_INPUT` run (the supplied run selected
epoch 99). Promotion authenticates the source identity, fixed budget, selected
checkpoint and validation cache; it does not hard-code a preferred epoch.
Only first-linear position columns are discarded. All remaining tensors are
copied unchanged. The export has no optimizer and is not a resume checkpoint.

Compact GEMM shapes can change floating-point accumulation. A FULL-only fixed
BEST validation is therefore required before training either one-way arm.
The replay uses the same targets, station and sample axes and bounded numerical
metric tolerance, without choosing another checkpoint or changing the budget.
Converted predictions are actually evaluated; source metrics are references,
not relabeled compact-model results. Old runs and caches remain immutable.

## Server

Working directory:
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code`

```bash
conda activate igstgnn
git clone --single-branch --branch research/pase-compact-v3 \
  https://github.com/Serelity/IGSTGNN.git PASE-compact-code
bash PASE-compact-code/scripts/run_pase_compact_v3.sh
```

For the job submission form, use that working directory, environment `igstgnn`,
one GPU, and startup command:

```bash
bash PASE-compact-code/scripts/run_pase_compact_v3.sh
```

Default completed source:
`PASE-MLP-code/experiments/chronological_runs/pase_mlp_v2/Contra_Costa`
(under the working directory). It must contain `s2025` and
`evaluation_ablation_val_s2025`. Override with `PASE_COMPACT_SOURCE_ROOT` if moved.

Default new root:
`PASE-compact-code/experiments/chronological_runs/pase_compact_v3/Contra_Costa`.
Override using `PASE_COMPACT_RUN_ROOT`. Device defaults to `cuda:0`, threads to 3;
use `PASE_COMPACT_DEVICE`, `PASE_COMPACT_THREADS`, or `PASE_COMPACT_DATA_DIR`
for an explicit change. Old `PASE_V2_ARMS` and other old launch variables are ignored.

The launcher tests code, promotes FULL without training, replays compact FULL,
trains the two one-way arms sequentially, then evaluates the three BESTs on val.
It does not evaluate test or measure inference speed.

After interruption, use the same command or append `resume`. It restores the
last committed epoch, Adam, scheduler, RNG and sample order; completed arms are
skipped. A completed FULL-only replay is reused. If initialization was killed
before identity.json was committed, preserve the partial directory and choose
a fresh `PASE_COMPACT_RUN_ROOT` (no guessing or deleting existing results).

Results:

```bash
export PASE_COMPACT_RESULTS="$PWD/PASE-compact-code/experiments/chronological_runs/pase_compact_v3/Contra_Costa"
cat "$PASE_COMPACT_RESULTS/evaluation_ablation_val_s2025/report.md"
cat "$PASE_COMPACT_RESULTS/evaluation_ablation_val_s2025/selected_horizons.csv"
```

`promoted_full/report.md` describes conversion only; the completed validation
report above contains actual compact-model predictions. A one-seed development
comparison does not establish statistical significance or independent holdout
confirmation. Existing low-rank and explicit-position results retain their
original version labels, including the old three-dataset test table and figure.

## Verification

Numerical tests run on GitHub Actions/server, not on the local workstation:

```bash
bash PASE-compact-code/scripts/run_pase_compact_v3.sh check
```

The new package is isolated. No numerical sources in the old MLP v2, original
paper experiment or GW16 package are changed.
