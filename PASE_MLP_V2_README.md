# PASE-Net: MLP version

This is the selected MLP implementation of PASE-Net. The original low-rank
version and its immutable experiments remain in their original packages. The
new default FULL is numerically identical to the completed paper experiment's
CONCAT_MLP arm. It preserves the region/local decomposition, synchronous
bidirectional messages, persistent local centering, shared horizon transition,
incident conditions, last-observation anchor and training recipe.

At each node and forecast step, concatenate position, origin/incident/lead
condition, lifted normalized regional state and normalized local state:

```text
[p_i, U_h,i, S LN(C_h-1)_i, LN(L_h-1)_i]  (128 dimensions)
  -> Linear(128,16) -> GELU -> Linear(16,2) -> tanh -> regional centering Q
  -> two exchange coefficients (local-to-regional, regional-to-local)
```

The two outputs share the 16-dimensional hidden layer. The generator contains
2098 parameters; the Contra Costa complete model contains 73869 (+34 relative
to the previous model). Centering follows tanh, so final coefficients should
not be described as probabilities or as strictly bounded to [-1,1].

## Recorded selection evidence

The user-provided completed Contra Costa seed2025 validation report selected
the following BESTs after 100 epochs per arm. These are validation results,
not new test results or results for other counties:

| Generator | MAE | RMSE | MAPE % | BEST epoch |
|---|---:|---:|---:|---:|
| Previous low-rank | 12.529133 | 21.583389 | 17.448497 | 99 |
| Selected MLP | 12.474597 | 21.512876 | 17.090770 | 100 |
| Regional Softmax | 12.538162 | 21.589514 | 17.124463 | 98 |

MLP's MAE improvement is approximately 0.44%; this is one dataset and one seed.
It does not establish the necessity of every condition or a speed advantage.
Previously reported Alameda/Orange and held-out test results belong to the
previous low-rank version; do not relabel them as MLP results.

## Reuse the completed Contra Costa model

Use a separate checkout so original run/source fingerprints remain intact:

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git clone --single-branch --branch research/pase-mlp-v2 \
  https://github.com/Serelity/IGSTGNN.git PASE-MLP-code
bash PASE-MLP-code/scripts/run_pase_mlp_v2.sh
```

The default action authenticates the completed source identity, validation BEST
and evaluation cache, then exports a separate model.pt plus provenance metadata
and a report. It performs zero new training, model forwards or test evaluation.
It does not fabricate an optimizer state or a new 100-epoch training history.
Repeating promotion authenticates/reuses the same artifact; it cannot select a
different checkpoint in place.

Default source:
`IGSTGNN-code/PASE-paper-code/experiments/chronological_runs/pase_paper_cc_core`.
If moved, set `PASE_MLP_SOURCE_ROOT` to that existing root. Default output:
`PASE-MLP-code/experiments/chronological_runs/pase_mlp_v2/Contra_Costa/promoted_full`.
`load_promoted(path)` returns weights, an explicit source BEST role and metadata;
the normal model interface is `build_model(data, config, arm='FULL', device=...)`.

## Three-dataset entry and matching ablations

The entry supports Contra_Costa (496 nodes), Alameda (521) and Orange (990),
using their existing certified release adapters and vocabulary sizes. Existing
Contra Costa MLP weights are reused; other counties require their own MLP
training. No other-county MLP run was performed during this code update.

```bash
# Explicitly start one county's FULL model, seed2025, fixed 100 epochs:
PASE_DATASET=Alameda bash PASE-MLP-code/scripts/run_pase_mlp_v2.sh train
PASE_DATASET=Orange bash PASE-MLP-code/scripts/run_pase_mlp_v2.sh train
# Repeat the corresponding command with resume after interruption.
```

These commands run one dataset at a time. Set PASE_DEVICE when allocating
separate GPUs; do not run the same output directory twice. No three-seed job is
launched. Changing PASE_V2_RUN_ROOT changes the experiment location; keep it
stable for resume. The default train action requests FULL only, not all ablations.

Matching MLP arms are FULL, NO_EXCHANGE, UP_ONLY, DOWN_ONLY, ORIGIN_EXCHANGE,
NO_POSITION_INPUT and NO_INTERIOR_CENTER. An explicit PASE_V2_ARMS list selects
additional from-scratch component experiments under the same 100-epoch budget.
FULL may reference the promoted completed model without training it again.

NO_POSITION_INPUT zeros only the explicit 32-dimensional position block of the
MLP input; position remains in the encoder, conditions and fixed partition.
Registered parameter count stays 2098, with 512 input weights structurally
inactive. One-way variants share the entire hidden layer, so only 17 scorer
parameters are inactive in the unused output, not half the MLP.

Old low-rank single-direction/position/state/centering results are not MLP
ablations. NO_EXCHANGE has an equivalence contract with the previous inactive
exchange function; actual legacy evidence reuse needs its own authenticated
provenance and is not silently imported by this launcher. Do not rerun a
complete matrix simply to rename previous evidence.

Evaluation defaults to validation, includes native per-lead metrics and separate
pooled selection metrics, and freezes selected BESTs before inference. An
explicit direct evaluator test request requires completed 100-epoch BESTs for
every requested arm; optional component arms do not block evaluating FULL.
Test results cannot change the selected architecture, budget or checkpoint,
and this development update is not an independent held-out confirmation.

Changing the reporting arm list requires a fresh PASE_V2_EVAL_DIR, because an
existing evaluation cache is immutable. The promotion artifact itself is an
inference export; it cannot be passed as an optimizer-resume checkpoint.

## Verification and manuscript migration

```bash
bash PASE-MLP-code/scripts/run_pase_mlp_v2.sh check
```

Tests cover exact old-CONCAT_MLP equivalence, meaningful interventions,
parameter accounting, transactional training/resume, source-artifact promotion
and evaluation contracts. Local Python execution is intentionally not used;
numerical checks run on remote CPU CI and on the user's server.

The method equation and architecture panel must now show concatenation and
the shared MLP instead of rank8 position/state dot-product scoring. Existing
paper results and the approved figure are preserved as old-version artifacts
until the manuscript is migrated with correctly versioned evidence. The generic
MLP itself should not be presented as a novel module.
