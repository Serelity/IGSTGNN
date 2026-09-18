# Incident-information negative controls

These controls test whether report-time-safe incident inputs add predictive information beyond the
traffic backbone. They retain the frozen chronological split, seed, optimizer, epoch order, model
initialization, validation selection metric, and training budget used by fixed A.

## Variants

- `traffic_only` runs the fixed architecture without passing `incident_data` to the model.
- `shuffled_incident` pairs each traffic window with another incident from the same split. It
  shuffles `distances` and `report_age_minutes`, preserves the window's `forecast_tod` and
  `forecast_dow`, and forbids self-matches and matches with the same `t0`.
- Associated-node metrics always use the true incident mask, including for shuffled inputs.

The shuffle is fixed by the run seed. Its mapping hash and integrity counts are stored under
`incident_intervention` in `summary.json`; shuffled validation source sample IDs are also stored in
`best_validation_predictions.npz`.

## Server checks

Run in the existing Conda environment from the repository root:

```bash
conda activate igstgnn

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 \
python -u experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_traffic_only_cuda_check_01 \
  --protocol experiments/chronological/incident_negative_controls_v1.json \
  --variant traffic_only \
  --device cuda:0 \
  --seed 2025 \
  --check

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 \
python -u experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_shuffled_incident_cuda_check_01 \
  --protocol experiments/chronological/incident_negative_controls_v1.json \
  --variant shuffled_incident \
  --device cuda:0 \
  --seed 2025 \
  --check
```

After both checks report `ENGINEERING_CHECK_PASS`, remove only `--check` and use new output
directories ending in `_s2025_full_01` for the two complete runs.

## Interpretation

Compare both controls with fixed A on all nodes, true associated nodes, non-associated nodes,
H1-H6, H7-H12, paired events, clustered natural days, and unique `t0` windows.

- A better than both controls supports incremental value from correctly paired incident reports.
- A better than `traffic_only` but not shuffled incidents suggests structural or nuisance-feature
  gains rather than incident-specific information.
- Similar results for all three indicate that the current incident branch has not extracted a
  reliable incident residual and should be redesigned before adding more response flexibility.
