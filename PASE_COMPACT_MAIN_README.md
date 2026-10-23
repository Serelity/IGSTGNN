# Current PASE-Net three-dataset main results

Only compact FULL runs. No original IGSTGNN, other baselines, extra seeds or
ablation variants are trained. The model and locked per-dataset recipe are
unchanged: seed 2025, 100 epochs, validation pooled-MAE BEST including stage zero.

* Contra Costa: authenticate and evaluate the existing compact FULL on test.
  Do not retrain, choose a new epoch, or modify its old numerical sources.
* Alameda / Orange: train the same compact FULL from scratch, validate BEST,
  then evaluate that frozen BEST on test. Resume restores Adam, scheduler, RNG
  and sample order from the last committed epoch. Completed training is skipped.

Use the existing `igstgnn` environment and one GPU. The default runs datasets
sequentially. A scheduler job should run in:
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code`

```bash
conda activate igstgnn
git clone --single-branch --branch research/pase-compact-v3 \
  https://github.com/Serelity/IGSTGNN.git PASE-main-code
bash PASE-main-code/scripts/run_pase_compact_main.sh
```

For job submission, activate/select `igstgnn` and submit the last `bash` line.
After interruption use exactly the same command and keep the same run root,
software, GPU type and threads. Do not fetch changed numerical sources during
training. Partial initialization without identity.json is preserved and rejected;
select a fresh output root in that case, rather than deleting evidence.

Default data: `/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic`.
Default existing CC root:
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code/PASE-compact-code/experiments/chronological_runs/pase_compact_v3/Contra_Costa`.
That source's referenced promoted artifact must remain at its certified path.
The old compact source hashes must match; this change adds a separate package
without changing the original compact package or its dependencies.

Overrides: `PASE_MAIN_DATA_ROOT`, `PASE_MAIN_CC_ROOT`, `PASE_MAIN_RUN_ROOT`,
`PASE_MAIN_DEVICE` (default cuda:0), `PASE_MAIN_THREADS` (default 3).
Run only one dataset by appending `Contra_Costa`, `Alameda` or `Orange`.
If scheduling Alameda and Orange separately on two GPUs, use the same output
root and separate dataset arguments. Never run the same dataset concurrently.
After separate jobs, assemble results with:

```bash
cd PASE-main-code
python -m experiments.pase_compact_main.summarize \
  --run-root experiments/chronological_runs/pase_compact_main
```

Default results under `PASE-main-code/experiments/chronological_runs/pase_compact_main`:

* `main_report.md`: three-dataset FULL average MAE, RMSE, MAPE and parameters.
* `main_results.csv`: unrounded H3/H6/H12 and 12-lead average values.
* `<dataset>/evaluation_test_s2025`: frozen selection, audited predictions,
  per-horizon metrics and parameters. Completed caches are verified and reused.

Metrics are the original native FP32 per-horizon values averaged over 12 leads;
RMSE is not pooled RMSE. No calibration, additional epochs or test-driven
selection occurs. Previously used public test splits are not fresh independent
holdouts. These replace only PASE rows in the historical paper table.

Numerical checks run in Linux GitHub Actions and on the server; no local Python
training/evaluation is needed. The training/evaluation/state implementation is
an isolated snapshot of the existing compact protocol, narrowed to FULL-only
scratch runs for Alameda/Orange. Model mathematics are imported unchanged.
