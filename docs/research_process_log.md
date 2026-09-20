# Research process and correction log

This is an append-only research record for the chronological IGSTGNN study. Each entry separates
the decision available at the time from later evidence. A correction does not rewrite or delete the
earlier record; it states what changed, why it changed, and which artifact can reproduce the change.
Server operation notes remain local and are not part of this record.

Each substantive step should record:

- date and research gate;
- question and prior assumption;
- evidence inspected or produced;
- decision and explicit non-claims;
- correction to earlier reasoning, when applicable;
- code, protocol, data, and result artifacts;
- next acceptance gate.

## 2026-09-19: From incident-only flexibility to matched routine controls

### Question and prior path

The first development path compared fixed, shared, conditioned, event-phase, and residual-phase
responses only on incident-centered windows. This was a valid controlled architecture screen, but it
could not establish whether a model distinguished incident disruption from ordinary traffic. Adding
more temporal flexibility inside the same incident-only population did not repair that identification
gap.

The fixed A development result remained stronger than shared B, conditioned C, and phase C'. C'
also collapsed toward the fixed curve. C'' restored trainability, but its result did not justify
continuing to add response flexibility before constructing a routine-traffic comparison population.
The incident-information negative controls test whether incident fields add information within the
incident sample population; they do not create ordinary non-incident observations.

### New source evidence

The paper describes continuous five-minute traffic readings aligned with incident logs, while the
released IGSTGNN samples are incident-centered. The local v8 source cache retains the continuous
January--October rows for 496 Contra Costa stations. The raw 2023 incident file contains 476,768
rows; a preliminary road/direction/postmile audit found 27,062 recorded incidents capable of
intersecting at least one Contra sensor corridor. Therefore, treating every time not selected in the
5,585 released samples as non-incident would create false negative labels.

A preliminary read-only capacity check used the frozen train/validation chronology, exact weekday,
exact five-minute slot, a 56-day search radius, and local sensor exposure. With a conservative
blackout based on one hour before report and at least two hours after report, it found at least one
candidate for 98.64% of train incidents and 93.35% of validation incidents. This was a feasibility
calculation, not a frozen artifact; the committed auditor below uses the manifest support guard
interval and must produce the authoritative counts.

### Correction and decision

The earlier working emphasis on learning a better accident decay curve was too narrow. The next gate
is now construction of recorded-incident-free matched controls at the affected-node level. A
graph-wide definition was rejected as the primary design because preliminary conservative coverage
left only 641 train and 8 validation five-minute centers. Remote incidents may exist in a matched
window, but no recorded incident may affect any node in the positive event's affected-node set during
the frozen blackout interval.

The term `recorded-incident-free matched control` is mandatory. These controls are not proven free of
unreported incidents and are not causal counterfactuals. Final incident duration is permitted only as
an offline exclusion label; it remains prohibited as a prediction-time model feature. Publisher
nominal time, timezone, and DST limitations remain unresolved.

### Frozen audit gate

Protocol: `experiments/chronological/matched_nonincident_v1.json`.

Auditor: `experiments/chronological/audit_matched_controls.py`.

The metadata-only audit:

- reads only train/validation manifests, event identity, sensors, raw incident logs, and its protocol;
- never reads traffic arrays or forecast targets;
- matches exact weekday and five-minute slot within 56 days and within the same split;
- excludes the same event time and nominal DST transition dates;
- defines affected sensors by the existing same-freeway, same-direction, 10-postmile rule;
- excludes candidates whose guarded support overlaps a recorded incident capable of affecting any
  positive affected sensor;
- reports candidate pairs, per-sample coverage, source quality, reuse pressure, and SHA256 inputs;
- refuses an existing output directory and does not assign or materialize final control windows.

Acceptance requires train coverage at least 98%, validation coverage at least 90%, and median
candidate counts of at least three and two respectively. Passing this gate authorizes traffic-X-only
ranking and final control assignment; it does not authorize model training or test access.

### Next gate

Run the frozen audit against the existing local evidence. Review zero-candidate samples, candidate
reuse pressure, duration anomalies, and per-road coverage. Only after that review may a second
protocol define traffic-X similarity, deterministic control assignment, and data materialization.

## 2026-09-19: Frozen v1 metadata audit execution

### Execution and operational correction

The first invocation used `data/xtraffic/Contra_Costa/sensors.csv` relative to the code repository.
The local source tree actually stores this file in the repository sibling `../data/xtraffic`. The
command failed at its first file read, before creating the requested output directory. The corrected
command passed the explicit sibling path. This was an invocation-path error, not a data or protocol
failure; both attempts and the correction are retained here so the server command is not copied from
the failed local path.

### First successful result and diagnostic correction

The first successful run produced
`../论文学习/匹配常规窗口审计_20260919/v1_local_audit` and passed the frozen gates. It found
3,549/3,604 train samples with at least one candidate (98.47%, median 5) and 852/917 validation
samples (92.91%, median 3). There were 20,957 train and 2,873 validation candidate pairs across
18,856 distinct split/candidate times. At least 75% of candidate times belonged to only one positive
sample; the maximum candidate degree was seven. These are membership counts, not final assignments.

Review then found that the source-quality duration counters included incidents sharing one of the 14
sensor road/direction keys even when their postmile was farther than 10 from every Contra sensor.
The candidate cleanliness calculation already applied the affected-sensor spatial test, so this did
not change candidate membership. However, the diagnostic scope was broader than its label suggested.
The auditor was corrected to index and report only incidents capable of affecting at least one
sensor, to name duration counters with their indexed scope, and to add per-road coverage. The first
output is preserved; the corrected rerun must use a new `_02` directory and confirm identical
candidate membership before it becomes the reference audit.

The `_02` rerun confirmed byte-identical `candidate_pairs.csv` and `sample_audit.csv`. Before commit,
the protocol was tightened once more to state explicitly that treating duration as minutes is an
uncertified source assumption, to require exactly 496 sensors, and to reject positive samples whose
guarded support or report identity disagrees with the frozen split/sidecar. These checks do not alter
candidate selection. A final `_03` run from the committed code shape is required as the reference.

### Reference result and review

The final reference is
`../论文学习/匹配常规窗口审计_20260919/v1_local_audit_03`. Its status is
`MATCHED_NONINCIDENT_CANDIDATE_AUDIT_PASS`, with protocol SHA256
`ec56655ffee8cb0f220058f7de3dd7ea654c855f9928ba29861a78cb8c1cb0c5`.

The corrected incident index contains 27,062 source rows capable of affecting at least one of the
496 sensors. Of these, 489 have missing/non-finite duration and 115 have negative duration; both use
the frozen 120-minute offline fallback. Another 24,370 valid durations below 120 minutes are raised
to that conservative minimum. The duration field is never exposed to the model.

Final candidate coverage is:

| Split | Positives | With candidate | Coverage | Candidate pairs | Median candidates |
|---|---:|---:|---:|---:|---:|
| train | 3,604 | 3,549 | 98.47% | 20,957 | 5 |
| validation | 917 | 852 | 92.91% | 2,873 | 3 |

The final candidate and sample CSV files are byte-identical to both earlier successful runs:

- `candidate_pairs.csv`: `aad59cbc9e183399ce1e91d82cd95a7d0f6e9ce448db0e05d3f413fff37c061a`
- `sample_audit.csv`: `1b0bebb1adc4b77a2d402045f94dede3af5f4105dc1cbe6004e3df1a7ee162bf`

There are 120 unmatched positives: 69 on SR4-E, 47 on SR4-W, and four on SR24-E. Validation SR4-E
is the weakest stratum at 86.40% coverage; all validation SR24/SR242 directions have 100% coverage.
The unmatched rows still have calendar candidates and cluster mainly in afternoon/evening incident
periods, so they are retained as an explicit unmatched stratum rather than rescued by weakening the
blackout. Final matched evaluation must report coverage and may not silently drop this stratum when
claiming population-wide performance.

Nine dedicated standard-library tests pass, including split support, exact calendar matching, DST
date exclusion, spatial exposure, blackout overlap, duration fallback, and protocol rejection. The
full repository suite could not run in the local WSL system interpreter because that interpreter has
neither NumPy nor PyTorch. No dependency was installed. Full regression remains a server-side gate
in the existing `igstgnn` Conda environment; this environment limitation is not recorded as a test
pass.

## 2026-09-19: Server regression corrected three test-boundary defects

### Evidence

The first server full-suite run used the existing `igstgnn` Conda environment and reported 72 tests,
with one failure and four errors. None arose from the matched-control auditor:

- the v8 source module imported optional `requests` at module load, preventing non-network builder
  tests from loading when the downloader dependency was absent;
- the CUDA determinism unit test enabled PyTorch deterministic algorithms globally and did not
  restore the prior process state, causing three later CUDA convolution cases to fail for missing
  `CUBLAS_WORKSPACE_CONFIG` even though those cases did not request deterministic execution;
- the negative-control integration test built a two-node fixture but asserted the 496-node production
  parameter count, producing `431789 != 443645`.

### Correction and non-impact

`source_v8.py` now imports `requests` optionally and raises an explicit error only when a network
download is actually requested. The two HTTP-specific tests skip when that optional dependency is
absent; cache, header, range validation, and all packaged-data paths remain testable. The determinism
test now restores algorithm, cuDNN, and TF32 global state in `finally`. The negative-control test now
asserts the correct 431,789 parameters for its two-node fixture and documents that 443,645 belongs to
the real 496-node package.

These changes do not alter model construction, production parameter counts, training determinism, or
matched-control candidates. They correct dependency scope, test isolation, and a fixture-specific
expected value. The server full suite must be rerun after pulling this correction; an optional-test
skip is acceptable only for the two HTTP cases when `requests` is absent.

## 2026-09-19: Traffic-X-only scoring and unique control assignment

### Question and frozen design

The metadata audit established that routine-window candidates exist, but it did not establish that
their pre-event traffic state resembles the corresponding incident window or select one final
control per positive. The second gate asks whether the v1 candidates can be assigned at useful
coverage without consulting the forecast target, test split, incident description, or incident
type.

Protocol: `experiments/chronological/matched_nonincident_x_v2.json`.

Scorer: `experiments/chronological/score_matched_controls.py`.

For each v1 candidate, the scorer reads only the 12 history slots from T-65 through T-10 on the
positive event's affected nodes. It excludes non-finite and negative source values, requires at
least 90% pairwise valid overlap, and ranks eligible edges by mean absolute raw-flow difference
divided by the frozen train global standard deviation. Missing-pattern mismatch, absolute day
distance, and candidate timestamp are deterministic tie breakers. A preference-ordered augmenting
path algorithm maximizes assignment cardinality while limiting each candidate timestamp to one
positive. It does not claim globally minimum total matching cost or a causal counterfactual.

Acceptance was frozen before execution: at least 95% train and 88% validation assignment coverage,
median valid overlap at least 98%, maximum absolute SMD at most 0.1 across affected-node history
mean, last-step mean, and late-three-minus-early-three trend, and candidate reuse no greater than
one. Passing remains an offline data-construction gate and does not by itself authorize the main
model experiment.

### Pre-execution corrections and tests

Review before the first real run found that the assignment CSV schema omitted fields present in its
rows, which would have caused `csv.DictWriter` to fail after all scoring work. The schema now equals
the complete edge-score schema plus preference rank. Balance summaries were also made JSON-safe for
an all-nonfinite feature: unavailable values are written as JSON `null`, and an unobserved SMD cannot
silently pass the balance gate. Protocol validation now requires all four prohibited input classes,
not only forecast Y.

Ten new tests cover those output and information boundaries, input-fingerprint drift, negative and
non-finite traffic filtering, the 90% overlap rule, a history window crossing a month boundary,
JSON-safe unavailable balance, deterministic assignment, candidate capacity, and an example where
an augmenting path is required to recover maximum cardinality. Together with the nine v1 tests,
19/19 targeted tests pass in the existing local Conda environment. No package was installed or
environment changed.

### Reference execution

The first two validation outputs, `v2_x_assignment_01` and `v2_x_assignment_02`, are byte-identical.
Protocol enforcement was then tightened to reject drift in source year/version, station axis,
history spacing, node scope, valid-value rule, and distance definition. Because this changed the
recorded code fingerprint, the final reference output is
`../论文学习/匹配常规窗口审计_20260919/v2_x_assignment_03`. Its three CSV artifacts remain
byte-identical to the first two runs; its summary values are identical and its summary file differs
only by the corrected code fingerprint. Each run verified 4,960 cached row reads totaling
521,109,504 bytes and all ten monthly source-manifest fingerprints.

| Split | Positives | Assigned | Coverage | Median X distance | Median overlap | Maximum preference rank |
|---|---:|---:|---:|---:|---:|---:|
| train | 3,604 | 3,519 | 97.64% | 0.1484 | 100% | 4 |
| validation | 917 | 820 | 89.42% | 0.1566 | 100% | 3 |

All 4,339 selected controls have zero missing-pattern mismatch, and 3,921 use their positive's
first-ranked candidate. Thirty train and 32 validation positives remain unmatched because candidate
capacity is exhausted; another 55 train and 65 validation positives had no v1 metadata candidate.
Candidate reuse is exactly one at maximum. Across the six split-feature balance checks, absolute SMD
ranges from 0.0148 to 0.0428 and passes the 0.1 gate. Fifty-four assignments have normalized X
distance above 0.5 and nine exceed 1.0; these tails must remain visible in later sensitivity analysis
rather than being hidden by aggregate balance.

Validation SR4-E remains the weakest stratum: 291/353 positives are assigned (82.44%), including 48
without a metadata candidate and 14 lost to capacity conflict. Overall validation passes the frozen
gate, but a population-wide claim may not rely on the 89.42% aggregate alone. Later evaluation must
report road/direction coverage and the unmatched population explicitly.

Reference output fingerprints are:

- `assignments.csv`: `de2c1b65e6601c3613239407f45cb0112ee6406ac2fbd41ab398200cf2c0a7bc`
- `edge_scores.csv`: `17b14bac7db198b51888227b00598cc3f528bf29b99b2d8adde1fcd2f7d76d25`
- `unmatched.csv`: `d9d548e9a36e748e9b0308c1158863d6bc50e1b393b3bcb13a042684855e45b1`
- `summary.json`: `59fa1532ce6845a8cb6e75085e02b2b1517b1b3bc6a961ade95031aedb7d5875`

### Decision and next gate

The v2 X-only assignment gate passes and the assignment is now frozen for downstream construction.
The next step is a separate materialization protocol that reads the selected candidate timestamps
and extracts their complete 26-slot raw windows only after assignment. That step may expose routine
future observations as supervised targets, but it may not rerank or replace controls using those
future values. It must verify source fingerprints, split confinement, byte-stable reconstruction,
window overlap diagnostics, and exact preservation of the frozen assignment before any
incident-versus-routine model objective is designed.

## 2026-09-20: Post-assignment full-window control materialization

### Question and information boundary

The v2 assignment fixed which routine timestamp belongs to each positive using history X only. Model
development still requires the selected controls' complete raw windows, including future
observations that can serve as supervised targets. Reading those future values before freezing the
assignment would leak outcome information into control selection. The v3 gate therefore materializes
Y only from the immutable v2 assignment and forbids Y from ranking, replacing, or removing controls.

Protocol: `experiments/chronological/matched_nonincident_materialize_v3.json`.

Materializer: `experiments/chronological/materialize_matched_controls.py`.

Each output row contains the same 26 five-minute slots and 496-station axis as the positive package:
X at indices 0:12, the excluded latency slots at 12:14, and Y at 14:26. Values remain raw float32;
there is no filling, clipping, or normalization. Each row also has a 496-element affected-node mask
derived from the positive incident's freeway, direction, postmile, and frozen 10-postmile radius.
The full graph is retained as model context, but the recorded-incident-free guarantee applies only
to the affected-node mask. Remote recorded incidents and unreported incidents may remain elsewhere
in the graph, so these windows are not graph-wide negatives or causal counterfactuals.

### Implementation corrections and tests

The first implementation review found that a non-finite score disagreement could place `Infinity`
in a rejection diagnostic and prevent strict JSON output. Non-finite disagreements are now counted
separately, leaving every pass and rejection summary valid under `allow_nan=False`.

Ten v3 tests cover immutable assignment-before-Y semantics, chronological boundary rejection,
26-slot X/gap/Y alignment, cross-month extraction plans, positive identity and affected-mask
construction, duplicate candidate rejection, within-split overlap reporting, exact X-score
reproduction, and raw negative/non-finite diagnostics. Together with the v1 and v2 tests, 29/29
targeted tests pass in the existing local Conda environment. No dependency or environment was
changed.

### Reference execution

The reference output is
`../论文学习/匹配常规窗口审计_20260919/v3_materialized_01`. An independent rerun in
`v3_materialized_02` produced byte-identical arrays, masks, manifests, and summary. Both runs read
and verified all 4,960 cached source rows (521,109,504 bytes) and all ten monthly manifest
fingerprints.

| Split | Control windows | Array shape | Affected values valid | Assignment-score max error |
|---|---:|---|---:|---:|
| train | 3,519 | 3519 x 26 x 496 | 100% | 0 |
| validation | 820 | 820 x 26 x 496 | 100% | 0 |

All full-graph X, latency, and Y values are finite and nonnegative. Recomputing the v2 distance,
overlap, missing-pattern, and six history-feature values from the materialized arrays reproduces all
4,339 frozen assignment rows exactly. Candidate timestamp reuse remains one, and train/validation
source-slot overlap is zero.

Reference output fingerprints are:

- `train_control_flow.npy`: `b6eafd55f5a605264f81a90f30d7f15d60c7386f2c439d0cf40f1dfa1b79dc09`
- `train_affected_mask.npy`: `d4eaf7f06d6de8ab4aa90b58cd25bf67b1e4055d8f897d5c24602214d1127464`
- `train_control_manifest.csv`: `cb48d8e00db29aa5e6baa3e3c26ea3ebdc2bbfcf1df70eb611b9565c84620d02`
- `val_control_flow.npy`: `7eab02eb3b1ea7efb4ec0899f3b3770ee6b38d1b1c4cfaf70d0d3415a180c15c`
- `val_affected_mask.npy`: `9825aa641974c5dda7e5ceb48a17212850f673831d98b77265727e318c1958f3`
- `val_control_manifest.csv`: `fa1d1f04ee20587c5526251f4d4b7a9f7e70aea3c54eafba326d7159f8832698`
- `summary.json`: `f7ef45ff7451fd7c80ad260380586cf09001f2260742dcf5bf3d17b018e8e498`

The server-transfer artifact contains only the final `_01` directory:
`Contra_Costa_v8_matched_controls_v3_20260920.tar.gz`, 94,594,834 bytes, SHA256
`7247e80b2332487dc2f767f870172edade0f8846ed8621df2e21780af213bee8`.

### New overlap evidence and correction to the analysis plan

Unique candidate centers do not imply independent 130-minute windows. In train, 3,258/3,519
controls (92.58%) share at least one five-minute source timestamp with another control; 21,899 of
38,075 unique source timestamps are reused and the maximum timestamp reuse is 15. In validation,
746/820 controls (90.98%) share a timestamp; 4,932 of 9,377 timestamps are reused and the maximum
reuse is 12.

This is not train-validation leakage because their source-slot intersection is empty, and it does
not invalidate the materialization. It does invalidate treating all matched windows as independent
replicates when estimating uncertainty. The earlier plan to proceed directly from unique centers to
ordinary sample-level confidence intervals is therefore corrected. Main results must use
time-blocked or overlap-cluster-aware uncertainty, and a non-overlapping control subset must be a
sensitivity analysis. Training may use all matched controls, but overlapping windows require an
explicit weighting or sampling decision rather than being silently counted as independent evidence.

### Decision and next gate

The v3 materialization gate passes. The next gate is no longer data discovery; it is a frozen model
and evaluation protocol for incident-versus-routine separation. Before implementing a new module,
that protocol must specify the prediction target on affected nodes, how the incident-present and
routine branches share parameters, how overlapping controls are sampled or weighted, paired and
unmatched evaluation populations, and road/direction reporting for the weak SR4-E validation
stratum. Test access remains prohibited.

## 2026-09-20: Matched outcome audit before paired model development

### Why the model refactor was paused

The v3 controls make paired training technically possible, but they do not establish that the
future incident/control difference is stable enough to justify a new incident-residual model. The
v4 gate therefore examines outcomes only after assignment was frozen. It reads no test data or model
prediction and cannot change, replace, or remove a match using Y.

Protocol: `experiments/chronological/matched_outcome_audit_v4.json`.

Auditor: `experiments/chronological/audit_matched_outcomes.py`.

For each pair and time step, flow is first averaged over that event's affected nodes. The primary
matched change contrast is then

```text
(incident_t - control_t) - mean_t=-20,-15,-10(incident_t - control_t)
```

and events, rather than sensor values, receive equal weight. H1-H6 and H7-H12 were frozen as early
and late horizons. Main uncertainty uses 2,000 positive-incident ISO-week cluster-bootstrap draws.
A deterministic outcome-blind sensitivity subset greedily excludes a pair whenever any of its 52
positive/control source slots was already used on either side of an accepted pair. This is stricter
than checking positive and control overlap separately.

The primary phenomenon gate was frozen before the first Y audit: both train and validation must
have an absolute H7-H12 contrast of at least 0.05 train standard deviations, week-block intervals
must exclude zero, directions must agree, and the strict non-overlap subsets must agree in direction
with at least 0.025 standard deviations. These thresholds are not relaxed after observing failure.
The estimand is explicitly a matched observational contrast, not a causal treatment effect.

### Implementation corrections

The first local command invoked the file directly under Windows Conda and failed at import time
because the repository package root was not on that interpreter's module path. No data was read and
no output directory was created. The supported invocation is now
`python -m experiments.chronological.audit_matched_outcomes`, which also avoids local/server path
differences.

Static review before reading outcomes corrected NumPy boolean indexing from a mixed advanced-index
form to a two-stage `[sample][:, mask]` form so that time remains the first axis. It also passes the
train standard deviation explicitly instead of reconstructing it from a possibly zero observed
contrast.

The first completed output, `v4_outcome_audit_01`, contained the frozen primary gate but omitted
early-horizon intervals and the promised road/direction table. That reporting omission did not
change any match or primary criterion. The final protocol adds those descriptive outputs and
explicitly prohibits them from changing the primary gate. A later test-only correction isolated the
cross-side overlap case by removing a redundant same-side overlap; it did not change production
logic or the four CSV files. `v4_outcome_audit_04` is the final reference, and an independent
`v4_outcome_audit_05` rerun is byte-identical for all five artifacts.

Eleven v4 tests cover the information boundary, frozen horizons, non-overlap cross-side exclusion,
cluster-bootstrap determinism and missing strata, and positive and negative gate cases. Together
with v1-v3, 40/40 matched-control tests pass in the existing local Conda base environment. No
environment or package was created or modified.

The wider local discovery run executed 67 test entries: available non-PyTorch tests passed and 11
Linux launch-script tests skipped as designed, but seven existing modules could not import because
the Windows Conda base environment has no PyTorch and no Linux-only `resource` module. This is an
environment limitation, not a full-suite pass. The complete repository regression remains a
server-side gate in the existing Linux `igstgnn` Conda environment.

### Reference result and failed primary gate

| Split/population | Pairs | Pre pair MAE | H1-H6 pair MAE | H7-H12 pair MAE | H1-H6 change | H7-H12 change |
|---|---:|---:|---:|---:|---:|---:|
| Train/all matched | 3,519 | 25.735 | 30.934 | 29.151 | -1.810 | 0.163 |
| Train/non-overlap | 638 | 23.479 | 28.226 | 26.903 | -2.002 | 0.827 |
| Validation/all matched | 820 | 26.210 | 29.730 | 28.530 | -0.699 | 1.184 |
| Validation/non-overlap | 154 | 23.469 | 27.110 | 26.241 | -4.478 | -2.423 |

The all-matched H7-H12 changes are only 0.0010 and 0.0075 train standard deviations. Their 95%
week-block intervals are `[-0.807, 1.111]` for train and `[-0.517, 2.831]` for validation. Both
cross zero and both fall well below the predeclared 0.05 threshold. The non-overlap late direction
also changes from positive in train to negative in validation. The primary phenomenon gate therefore
fails and `paired_benchmark_ready` is false.

There is a limited onset signal: train H1-H6 is -1.810 with interval `[-2.741, -0.968]`, while
validation is -0.699 with interval `[-2.893, 1.053]`. Per-step trajectories show the clearest drop
around the excluded latency slots and first four forecast horizons, followed by decay. However, the
validation interval crosses zero, the magnitude is at most 0.0114 train standard deviations in the
all-matched population, and road/direction signs are not stable across splits. This can motivate a
future hypothesis but cannot replace the failed primary endpoint.

Matched coverage remains 3,519/3,604 (97.64%) in train and 820/917 (89.42%) in validation. The
unmatched population and weak validation SR4-E coverage remain outside this outcome estimand.

Reference fingerprints are:

- `pair_metrics.csv`: `3a451c9c1c7c8acf8e2513c95e957569fd00323904c0d5c37e6eb4d1ac463e19`
- `trajectory.csv`: `8f62533d126655dde8355a9954de1cefe541995fe467a31dedeb295d3d7bd6d8`
- `distance_bands.csv`: `b55450f2d5b05755663a12f0f779c0a696598d856aa0cfef911279c609d2ef0a`
- `road_direction.csv`: `8ddcf13be1f4d63c92f0dcf9057c9bcf5090167fad4c832a73be904fa065d3b2`
- `summary.json`: `fd53afeb54ba42f4c947ad7754f2b8c50529c77b31a6fce8806114c3d65f90be`

### Decision and next falsification gate

The failed average signed late effect does not prove that incident effects contain no predictable
heterogeneity; positive and negative event-specific changes can average to zero. It does show that
the current data do not support a stable population-wide late-flow shift, so the planned paired
model refactor is not authorized by v4.

The next evidence task is a control-control placebo constructed without Y: assign and materialize a
second routine window for each eligible positive, then compare incident-control divergence with
routine-routine divergence under the same horizons, masks, weighting, and overlap-aware uncertainty.
Only excess divergence beyond that natural routine variability would justify testing whether event
attributes predict heterogeneous residuals. Lowering the v4 threshold or redefining H7-H12 after
seeing these results is prohibited.

## 2026-09-20: Frozen second routine-control assignment (v5a)

### Why a second control is needed

The v4 signed late contrast failed, but that failure cannot distinguish weak incident signal from
ordinary day-to-day traffic variation between two history-matched windows. The next falsification
gate therefore needs two routine windows for the same positive event. Their difference supplies a
placebo level against which incident-versus-routine divergence can be compared before any model
refactor is authorized.

The original v2 primary assignment remains immutable. Protocol
`experiments/chronological/second_matched_control_v5a.json` selects the second control only from the
already frozen v2 `edge_scores.csv`. It reads no traffic array, future Y, test split, incident text,
incident type, or v4 outcome. Every center already used by any primary control is excluded globally,
and the remaining second-control centers have capacity one. The same maximum-cardinality preference
augmenting-path algorithm is used; it maximizes assignment count but makes no global cost-optimality
claim.

Before assignment, the acceptance thresholds were frozen at 85% train coverage, 73% validation
coverage, 60% minimum road/direction coverage, 0.10 maximum absolute feature SMD, and one maximum
candidate-center use across both control sets. These thresholds may not be weakened after inspecting
the assignment.

### Reference assignment and limitations

The reference output is
`../论文学习/匹配常规窗口审计_20260919/v5a_second_assignment_01`; an independent rerun in
`v5a_second_assignment_02` is byte-identical for all three files.

| Split | Frozen primary pairs | Second controls | Coverage | No unused candidate | Capacity conflict |
|---|---:|---:|---:|---:|---:|
| Train | 3,519 | 3,106 | 88.26% | 361 | 52 |
| Validation | 820 | 618 | 75.37% | 190 | 12 |

The maximum absolute feature SMD is 0.0275, all primary and secondary centers together have maximum
reuse one, and the median original v2 preference rank of a second control is two. First and second
controls are never the same center and are separated by at least seven days.

All frozen gates pass, but coverage is heterogeneous. Validation SR4-E retains only 187/291 primary
pairs (64.26%) and SR4-W retains 221/309 (71.52%). The future placebo estimand therefore applies to
the common three-window subset, not all v4 pairs, and must report road/direction strata rather than
silently generalizing to the unmatched population.

Eight v5a tests cover the information boundary, primary-center exclusion, strict CSV booleans,
maximum-cardinality reassignment, ineligible-edge exclusion, capacity-conflict reporting, and
preference-rank semantics. Together with v1-v4, 48/48 targeted tests pass locally. The full Linux
repository regression remains a server-side gate in the existing `igstgnn` Conda environment.

Reference fingerprints are:

- `secondary_assignments.csv`: `82df0a4d9e43c6fac3220e7d828240db2d9746319a841cce95e6e83c870e7f97`
- `unmatched_primary_pairs.csv`: `12f8fd33e6ab83aae53504b675f77b96aaf5434953c378659253111752a615b4`
- `summary.json`: `2bb44e2c1f2a7952eeac0534f3be14f07b1eba33497b639f8a7f51fc884005bb`

### Next gate

v5b must materialize exactly 3,106 train and 618 validation second-control windows only after this
assignment is frozen. It must reproduce every X score, preserve raw missing values, verify positive
identity and affected masks, enforce chronological split isolation and source fingerprints, and
report dependence across the complete incident/C1/C2 triples. Materialized Y may audit the frozen
assignment but may never rank, replace, or remove a control.

## 2026-09-20: Second routine-control materialization (v5b)

Protocol `experiments/chronological/second_matched_control_materialize_v5b.json` and materializer
`experiments/chronological/materialize_second_matched_controls.py` freeze the v5a output before
reading future values. They also fingerprint the v3 primary-control manifests, so every output row
must complete one unchanged incident/C1/C2 triple. The second-control files have an explicit
`second_control` prefix and cannot overwrite v3 outputs.

The first reference attempt was rejected before creating its output directory because the new
three-way identity check compared integer freeway and node-count values constructed in memory with
their CSV string representations. This was a representation error, not a data mismatch. The check
now performs typed integer comparison, and a regression test covers mixed in-memory/CSV types; no
identity field, assignment, or acceptance threshold was weakened.

The final reference is
`../论文学习/匹配常规窗口审计_20260919/v5b_second_materialized_01`. An independent `_02`
rerun is byte-identical for all seven artifacts. Both runs verified all 4,960 cached source rows and
all ten monthly fingerprints.

| Split | Shape | Affected values valid | X-score max error | Strict three-window subset |
|---|---|---:|---:|---:|
| Train | 3106 x 26 x 496 | 100% | 0 | 389 (12.52%) |
| Validation | 618 x 26 x 496 | 100% | 0 | 90 (14.56%) |

No triple has internal source-slot overlap, no primary and secondary control use the same center,
and no train source slot appears in validation. Across different triples, however, dependence is
substantial: 3,105/3,106 train triples and 618/618 validation triples share at least one source slot
with another triple. Maximum combined source-slot reuse is 19 in train and 13 in validation. The
v5c main analysis must therefore retain positive-incident ISO-week block bootstrap uncertainty; the
389/90 strictly non-overlapping triples are a smaller direction-and-magnitude sensitivity analysis.

Eight v5b tests cover the frozen information boundary, rank normalization, triple identity,
numeric CSV/in-memory identity equivalence, duplicate-center rejection, combined overlap, and
strict-subset semantics. All 56 v1-v5b targeted tests pass in the actual local Conda base Python.
An earlier shell probe had resolved `conda` to an empty Windows system placeholder and returned no
test output despite status zero; this was not counted as evidence. The recorded 56-test pass used
`D:\\anaconda\\set\\python.exe`. Server regression must still use the existing `igstgnn` Conda
environment.

Reference fingerprints are:

- `train_second_control_flow.npy`: `bca9e7a728a9449392b0bd1e000ff1863f0c73e8482feec740a497fface38bdf`
- `train_second_affected_mask.npy`: `4e37b57fc736cdebcea64cf3b4e039ccea37b895813efacebdba78ad6bd81f37`
- `train_second_control_manifest.csv`: `4236dc659e4ea9188c5263f689b45d1bb9fed9f60c1cd35e112550a5a30ef286`
- `val_second_control_flow.npy`: `c9a80fe907938b319578b67321976db228f839123ce8b5aaf50be557ed204585`
- `val_second_affected_mask.npy`: `c6a39971a15ed38c6e4d8ace07181128cbcbe36aaef37a80c71a473b2eb684fc`
- `val_second_control_manifest.csv`: `b1da9fc365dfd90462a0caebddeff00d879cc11f5b8006e97f3bd093b9521ac3`
- `summary.json`: `5705d9118d1b91d2e111ecebdffe35169990c3b19a15441eb43931603a15fb6f`

### Frozen v5c decision before outcome inspection

For each event and step, v5c will compare symmetric incident divergence
`0.5 * (mean_nodes(|I-C1|) + mean_nodes(|I-C2|))` with routine placebo divergence
`mean_nodes(|C1-C2|)`. Each series first subtracts its own H-3:H-1 baseline mean. The primary
quantity is their event-level difference, with events weighted equally.

The late H7-H12 endpoint remains primary to avoid switching to the v4 onset pattern after seeing
it. Both train and validation must have late excess divergence of at least 0.05 train standard
deviations and positive-week block-bootstrap lower bounds above zero. Their strict three-window
non-overlap subsets must contain at least 350 and 80 events respectively and show positive late
excess of at least 0.025 train standard deviations. Early H1-H6 and road/direction estimates are
required descriptive outputs but cannot change this gate. No threshold may change after v5c reads Y.

## 2026-09-20: Incident-versus-routine placebo audit (v5c)

### Implementation and pre-outcome corrections

Protocol `experiments/chronological/matched_placebo_audit_v5c.json` and auditor
`experiments/chronological/audit_matched_placebo.py` implement the frozen common-triple estimand.
They verify every positive, primary-control, and secondary-control array, mask, manifest, summary,
and protocol fingerprint before extraction. Test access and model training remain prohibited.

Before the first outcome audit, one synthetic unit-test expectation was corrected from 7 to 6: in
that fixture, symmetric incident divergence rises from 2 to 8, so its baseline-adjusted value is 6.
The implementation formula did not change. The subsequent input-only preflight also caught a
manually transcribed primary validation-flow SHA256 missing one zero. It was corrected to the
already recorded v3 fingerprint before any v5c outcome was read. These corrections changed neither
the estimand nor a gate.

Eleven v5c tests cover the information boundary, immutable late primary horizon, outcome-blind
strict subset, symmetric divergence formula, separate baseline adjustments, invalid-value
rejection, shared three-side exclusion set, and positive/negative gate cases. All 67 v1-v5c
targeted tests pass in the actual local Conda base Python.

### Reference result and failed gate

The final reference is
`../论文学习/匹配常规窗口审计_20260919/v5c_placebo_audit_01`; independent `_02` output is
byte-identical for all four artifacts.

| Split/population | Triples | Early incident change | Early routine change | Early excess | Late incident change | Late routine change | Late excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train/all common | 3,106 | 3.629 | 1.239 | 2.390 | 2.080 | 1.754 | 0.327 |
| Train/strict | 389 | 4.471 | 1.995 | 2.476 | 3.124 | 2.244 | 0.880 |
| Validation/all common | 618 | 2.632 | 1.149 | 1.484 | 1.536 | 1.697 | -0.161 |
| Validation/strict | 90 | 3.231 | 0.646 | 2.585 | 3.273 | 2.218 | 1.055 |

The all-common late excess is only 0.0021 train standard deviations in train and -0.0010 in
validation. Its week-block intervals are `[-0.209, 0.864]` and `[-1.545, 1.249]`; both cross zero.
The strict subsets are positive but only 0.0056 and 0.0067 train standard deviations, far below the
frozen 0.025 sensitivity threshold, and their intervals also cross zero. Thus the magnitude and
main interval checks fail in both splits, and `heterogeneity_screening_ready=false`.

The descriptive early excess is more coherent: 2.390 in train (95% interval `[1.846, 3.000]`) and
1.484 in validation (`[0.061, 3.112]`), equal to 0.0151 and 0.0094 train standard deviations. All
12 road/direction early point estimates are positive. This is evidence of a weak immediate-onset
divergence, not evidence for the frozen late endpoint: all 12 road/direction late intervals cross
zero and late signs vary. The early result must not be used to relabel the failed v5c gate as a pass.

Reference fingerprints are:

- `triple_metrics.csv`: `192b44e82223a9f943f3b47a5141ea1d19819b7cd0d0c75b0c234d372dcb5bc5`
- `trajectory.csv`: `8478a4b471233d378f63075c1bd8a646c4397d675787c91594723000eec2d4bc`
- `road_direction.csv`: `665511d8c23d0c1b9cac9ddf5e327988730228af65dc0a7fcb4fe883a642314d`
- `summary.json`: `c443581334f1f0aebece82c910a38c78b1dc29a22565d081f6a583298875571a`

### Decision after v5c

The current matched evidence does not authorize the proposed sustained incident-residual or
event-attribute heterogeneity model. Rebuilding that module now would optimize against a phenomenon
that is not distinguishable from routine variability at H7-H12. The defensible next decision is
between an explicitly new, prospectively evaluated short-onset objective and a data/label-quality
study; it is not to weaken the v5c gate, inspect test, or silently return to the rejected late model.
