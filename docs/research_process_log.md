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

## 2026-09-20: Short-onset module benefit review before implementation

### Question and correction to the working interpretation

The v5c descriptive result raises a narrower question: whether the positive H1-H6 excess divergence
justifies a dedicated short-onset incident module. It does not by itself. The 2.390 train and 1.484
validation values measure an observed incident-versus-routine flow divergence, not an error made by
fixed A and not a residual predictable from report-time-safe incident fields. Treating those values
as attainable MAE reductions would conflate phenomenon detection, predictability, and metric gain.

### Project-specific benefit scale

Fixed A has validation all-node MAE 22.6867. Its associated-node H1-H6 MAE is 22.2654. Associated
nodes account for 36,343/454,832 = 7.9904% of valid cells at each horizon, so an onset-only module
restricted to associated nodes and H1-H6 can directly alter only 3.9952% of the all-node evaluation
cells. Under the optimistic assumption that every other prediction is unchanged:

| Relative reduction inside associated-node H1-H6 | All-node MAE reduction | Relative all-node reduction |
|---:|---:|---:|
| 5% | 0.0445 | 0.196% |
| 10% | 0.0890 | 0.392% |
| 20% | 0.1779 | 0.784% |
| 100% (unattainable zero-error support ceiling) | 0.8896 | 3.921% |

This is metric-support arithmetic, not an oracle estimate of learnable incident signal. It shows why
a useful accident-subset improvement can produce only a small all-node change. It is already the
favorable setting in which every released example is incident-centered; an operational benchmark
containing mostly routine windows would dilute the aggregate gain further.

The existing single-seed negative controls reinforce the need for a predictive-information audit.
Fixed A is slightly better on all nodes than `traffic_only` (22.6867 versus 22.7363, 0.218% relative)
and `shuffled_incident` (22.6867 versus 22.7301, 0.191% relative). However, on the exact target subset,
associated-node H1-H6, A is worse than `traffic_only` (22.2654 versus 22.2030) and only slightly better
than shuffled incidents (22.2654 versus 22.2920). These are selection-time point estimates from three
separately trained models, without multiple-seed or paired-cluster uncertainty, and therefore do not
establish either benefit or harm. They do show that a larger response head is not yet justified.

### External evidence checked on 2026-09-20

- Xie et al., *Deep Graph Convolutional Networks for Incident-Driven Traffic Speed Prediction*,
  CIKM 2020, DOI `10.1145/3340531.3411873`, reports that its incident component changes MAPE from
  12.22% to 11.02% in SFO and 18.63% to 17.21% in NYC. The model first separates critical from
  non-critical incidents and evaluates 5, 10, and 15 minute prediction steps on about four weeks of
  data. This supports short-horizon, high-impact selection, not an unrestricted all-incident module.
- Xu et al., *Urban short-term traffic speed prediction with complicated information fusion on
  accidents*, Expert Systems with Applications 2023, DOI `10.1016/j.eswa.2023.119887`, reports only
  about 0.2% overall accuracy improvement and attributes the small gain to few accidents and weak
  accident impacts. This is the closest published magnitude warning for the present question.
- Yu et al., *Deep Learning: A Generic Approach for Extreme Condition Traffic Forecasting*, SDM
  2017, DOI `10.1137/1.9781611974973.87`, reports much larger gains for a mixture model specialized to
  extreme-condition prediction. Those conditional-task gains cannot be interpreted as gains on an
  all-window, all-node metric.
- Fukuda et al., *Short-term prediction of traffic flow under incident conditions using graph
  convolutional recurrent neural network and traffic simulation*, IET ITS 2020, DOI
  `10.1049/iet-its.2019.0778`, uses simulation to address the shortage of real incident examples.
  This identifies sample scarcity, rather than response-head capacity alone, as a central bottleneck.
- Yu et al., *FUSE-Traffic: Fusion of Unstructured and Structured Data for Event-aware Traffic
  Forecasting*, SIGSPATIAL 2025, DOI `10.1145/3748636.3762776`, arXiv `2510.16053`, reports roughly
  2%-6% average MAE improvement over its D2STGNN traffic backbone across three datasets. It combines
  accidents, weather, crime, and public-event semantics retrieved by an LLM, so it is an upper-scope
  comparator rather than evidence that the current accident fields warrant an LLM module.
- Sun et al., *Dual-level Graph Transformer for Spatiotemporal Incident Impact Prediction*, arXiv
  `2303.12238`, reports that simple incident classification, position encoding, and incident-metadata
  embedding attempts did not work. Its successful task predicts incident duration and spatial extent,
  emphasizing affected-sensor identification rather than generic metadata fusion.

This was a focused AI-assisted literature and full-text check, not a systematic review or
meta-analysis. Citation counts were used only for discovery; claims above were checked against the
paper text where lawful full text was available. The Fukuda claim is limited to its indexed abstract.

### Decision and prospective gate

The short-onset direction is worth one inexpensive benefit audit, but not immediate development of a
large module. The next artifact must use the saved A predictions and the common incident/C1/C2
triples, remain train/validation-only, and report all-node, associated-node H1-H6, associated-node
H7-H12, high-impact-event, and routine-control errors. A post-hoc constrained oracle may modify only
associated nodes at H1-H6; it is an upper-bound diagnostic and must never be reported as model
performance. A report-time-safe shallow probe must then test whether incident fields predict the
direction and magnitude of A's onset residual, with grouped validation by event/day and no target
information in its inputs.

Before outcome inspection, the implementation protocol should freeze these development rules:

- constrained-oracle all-node improvement below 0.3%: do not build an onset module;
- oracle improvement from 0.3% to 1.0%: permit only a small gated residual expert;
- oracle improvement above 1.0% with a stable validation interval: permit a formal module study;
- regardless of the oracle, the shallow report-time-safe probe must improve associated-node H1-H6,
  must not materially harm routine C1/C2 windows or H7-H12, and must survive multiple seeds and
  event/day-clustered uncertainty before architecture expansion.

If authorized, the defensible architecture is a high-impact event gate, an affected-node gate, a
fixed H1-H6 onset support that decays to zero for H7-H12, and a small residual head regularized back
to the traffic baseline for low-impact events. The research contribution would be selective onset
adaptation with matched routine-placebo evidence, not simply adding another attention block. Test
access remains prohibited throughout this development decision.

## 2026-09-20: Literature position for a baseline-anchored selective incident expert

### Proposed claim and necessary terminology correction

The proposed intuition is valid: preserve fixed A as the general predictor and invoke an incident
specialist only where it is expected to help. A conventional mixture of experts does not, however,
guarantee that global performance will not deteriorate. Jointly trained experts and a soft router can
change every prediction, route normal samples incorrectly, and reduce an average loss while harming
important strata. The intended design is therefore more precisely a **baseline-anchored selective
incident expert**, related to residual MoE and learning-to-defer, rather than a symmetric MoE.

The frozen fixed-A checkpoint should be the baseline expert. A new expert predicts only a residual:

`prediction = prediction_A + event_gate * node_gate * onset_mask * residual`.

`onset_mask` is exactly zero at H7-H12 and the spatial gate is exactly zero outside the report-time
candidate exposure set. The event and node gates must fall back to zero when confidence is
insufficient. This construction can guarantee exact equality with A where the hard masks or fallback
gate are zero, and exact equality at initialization. It cannot mathematically guarantee lower error
on unseen samples wherever the expert is active. That broader statement requires prospective
non-inferiority evidence, not architecture wording.

### Focused literature search and findings

Searches on 2026-09-20 used OpenAlex, Crossref, arXiv, and a bounded Semantic Scholar request for
combinations of traffic forecasting/prediction, incident/extreme/event, mixture of experts, gating,
fallback, learning to defer, and negative transfer. Citation counts below are OpenAlex counts on that
date. Accessible arXiv full text was checked for CP-MoE and TFMoE. ACM PDF candidates returned
Cloudflare HTML rather than PDF, and the new Elsevier DE-GAM full text was not accessible; no access
control was bypassed.

- Coric, Wang, and Vucetic, *Traffic speed forecasting by mixture of experts*, IEEE ITSC 2011,
  DOI `10.1109/ITSC.2011.6083118` (1 citation), already separates free-flow and congested regimes
  with two linear experts and a decision-tree gate. Regime-specialized traffic experts are therefore
  longstanding rather than novel.
- Yu et al., *Deep Learning: A Generic Approach for Extreme Condition Traffic Forecasting*, SDM
  2017, DOI `10.1137/1.9781611974973.87` (423 citations), uses a Mixture Deep LSTM to jointly model
  normal traffic and post-accident patterns. A normal/accident expert split is direct prior art.
- Li et al., *ST-MoE: Spatio-Temporal Mixture-of-Experts for Debiasing in Traffic Prediction*, CIKM
  2023, DOI `10.1145/3583780.3615068` (18 citations), is a plug-in that routes road-segment patterns
  to specialized subnetworks to reduce uneven spatial error and improve overall accuracy. It does
  not preserve a frozen baseline or establish incident-versus-routine effects.
- Li, Magli, and Francini, *To be Conservative or to be Aggressive? A Risk-Adaptive Mixture of
  Experts for Mobile Traffic Forecasting*, ICC 2023, DOI `10.1109/ICC45041.2023.10279534`, routes
  between conservative and aggressive experts when a peak trend is detected. Although its domain is
  cellular traffic, it is prior art for a risk-adaptive rare-peak router.
- Jiang et al., *Interpretable Cascading Mixture-of-Experts for Urban Traffic Congestion Prediction*,
  KDD 2024, DOI `10.1145/3637528.3671507` (21 citations), combines sparse graph experts with trend
  and periodic experts using learned confidence weights. It empirically improves congested and
  non-congested cases, but all components are jointly optimized and its reported robustness is not a
  frozen-baseline non-degradation guarantee.
- Lee and Park, *Continual Traffic Forecasting via Mixture of Experts*, arXiv `2406.03140` (1
  citation), protects earlier traffic knowledge under an evolving sensor network through clustered
  experts, consolidation, and replay. It addresses catastrophic forgetting rather than incident
  onset, but demonstrates that preserving a base capability requires explicit training constraints,
  not merely adding experts.
- Iqra et al., *DE-GAM: A dual-encoder graph-attention mixture-of-experts framework for post-crash
  traffic speed forecasting during freeway all-lane-closure incidents*, Transportation Research Part
  C 2026, DOI `10.1016/j.trc.2026.105784` (0 citations), is a direct title-level collision with an
  accident-specific graph MoE. Its inaccessible full text prevents claims about its router or
  controls, but the existence and scope are verified from Crossref, OpenAlex, and Semantic Scholar.
- Cui et al., *TransMoE: Multimodal traffic prediction with large language model and mixture of
  experts*, Transportation Research Part C 2026, DOI `10.1016/j.trc.2026.106018` (0 citations),
  combines heterogeneous urban context with MoE. It further removes novelty from generic
  multimodal-MoE fusion, although its grid-wise task and data differ from the current study.
- Madras, Pitassi, and Zemel, *Predict Responsibly: Improving Fairness and Accuracy by Learning to
  Defer*, arXiv `1711.06664`, and Mozannar and Sontag, *Consistent Estimators for Learning to Defer
  to an Expert*, arXiv `2006.01862`, establish the broader idea of learning when one predictor should
  pass to another. Fallback routing is not a new general ML concept.

### Novelty assessment

“Use MoE for accidents” is not sufficiently novel for AAAI, KDD, or Transportation Research Part C:
the normal/accident split dates to 2017 and DE-GAM now directly combines crash forecasting, graph
attention, and MoE. “Freeze a baseline and add a gate” is also insufficient by itself because
learning-to-defer and safe/negative-transfer research already cover that general principle.

A potentially defensible contribution remains in the combination of four properties not found
together in the checked traffic papers:

1. **Placebo-identified routing target:** matched incident/C1/C2 windows define whether an event has
   excess short-onset impact beyond routine traffic, rather than routing on congestion appearance.
2. **Three-axis selective support:** event impact, genuinely responsive node, and H1-H6 onset gates
   jointly determine where the specialist may intervene.
3. **Baseline-anchored non-inferiority:** fixed A is preserved exactly outside the intervention support,
   and training/calibration explicitly constrain all-node, routine-control, non-candidate-node, and
   H7-H12 regret relative to A.
4. **Falsifiable evaluation:** gains must appear on high-impact incidents and responsive nodes while
   paired day-cluster intervals demonstrate no material harm on the protected populations.

For a Transportation Research venue, this may be a meaningful incident-forecasting contribution if
DE-GAM is compared directly and the matched-control/affected-node evidence is strong. For AAAI or
KDD, it likely needs a general rare-event selective-forecasting formulation, multiple cities or event
types, and evidence that the routing principle transfers beyond one incident dataset.

### Meaning of “protect global performance”

Three levels must not be conflated:

- **Structural preservation:** frozen A and exact zero gates make predictions identical to A outside
  the allowed support. This is a true implementation guarantee.
- **Development non-inferiority:** on untouched validation, the paired upper confidence bound of
  `MAE_new - MAE_A` must be at or below a prospectively frozen margin for all nodes and each protected
  stratum. This is statistical evidence, not a universal guarantee.
- **Unseen-test generalization:** no router can guarantee lower error for every future accident from
  finite observational data. The honest claim is calibrated selective improvement with measured
  fallback risk.

The next pre-model gate should therefore estimate oracle expert advantage and train a shallow
report-time-safe advantage router. It should predict whether the specialist will beat A, not merely
whether an incident exists. Only if that router separates positive from negative regret on validation
should a neural residual expert be implemented. C1/C2 outcomes and post-event labels remain offline
training/evaluation evidence and are prohibited at inference; test access remains prohibited.

## 2026-09-20: Frozen v6a/v6b branch-benefit protocol before outcome inspection

### Correction: what the available counterfactual can and cannot establish

The planned same-checkpoint comparison required a terminology correction before implementation.
Fixed A is not a traffic-only backbone waiting for an incident expert: its published inference path
already enables ICSF/TIID when `incident_data` is supplied. Consequently, comparing the same fixed-A
checkpoint with and without `incident_data` does **not** estimate the benefit of a new expert over A.
It estimates the value of the existing incident branch and whether selectively suppressing that
branch can reduce A's errors. The two modes are now named `incident_on` and `incident_off`; positive
`absolute_error_off - absolute_error_on` means activation helps. Published fixed A remains the
`incident_on` anchor on positive incident windows.

This distinction also changes the interpretation of matched controls. C1/C2 are routine outcome
placebos to which the positive report's age and spatial exposure are applied synthetically. Their
traffic X/Y and forecast clock come from the control window. They test whether branch activation
would help or harm a routine-looking outcome under the paired report context. They are not actual
no-report production inputs and are not causal counterfactual outcomes.

### v6a: same-weight branch-error materialization

`incident_branch_materialize_v6a.json` freezes the following before any new model-error outcome is
read:

- fixed A, seed 2025, best epoch 99, plain state-dict SHA-256
  `b0c712ad9c00007417ccc6ea6268f373852d04063efba2d15d3f49f5497b8e13`;
- the v5c common-triple cohort of 3,106 train and 618 validation incidents;
- the complete positive incident cohort of 3,604 train and 917 validation samples, retained as a
  separate population for the global validation gate;
- three modes per split: positive incident, primary control C1, and secondary control C2;
- for each control, its own X/Y and candidate-t0 clock, but the paired positive incident's report age,
  distances, and candidate-node mask;
- one strictly loaded checkpoint for both `incident_on` and `incident_off`, no gradients, no
  optimizer, train/validation only, and no test access.

The materializer stores float32 per-cell absolute errors for both modes, the target-valid mask, the
candidate mask, and the positive sample identity for the three matched cohorts and the separate full
positive cohort. It also reports branch prediction differences on all nodes, candidate H1-H6,
candidate H7-H12, and noncandidate nodes. The latter is important because the current ICSF modifies
history before dynamic-graph construction, so the existing incident path must not be assumed to be
spatially local merely because TIID masks its context projection.

The materialized errors are sufficient for every declared v6b estimand while avoiding a larger and
unnecessary duplicate of predictions and targets. They do not authorize training and are not a model
comparison result.

### v6b: oracle ceilings and report-time-safe routing

`expert_benefit_audit_v6b.json` freezes three post-hoc oracle levels over candidate nodes and H1-H6:

1. event oracle: one on/off decision for all candidate onset cells in an event;
2. event-node oracle: one decision per event and candidate node, shared across H1-H6;
3. cell oracle: one decision per event/node/horizon cell.

All hybrid outputs remain exactly fixed-A `incident_on` outside candidate-node H1-H6. The cell oracle
is the loosest unattainable ceiling. Event-node is the primary development-tier oracle because it is
the closest of the three to the proposed event/node/horizon support. None may be reported as model
performance.

Two fixed weighted-ridge routers are fit on train and evaluated on validation: event and event-node.
The primary router is event-node. Train rows combine the incident, C1, and C2 cohorts with equal total
weight per event/cohort. Alpha is 10 and the activation threshold is exactly zero; validation cannot
tune either. Inputs are report age, cyclical forecast clock, candidate count, distance summaries,
pre-t0 history mean/last/trend/dispersion/valid fraction, and train-fitted freeway/direction one-hot
encodings. The event-node router additionally receives that node's distances and history state.
Forecast Y, C1/C2 outcome at inference, v5c early-excess labels, test, final duration, and post-event
fields are prohibited.

High-impact evaluation uses `early_excess_divergence >= train q75` from the already frozen v5c
triple table. The threshold is computed from train only and transferred unchanged to validation. The
label is used only to report a stratum and is never part of router features. Uncertainty is a 2,000-
draw positive-incident ISO-week cluster bootstrap at 95% confidence.

### Prospective decision rules

The previously recorded oracle tiers remain unchanged and now refer to the **complete 917-sample
validation incident cohort's** event-node oracle all-node improvement relative to fixed A. The first
implementation draft would have applied this gate only to the 618 common triples; that was corrected
before execution because a matched-subset result cannot establish full-validation non-inferiority:

- below 0.3%: stop this frozen branch-routing direction;
- 0.3% to below 1.0%: at most a small baseline-anchored branch-router study may proceed;
- at least 1.0%: a formal selective branch-routing study may proceed only if the router gate also
  passes.

Here "stop" applies only to routing between the frozen `incident_on` and `incident_off` candidates.
This oracle is not an upper bound on an arbitrary future learned residual. A distinct residual expert
would require a separately frozen signed-residual probe protocol; it cannot be authorized or rejected
by relabeling the present branch-toggle results.

The primary event-node router's global and onset conditions use all 917 validation incidents. Its
high-impact condition and C1/C2 placebo conditions necessarily use the 618 common matched incidents.
It must satisfy every condition below:

- the lower confidence bound of all-node improvement is no worse than a 0.1% fixed-A MAE margin;
- the lower confidence bound of candidate-node H1-H6 improvement is above zero;
- high-impact validation incidents have positive point improvement in candidate-node H1-H6;
- its switch-off fraction is between 5% and 95%, excluding trivial always-on/off behavior;
- candidate H7-H12 and all noncandidate predictions are exactly fixed A by construction;
- for each of C1 and C2, the upper confidence bound of candidate H1-H6 harm relative to
  `incident_off` is at most 0.5% of that control's off-mode MAE.

Failure of the router gate prohibits development of this branch-routing design even if an oracle is
large. Passing it authorizes only another train/validation branch-router stage; it does not authorize
test access, establish a learned residual expert, or support an unseen-incident no-degradation
guarantee.

### Implemented artifacts and engineering status

The implementation adds:

- `experiments/chronological/incident_branch_materialize_v6a.json`;
- `experiments/chronological/materialize_incident_branch.py`;
- `experiments/chronological/expert_benefit_audit_v6b.json`;
- `experiments/chronological/audit_expert_benefit.py`;
- focused protocol, counterfactual-construction, oracle, hard-support, and weighted-ridge tests.

The v6b pure-NumPy tests passed locally. The local WSL Python does not contain PyTorch, so v6a's
PyTorch-dependent tests received syntax compilation locally and must run in the existing server
`igstgnn` Conda environment before scientific materialization. The v6a CLI includes a two-triple
`--check` mode that verifies every frozen input and the real checkpoint but produces an engineering-
only status that v6b rejects. No v6a branch-error result was read while these protocols or gates were
written, and no test split was accessed.

## 2026-09-20: Server verification and frozen-input transfer correction

The server ran all five `test_incident_branch_materialization.py` tests and all eight
`test_expert_benefit_audit.py` tests successfully in the existing `igstgnn` Conda environment. This
closes the local PyTorch test gap but does not constitute a v6a scientific result.

A read-only search then established that the server did not contain the earlier v3 primary-control,
v5b secondary-control, or v5c placebo-audit materializations. Those artifacts existed only in the
local research workspace. Re-running their selection pipelines was rejected because it would add
unnecessary recomputation and a new opportunity for cohort drift. The exact frozen local inputs were
rechecked against the v6a/v6b protocol hashes and bundled without modification as
`v6_frozen_inputs_20260920.tar.gz` (168 MiB displayed size; SHA-256
`c634b9860a3ec096cd5fc6ef30a7fbe12870e0ff1c5e2aacf062c9d92ba40bc9`). The bundle contains
`v3_materialized_01`, `v5b_second_materialized_01`, and `v5c_placebo_audit_01` only. It remains an
external research artifact and must not be committed to Git. After upload, the archive hash and the
v6a program's per-file frozen hashes must pass before the bounded engineering check is accepted.

### Engineering-check failure and candidate-mask correction

The first two server engineering checks stopped after successfully materializing the two-sample
`train_incident_full` output. The saved traceback located the failure in an over-strict v6a input
assertion: it required the report-location model support (`any(distances != 0)`) to equal the v3/v5b
frozen affected-node mask. That equality had never been established by the earlier protocols. The
model support is defined by the actual `report_location_v1` tensor, whereas the control mask is the
same-freeway/direction geometric set with an inclusive 10-mile radius.

A full read-only comparison of all common triples found exactly one discrepancy: training positive
sample 1542, station 402510. The incident postmile is 0.249 and the sensor postmile is 10.249, so the
node lies exactly +10 miles from the report. The inclusive geometric rule retains it, while the
location feature's floor-subtracted Gaussian similarity is exactly zero there; all three model
distance channels are consequently zero. The discrepancy affects one node reference in one of
3,106 training triples. All 618 validation triples agree exactly, and no train or validation model-
connected node lies outside the frozen geometric mask.

The corrected v6a protocol keeps `distances != 0` as the candidate definition because that is the
support actually seen by the frozen model. It now requires model-connected candidates to be a subset
of the frozen geometric mask and freezes the sole geometric-only boundary pair by sample and station
identity. Any additional frozen-only pair or any model-connected-outside-frozen pair is rejected.
No oracle/router threshold, cohort, outcome, or model-error result was changed; the failure occurred
before any matched-cohort model output was produced. The v6a protocol SHA-256 is now
`a403dcc0e4e2dd8791a39b0708436c202beee94cb17e3d21dc95e48e88c1d710`, and v6b was rebound to this
corrected prospective input identity.

The corrected server engineering check completed at Git commit `68b3c81` with status
`ENGINEERING_CHECK_PASS` and process exit code zero. It strictly loaded the frozen checkpoint
`b0c712ad9c00007417ccc6ea6268f373852d04063efba2d15d3f49f5497b8e13`, reproduced every frozen
input hash, reported the single expected train geometric-only boundary pair and zero validation
differences, performed no training or gradient computation, and did not read test. All eight
two-sample output files were produced. Their metric values are engineering diagnostics only and
must not be interpreted as evidence of branch benefit.

### Complete v6a materialization

The complete server v6a run finished and saved all branch-error materializations. Frozen A's
`incident_on` validation MAE on all 917 positive samples was `22.686666155323856`, agreeing with the
prospectively required `22.686666155323852` to floating-point precision. The corresponding
`incident_off` MAE was `23.575018080312375`.

The common-triple aggregate all-cell MAEs also favored `incident_on`: validation incident
`21.764078348179172` versus `22.53537939615877`, C1 `21.07278643726087` versus
`21.83962609825824`, and C2 `20.981900671493552` versus `21.74491309274208`. This is not yet the
prospective oracle/router decision, but it cautions against interpreting the existing branch as a
purely accident-specific expert: paired positive report context also improved average prediction on
the routine pseudo-event controls. v6b must now determine whether any report-time-identifiable
event/node onset subset benefits from selective branch suppression while satisfying the frozen
global and placebo gates.

### Pre-v6b real-input feature-source correction

Before running v6b, a real-path audit found that the complete positive manifest intentionally does
not duplicate report freeway/direction fields. The first v6b implementation nevertheless attempted
to read those columns for the 917-sample `incident_full` validation population. Unit tests used
synthetic rows and had not exercised this real manifest schema; executing v6b would therefore have
failed before producing an audit result.

The frozen router already declared freeway and direction as report-time inputs, so dropping those
features after seeing v6a aggregate errors would be an unjustified protocol change. Instead, v6b now
derives the category from the nonzero `report_location_v1` distance support and the exact static
sensor metadata used to create that context. The sensor CSV is bound by SHA-256
`682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094`, and that hash must also appear
in the positive package's frozen `context_manifest.json` source list.

A full read-only check found exactly one freeway/direction category for every one of the 3,604 train
and 917 validation report contexts. All 3,106 train and 618 validation common-triple categories agree
with their frozen manifests, and both splits contain the same six categories: 4-E, 4-W, 24-E, 24-W,
242-N, and 242-S. The v6b implementation now enforces these properties, and focused tests cover
unique-category derivation and mixed-support rejection. This correction changes only how an already
declared report-time feature is recovered from its authoritative inputs; no router feature, target,
threshold, validation outcome, or v6b result was inspected or changed. The corrected v6b protocol
SHA-256 is `dcee4932cb8adde46f3a27737f130c705514153f5a18b01a2706e677c10ef55d`.
