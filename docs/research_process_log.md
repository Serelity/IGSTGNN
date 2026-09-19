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
