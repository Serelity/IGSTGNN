# Train-only incident location metadata excerpt

`train_locations.tsv` is a derived excerpt of TraffiDent / XTraffic v8
`incidents_y2023.csv`, Xiaochuan Gou et al., NeurIPS 2025.
[Source dataset](https://www.kaggle.com/datasets/gpxlcj/xtraffic/versions/8).
This data retains the source **CC BY-NC 4.0** license, rather than the code's
MIT license. [License](https://creativecommons.org/licenses/by-nc/4.0/).
No endorsement by the original authors is implied.

Public-source verification (2026-10-10): the authors' [published Croissant
metadata](https://github.com/XAITraffic/XTraffic/blob/main/xtraffic-metadata.json)
explicitly lists `incidents_y2023.csv`, its timestamp/road/latitude/longitude
fields, free accessibility and the CC BY-NC 4.0 license (that metadata snapshot
describes version 5; the byte-verified local source here is version 8).
Their [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/7813e19a86fd73d40f7e811ab15f6d5f-Paper-Datasets_and_Benchmarks_Track.pdf)
states in its ethics/safeguards checklist that released data are anonymous and
contain no personal information; Appendix A.2/A.3 states the dataset license.
This excerpt adds no person/vehicle identifiers or private observations.

The eight columns retain only a source row index, entity ID, recorded timestamp,
road/direction, postmile and coordinates. Timestamps/numeric road identifiers
are normalized; no duration, Type, DESCRIPTION or LOCATION text is included.
Rows are restricted to the 60-minute lookback of existing original TRAIN
cutoffs, matched to a model road/direction within 5km of a model station and
within 10 source postmile units, plus original trigger IDs. No new supervised
samples or targets are released. The manifest records the exact selection,
476,768-row source hash, filtered row count and excerpt hash. Identical IDs are
deduplicated; conflicting records are excluded and listed.

Recorded time is **assumed** to be availability time and the location is
**assumed** to have been known then, matching the existing offline conditional
protocol. The source supplies no first-publication timestamp or version
history. These are filtered report collections, not certified online feeds,
true active-incident sets, or observations of incident clearance.

To regenerate from the byte-verified original, use
`prepare_incident_capacity_network.py --extract-from-raw ...` with a fresh
`--report-bundle` directory. Server runs reuse this small metadata bundle;
they need neither the entire raw incident CSV nor a download.
