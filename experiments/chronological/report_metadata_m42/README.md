# Public development incident locations for M4.2

This is a location-only excerpt of TraffiDent / XTraffic (Xiaochuan Gou et al.,
NeurIPS 2025), under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The local source is [Kaggle version 8](https://www.kaggle.com/datasets/gpxlcj/xtraffic/versions/8).
Its exact SHA256 and the excerpt SHA256 are recorded in manifest.json.

Public-release evidence was checked on 2026-10-10 for the preceding M4.1 bundle:
the [authors' repository](https://github.com/XAITraffic/XTraffic) and their
[Croissant metadata](https://github.com/XAITraffic/XTraffic/blob/main/xtraffic-metadata.json)
explicitly list incidents_y2023.csv with incident timestamps, roads and coordinates,
free accessibility and the dataset license. That Croissant snapshot refers to v5;
the byte-verified local source here is v8, and those version identities are distinct.
The [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/7813e19a86fd73d40f7e811ab15f6d5f-Paper-Datasets_and_Benchmarks_Track.pdf)
states the released data are anonymous with no personal information and provides
the dataset license in Appendix A. This excerpt adds no private observations.

Exactly the same eight-column whitelist as M4.1 is used: source row, incident
entity ID, recorded time, road number, direction, postmile, latitude and longitude.
No person/vehicle identifiers, free text, Type, duration, target traffic or test
records are exported. Conflicting entity IDs are excluded. The filter selects
60-minute lookbacks for the original train/val cutoffs and training cutoffs shifted
20 minutes earlier, using the pre-existing same-road/direction geographic filter.
The original sample identities are frozen separately for train and val.

Recorded time is assumed available and location assumed known then. First-publication
times, report versions and true clearance are unavailable. The collection is neither
an online-certified feed nor an active-incident label. Input batches filter again
at each task's cutoff, including prefix tasks. Validation locations are evaluation
inputs, never fit a normalizer or graph, and no validation traffic labels are bundled.

Server runs reuse this excerpt without downloading the full source. Regeneration
uses prepare_capacity_development_reports.py with a fresh output directory and
the exact original raw source, data package and sensors. No test artifact is used.
