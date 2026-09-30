# Chapter 3 reference-membership correction

All 26 synthetic runs, the early-case sweep, temporal partitions, 60 TreeCluster settings, and Boston summaries have been recalculated.

The corrected reference covers exactly 4,990 cases, including root 4537061. Historical F1 and reference-independent metrics were checked against archived outputs before accepting changes. Baseline pair labels and scores reproduced; AP summaries were retained.
The root has one membership and every other case has two. Input data, dated trees, EpiLink model sources, simulation settings, thresholds, resolution grids, and restart counts are unchanged. The sole analysis-configuration change is Boston's selected resolution.

| Model | Previous F1 | Corrected F1 | Precision | Recall | Baseline resolution | Temporal resolution | Temporal Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDD | 0.629347 | 0.555676 | 0.479434 | 0.660752 | 0.1 | 0.2 | 0.876063 |
| EDS | 0.495397 | 0.476488 | 0.403138 | 0.582466 | 0.1 | 0.1 | 0.755177 |
| ESD | 0.643841 | 0.600954 | 0.651174 | 0.557926 | 0.3 | 0.3 | 0.850589 |
| ESS | 0.558696 | 0.533612 | 0.540086 | 0.527291 | 0.3 | 0.3 | 0.800874 |
| LD | 0.647290 | 0.688429 | 0.737468 | 0.645505 | 0.2 | 0.2 | 0.887327 |
| LS | 0.572059 | 0.570369 | 0.656490 | 0.504222 | 0.1 | 0.2 | 0.926438 |

Boston's minimum-mean-shortfall resolution changed from 0.3 to 0.2. Its regenerated memberships and descriptive checks are recorded in validation.json.

| TreeCluster input | Selected method | Threshold (days) | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| deterministic | max_clade | 35 | 0.746577 | 0.599796 | 0.665186 |
| stochastic | avg_clade | 28 | 0.631356 | 0.600128 | 0.615346 |

The largest corrected sensitivity loss occurs with increased incubation CV under mismatch: all six models lose 55.7–65.6% of baseline F1. At their best settings, precision remains 0.979–0.997 but recall falls to 0.112–0.134. The same partitions exactly reproduce the archived F1 values under the historical reference, and AP is unchanged. The F1 figure's axis now includes these losses in full.

The sparse BCubed calculation is mathematically equivalent to the package definition and was tested against that independent implementation on overlapping and hard assignments. Self-pairs, multiplicities, and case weighting are retained.
Scenario-summary figures keep the existing 95% bootstrap-interval method and use seed 12345 for reproducible rendering; the earlier plot did not specify this seed. Saved baseline AP confidence intervals are unchanged.

## Scope

Chapter 4 now distinguishes the historical Scottish resolution 0.3 from the corrected Boston selection 0.2. Scottish results and their primary-resolution choice were not reanalysed in this correction.

## Reproduction

See README.md for audit commands. archive_manifest.json records input hashes, archived file hashes, package versions, and EpiLink source hashes. Per-run paired resolution sweeps and reproduction checks are in runs/.
