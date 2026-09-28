# EpiLink-only Boston resolution selection

The empirical Boston selection now minimises equally weighted mean F1 shortfall across EDD, EDS, ESD, and ESS. These four configurations select 0.3 (mean shortfall 0.012553), compared with 0.059553 at 0.2. The six-model criterion and its 0.2 result remain recorded in the parent boston_resolution_selection.json for provenance. Logistic benchmark and temporal results are unchanged.

Boston was rerun with the existing seed 12345, threshold, restarts, and model. All 772 sequence memberships and all three summary tables exactly reproduce the original archived 0.3 results. There are 77 non-singleton clusters containing 641 sequences and 131 singletons; five clusters include Conference or SNF exposures. Exposure summaries, figures, and TreeCluster overlaps were regenerated and checked.

The thesis presents the final method without draft correction history. Reference definitions and implementation validation remain documented. Scottish analyses and unrelated manuscript edits are unchanged.

Validation: 11 regression tests passed; the numerical audit passed; the plotted mean uses exactly the four EpiLink configurations; 25 synced thesis assets match their canonical outputs; 36 other numerical/figure files remain byte-identical. The thesis compiled to 247 pages, with 15 affected pages visually inspected and no new layout defects or unresolved references.

The before/ directory and archive_manifest.json preserve 71 files from before this revision. validation.json records checks; final_manifest.json records final source, output, and thesis hashes. Logs contain the Boston rerun, membership recovery, unit tests, scientific audit, workflow dry run, and thesis build.
