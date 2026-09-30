# Pre-reset evaluation archive

The former evaluation was relocated on 2026-09-30. `manifest.json` records every
preserved file (including ignored local artifacts), original path, byte count,
SHA-256, tracking status, and the original Git revision. All 755 files were
verified after relocation. The original project README is `README.md` here.

The three synthetic workflows, `src/evaluation`, tests, notebooks, configurations,
results, phylogenetic products, and manuscript schematics are reference material.
`data` is a relative symlink to the preserved repository input directory.
Git LFS continues to manage the previously tracked data formats. Ignored large
artifacts and the reference-correction archive must accompany this working copy
when moving it; Git alone does not preserve them.

Relative legacy paths can be interpreted from this directory. Embedded absolute
paths in old caches/manifests still describe the original workspace; the archive
is a preserved record, not a migrated cache for the new workflow. New runs use
the `epilink_evaluation` package and separate output directories.
