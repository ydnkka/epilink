"""One-time, verified relocation of the pre-reset evaluation (standard library only).

Run with --execute after reviewing the default inventory. Raw/processed data stay
in place. The archive's data symlink preserves relative legacy input paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "archive/pre_reset_2026-09-30"
TARGETS = (
    "synthetic_baseline", "synthetic_exploration", "synthetic_informativeness",
    "src", "tests", "results", "phylo", "docs", "config.yaml", "Snakefile",
    "README.md", "requirements.txt", "genetic_ties_compatibility.ipynb",
    "phylo_cluster_pipeline.ipynb", "synthetic_treecluster_comparison.ipynb",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if DESTINATION.exists():
        raise SystemExit(f"Archive already exists: {DESTINATION}")
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    paths = []
    for target in TARGETS:
        path = ROOT / target
        if not path.exists():
            raise SystemExit(f"Missing archive source: {path}")
        paths.extend(sorted(p for p in path.rglob("*") if p.is_file())
                     if path.is_dir() else [path])
    print(f"{len(paths):,} files; {sum(p.stat().st_size for p in paths) / 1024**3:.2f} GiB")
    print(f"Destination: {DESTINATION}")
    if not args.execute:
        return
    records = [{"original_path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size, "sha256": digest(p),
                "tracked": str(p.relative_to(ROOT)) in tracked} for p in paths]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "status": "relocating", "targets": TARGETS, "files": records,
                "shared_inputs": "../../data (preserved in place)",
                "note": "Ignored local caches are preserved; embedded absolute paths may refer to the original workspace."}
    DESTINATION.mkdir(parents=True)
    manifest_path = DESTINATION / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for target in TARGETS:
        shutil.move(str(ROOT / target), str(DESTINATION / target))
    (DESTINATION / "data").symlink_to("../../data", target_is_directory=True)
    for record in records:
        saved = DESTINATION / record["original_path"]
        if saved.stat().st_size != record["bytes"] or digest(saved) != record["sha256"]:
            raise RuntimeError(f"Archive verification failed: {saved}")
    manifest["status"] = "verified"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified all {len(records):,} archived files, including local ignored assets.")


if __name__ == "__main__":
    main()
