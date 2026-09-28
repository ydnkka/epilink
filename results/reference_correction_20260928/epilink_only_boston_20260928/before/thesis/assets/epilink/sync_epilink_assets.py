"""Copy Chapter 3 figure exports from their canonical source repositories.

Plotting code is maintained in epilink-evaluation/src/evaluation/figures.py;
the Chapter 3 schematic is maintained in epilink-evaluation/docs/chapter3_epilink_schematic.tex.
Regenerate the source exports there before running this script.
"""

import argparse
import shutil
from pathlib import Path


def main():
    asset_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=asset_dir.parents[3] / "epilink-evaluation",
    )
    parser.add_argument(
        "--epilink-root",
        type=Path,
        default=asset_dir.parents[3] / "epilink",
    )
    args = parser.parse_args()
    files = []
    for stem in ("surfaces", "baseline", "f1_loss", "ap_loss", "resolution_regret", "boston", "temporal", "perturbation"):
        for suffix in ("pdf", "png"):
            files.append(args.evaluation_root / "results/figures" / f"{stem}.{suffix}")
    for stem, directory in (("treecluster_comparison", "phylo/synthetic"),
                            ("epilink_vs_treecluster_similarity", "phylo")):
        for suffix in ("pdf", "png"):
            files.append(args.evaluation_root / directory / f"{stem}.{suffix}")
    for suffix in ("pdf", "png"):
        files.append(args.evaluation_root / "docs" / f"chapter3_epilink_schematic.{suffix}")
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(f"Missing source export: {source}")
    for source in files:
        target_name = source.name.replace("chapter3_epilink_schematic", "epilink_schematic")
        shutil.copy2(source, asset_dir / target_name)
        print(f"Updated {source.name}")

    descriptive_dir = args.evaluation_root / "results/chapter3_descriptives"
    destination_dir = asset_dir / "ch3_descriptives"
    destination_dir.mkdir(exist_ok=True)
    for name in (
        "boston_cluster_overlaps.csv", "boston_best_cluster_overlaps.csv",
        "boston_named_cluster_overlaps.json", "boston_descriptive_validation.json",
        "boston_overlap_table.tex",
    ):
        shutil.copy2(descriptive_dir / name, destination_dir / name)
        print(f"Updated {name}")


if __name__ == "__main__":
    main()
