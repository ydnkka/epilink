"""Copy the verified baseline backbone to the fresh derived-input location."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/pre_reset_2026-09-30/results/scovmod/scovmod_tree.gml"
DESTINATION = ROOT / "data/derived/scovmod/transmission_tree.gml"


def main():
    checksum = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = json.loads((SOURCE.parents[2] / "manifest.json").read_text())
    expected = next(row["sha256"] for row in manifest["files"]
                    if row["original_path"] == "results/scovmod/scovmod_tree.gml")
    if checksum != expected:
        raise ValueError("Archived transmission tree checksum mismatch")
    if DESTINATION.exists():
        if hashlib.sha256(DESTINATION.read_bytes()).hexdigest() != checksum:
            raise ValueError("Refusing to replace an existing different backbone")
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE, DESTINATION)
    (DESTINATION.parent / "source.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)), "sha256": checksum,
        "role": "fixed 4990-case baseline backbone", "git_revision": manifest["git_revision"],
        "regeneration": "epilink-evaluate prepare-tree --config synthetic_baseline/config.yaml",
    }, indent=2) + "\n")
    print(f"Promoted and verified: {DESTINATION}")


if __name__ == "__main__":
    main()
