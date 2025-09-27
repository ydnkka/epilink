import sys
import subprocess
from pathlib import Path

def run_cli(*args: str) -> subprocess.CompletedProcess:
    # Use module entry to avoid console script PATH issues
    cmd = [sys.executable, "-m", "epilink.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True)

def test_cli_point_single():
    cp = run_cli("point", "-g", "2", "-t", "4", "--nsims", "200")
    assert cp.returncode == 0, cp.stderr
    float(cp.stdout.strip())  # prints a single float

def test_cli_grid_csv(tmp_path: Path):
    out = tmp_path / "grid.csv"
    cp = run_cli(
        "grid",
        "--g-start", "0", "--g-stop", "2", "--g-step", "1",
        "--t-start", "0", "--t-stop", "2", "--t-step", "1",
        "--nsims", "200", "--out", str(out),
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists()
    text = out.read_text().strip().splitlines()
    assert len(text) >= 2  # header + at least one row

