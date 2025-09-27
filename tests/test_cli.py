import sys
from pathlib import Path
import subprocess

def run_cli(*args: str) -> subprocess.CompletedProcess:
    exe = [sys.executable, "-m", "epilink.cli"]
    return subprocess.run(exe + list(args), capture_output=True, text=True, check=False)

def test_cli_point_single():
    cp = run_cli("point", "-g", "2", "-t", "4", "--nsims", "200")
    assert cp.returncode == 0
    # Should print a single float
    float(cp.stdout.strip())

def test_cli_grid_csv(tmp_path: Path):
    out = tmp_path / "grid.csv"
    cp = run_cli(
        "grid", "--g-start", "0", "--g-stop", "2", "--g-step", "1",
        "--t-start", "0", "--t-stop", "2", "--t-step", "1",
        "--nsims", "200", "--out", str(out)
    )
    assert cp.returncode == 0
    assert out.exists()
    assert out.read_text().strip() != ""
