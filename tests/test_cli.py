# tests/test_cli.py
import sys
import subprocess
from pathlib import Path

def run_cli_module(*args: str) -> subprocess.CompletedProcess:
    # Run the CLI module directly to avoid PATH resolution issues
    cmd = [sys.executable, "-m", "epilink.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True)

def run_pkg_main(*args: str) -> subprocess.CompletedProcess:
    # Exercise __main__ (python -m epilink)
    cmd = [sys.executable, "-m", "epilink", *args]
    return subprocess.run(cmd, capture_output=True, text=True)

def test_cli_help():
    cp = run_cli_module("--help")
    assert cp.returncode == 0, cp.stderr
    assert "Estimate transmission linkage probabilities" in cp.stdout

def test_pkg_main_help():
    cp = run_pkg_main("--help")
    assert cp.returncode == 0, cp.stderr
    assert "Estimate transmission linkage probabilities" in cp.stdout

def test_cli_point_single():
    cp = run_cli_module("point", "-g", "2", "-t", "4", "--nsims", "200")
    assert cp.returncode == 0, cp.stderr
    # Should print a single float
    float(cp.stdout.strip())

def test_cli_grid_csv(tmp_path: Path):
    out = tmp_path / "grid.csv"
    cp = run_cli_module(
        "grid",
        "--g-start", "0", "--g-stop", "2", "--g-step", "1",
        "--t-start", "0", "--t-stop", "2", "--t-step", "1",
        "--nsims", "200", "--out", str(out),
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    # header + at least one row
    assert len(lines) >= 2
    assert lines[0].lower().strip() == "g,t,p"
