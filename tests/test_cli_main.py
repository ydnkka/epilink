# tests/test_cli_main.py

import sys
import subprocess
from pathlib import Path
import pytest

from epilink.cli import main as cli_main


# ------------------------------
# Helpers to run CLI via subprocess
# ------------------------------

def run_cli_module(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI module directly to avoid PATH issues."""
    cmd = [sys.executable, "-m", "epilink.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True)

def run_pkg_main(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m epilink` to exercise __main__."""
    cmd = [sys.executable, "-m", "epilink", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


# ------------------------------
# Help / basic execution tests
# ------------------------------

def test_pkg_main_help_subprocess():
    """Run `python -m epilink --help` to exercise __main__.py via subprocess."""
    result = run_pkg_main("--help")
    assert result.returncode == 0, result.stderr
    assert "Estimate transmission linkage probabilities" in result.stdout

def test_cli_help_subprocess():
    cp = run_cli_module("--help")
    assert cp.returncode == 0, cp.stderr
    assert "Estimate transmission linkage probabilities" in cp.stdout

def test_cli_point_help():
    """Calling `epilink point -h` should exit cleanly with code 0."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["point", "-h"])
    assert excinfo.value.code == 0

def test_cli_grid_help():
    """Calling `epilink grid -h` should exit cleanly with code 0."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["grid", "-h"])
    assert excinfo.value.code == 0


# ------------------------------
# Normal use tests
# ------------------------------

def test_cli_point_single():
    """Test single 'point' computation via CLI; output should be a float."""
    cp = run_cli_module("point", "-g", "2", "-t", "4", "--nsims", "200")
    assert cp.returncode == 0, cp.stderr
    float(cp.stdout.strip())

def test_cli_grid_csv(tmp_path: Path):
    """Test 'grid' command output to CSV."""
    out = tmp_path / "grid.csv"
    cp = run_cli_module(
        "grid",
        "--g-start", "0", "--g-stop", "2", "--g-step", "1",
        "--t-start", "0", "--t-stop", "2", "--t-step", "1",
        "--nsims", "200",
        "--out", str(out),
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 2  # header + data row
    assert lines[0].lower().strip() == "g,t,p"


# ------------------------------
# Edge-case tests
# ------------------------------

def test_point_negative_genetic_distance():
    """Negative genetic distance should raise an error."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["point", "-g", "-1", "-t", "4", "--nsims", "100"])
    assert excinfo.value.code != 0

def test_point_zero_simulations():
    """Zero simulations should raise an error."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["point", "-g", "2", "-t", "4", "--nsims", "0"])
    assert excinfo.value.code != 0

def test_point_multiple_pairs_runs(capsys):
    """Provide multiple g/t pairs; main should return 0 and print CSV."""
    rc = cli_main(["point", "-g", "1", "2", "-t", "0", "3", "--nsims", "100"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    # header + two rows
    assert out[0].lower().strip() == "g,t,p"
    assert len(out) == 3

def test_grid_invalid_steps(tmp_path: Path):
    """Negative step sizes should raise an error."""
    out = tmp_path / "grid.csv"
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "grid",
            "--g-start", "0", "--g-stop", "5", "--g-step", "-1",
            "--t-start", "0", "--t-stop", "5", "--t-step", "1",
            "--nsims", "100",
            "--out", str(out),
        ])
    assert excinfo.value.code != 0

def test_grid_small_range(tmp_path: Path):
    """Grid with a single point in g and t."""
    out = tmp_path / "grid.csv"
    rc = cli_main([
        "grid",
        "--g-start", "2", "--g-stop", "2", "--g-step", "1",
        "--t-start", "3", "--t-stop", "3", "--t-step", "1",
        "--nsims", "100",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].lower().strip() == "g,t,p"

def test_grid_output_file(tmp_path: Path):
    """Ensure output CSV is created and non-empty."""
    out = tmp_path / "grid.csv"
    rc = cli_main([
        "grid",
        "--g-start", "0", "--g-stop", "2", "--g-step", "1",
        "--t-start", "0", "--t-stop", "2", "--t-step", "1",
        "--nsims", "50",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    assert out.read_text().strip()
