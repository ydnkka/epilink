import csv
import io

from epilink.cli import main


def _read_csv(output: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(output)))


def test_cli_point_outputs_csv(capsys):
    exit_code = main(
        [
            "point",
            "-g",
            "2",
            "-t",
            "4",
            "--nsims",
            "50",
            "--seed",
            "123",
            "--no-intermediates",
            "2",
            "-m",
            "0",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    rows = _read_csv(output)
    assert rows[0] == ["GeneticDistance", "TemporalDistance", "Probability"]
    assert len(rows) == 2
    assert rows[1][0] == "2.0"
    assert rows[1][1] == "4.0"
    prob = float(rows[1][2])
    assert 0.0 <= prob <= 1.0


def test_cli_grid_outputs_csv(capsys):
    exit_code = main(
        [
            "grid",
            "--g-start",
            "0",
            "--g-stop",
            "2",
            "--g-step",
            "1",
            "--t-start",
            "0",
            "--t-stop",
            "2",
            "--t-step",
            "1",
            "--nsims",
            "20",
            "--seed",
            "123",
            "--no-intermediates",
            "2",
            "-m",
            "0",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    rows = _read_csv(output)
    assert rows[0] == ["GeneticDistance", "0.0", "1.0"]
    assert len(rows) == 3
    assert rows[1][0] == "0.0"
    assert rows[2][0] == "1.0"
