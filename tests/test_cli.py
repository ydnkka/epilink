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
            "--num-simulations",
            "50",
            "--seed",
            "123",
            "--max-intermediate-hosts",
            "2",
            "-m",
            "0",
            "--grid-min-days",
            "0",
            "--grid-max-days",
            "40",
            "--incubation-shape",
            "5.0",
            "--incubation-scale",
            "1.1",
            "--latent-shape",
            "2.0",
            "--symptomatic-rate",
            "0.4",
            "--symptomatic-shape",
            "1.2",
            "--rel-presymptomatic-infectiousness",
            "2.0",
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
            "--genetic-start",
            "0",
            "--genetic-stop",
            "2",
            "--genetic-step",
            "1",
            "--temporal-start",
            "0",
            "--temporal-stop",
            "2",
            "--temporal-step",
            "1",
            "--num-simulations",
            "20",
            "--seed",
            "123",
            "--max-intermediate-hosts",
            "2",
            "-m",
            "0",
            "--grid-min-days",
            "0",
            "--grid-max-days",
            "40",
            "--incubation-shape",
            "5.0",
            "--incubation-scale",
            "1.1",
            "--latent-shape",
            "2.0",
            "--symptomatic-rate",
            "0.4",
            "--symptomatic-shape",
            "1.2",
            "--rel-presymptomatic-infectiousness",
            "2.0",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    rows = _read_csv(output)
    assert rows[0] == ["GeneticDistance", "0.0", "1.0"]
    assert len(rows) == 3
    assert rows[1][0] == "0.0"
    assert rows[2][0] == "1.0"
