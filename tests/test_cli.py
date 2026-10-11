"""Runtime recovery options must fail validation before loading study inputs."""

import pytest

from epilink_evaluation.cli import main


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "-inf"])
def test_phylogeny_timeout_requires_finite_positive_seconds(seconds, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "baseline", "--config", str(tmp_path / "missing.yaml"),
                f"--phylogeny-timeout={seconds}",
            ]
        )
    assert error.value.code == 2
    assert "--phylogeny-timeout must be finite and positive" in capsys.readouterr().err


@pytest.mark.parametrize(
    "threads, message",
    [
        ("0", "--phylogeny-threads must be a positive integer"),
        ("-1", "--phylogeny-threads must be a positive integer"),
        ("2.5", "invalid int value"),
        ("nan", "invalid int value"),
        ("inf", "invalid int value"),
        ("AUTO", "invalid int value"),
    ],
)
def test_phylogeny_threads_requires_positive_integer(threads, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "baseline", "--config", str(tmp_path / "missing.yaml"),
                f"--phylogeny-threads={threads}",
            ]
        )
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--phylogeny-timeout", "--phylogeny-threads"])
@pytest.mark.parametrize(
    "command",
    [
        "diagnostics", "check", "scovmod", "prepare-boston",
        "perturbation", "boston", "reset-outputs",
    ],
)
def test_phylogeny_overrides_are_baseline_only(option, command, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                command, "--config", str(tmp_path / "missing.yaml"),
                option, "4",
            ]
        )
    assert error.value.code == 2
    assert f"{option} applies to baseline only" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--phylogeny-timeout", "--phylogeny-threads"])
@pytest.mark.parametrize("stage", ["prepare", "pairwise", "report"])
def test_phylogeny_overrides_require_inference_stage(option, stage, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "baseline", "--stage", stage,
                "--config", str(tmp_path / "missing.yaml"),
                option, "4",
            ]
        )
    assert error.value.code == 2
    assert f"{option} requires develop, clusters, select, evaluate or all" in (
        capsys.readouterr().err
    )
