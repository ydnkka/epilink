import logging

import pytest

from epilink_evaluation import cli
from epilink_evaluation.config import load_config
from epilink_evaluation.progress import logged_stage, progress_stage
from epilink_evaluation.workflows.settings import settings_registry


def test_stage_logging_preserves_results_and_partial_status(caplog):
    @logged_stage("Test work", fields=("seed",))
    def work(seed=7):
        return False

    with caplog.at_level(logging.INFO):
        assert work() is False
    assert "Stage started: Test work seed=7" in caplog.text
    assert "Stage finished: Test work seed=7 status=partial" in caplog.text
    assert "elapsed_seconds=" in caplog.text


def test_stage_logging_does_not_swallow_errors(caplog):
    error = ValueError("invalid source data")
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(ValueError) as raised,
        progress_stage(logging.getLogger(__name__), "Input validation"),
    ):
        raise error
    assert raised.value is error
    assert "Stage failed: Input validation" in caplog.text
    assert "error=invalid source data" in caplog.text


def test_named_task_context_does_not_conflict_with_stage_label(caplog):
    @logged_stage("Named task", fields=("name",))
    def work(name):
        return name

    with caplog.at_level(logging.INFO):
        assert work("GD_D") == "GD_D"
    assert "Stage started: Named task name=GD_D" in caplog.text


def test_cli_progress_log_file_closes_handler_on_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "logs" / "progress.log"
    root = logging.getLogger()
    original_handlers = list(root.handlers)

    def fail(arguments):
        logging.getLogger("epilink_evaluation.inputs").info(
            "Input preparation milestone"
        )
        raise ValueError("bad input")

    monkeypatch.setattr(cli, "_execute", fail)
    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="bad input"):
        cli.main(["baseline", "--log-file", str(path)])
    text = path.read_text()
    assert "Execution started:" in text
    assert "Input preparation milestone" in text
    assert "Execution finished: status=failed" in text
    assert root.handlers == original_handlers


def test_smoke_explicitly_includes_raw_and_dated_treecluster():
    config = load_config("evaluation/01_synthetic_baseline/config.yaml")
    config["treecluster"]["enabled"] = False
    smoke = cli.smoke_config(config)
    assert smoke["treecluster"]["enabled"]
    definitions = settings_registry(smoke)
    trees = [d for d in definitions.values() if d["kind"] == "treecluster"]
    assert {(d["data_process"], d["tree_kind"]) for d in trees} == {
        ("deterministic", "raw"),
        ("deterministic", "dated"),
        ("stochastic", "raw"),
        ("stochastic", "dated"),
    }
    assert {d["method"] for d in trees} == set(config["treecluster"]["methods"])
