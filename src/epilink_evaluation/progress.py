"""Consistent stage, task-count, and timing logs across evaluation workflows."""

import inspect
import logging
from contextlib import contextmanager
from functools import wraps
from time import perf_counter


def _details(fields):
    return "".join(
        f" {key}={value}" for key, value in fields.items() if value is not None
    )


@contextmanager
def progress_stage(logger, name, /, **fields):
    started = perf_counter()
    details = _details(fields)
    state = {"status": "complete"}
    logger.info("Stage started: %s%s", name, details)
    try:
        yield state
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        logger.log(
            logging.WARNING if status == "interrupted" else logging.ERROR,
            "Stage %s: %s%s elapsed_seconds=%.3f error=%s",
            status,
            name,
            details,
            perf_counter() - started,
            error,
        )
        raise
    else:
        logger.info(
            "Stage finished: %s%s status=%s elapsed_seconds=%.3f",
            name,
            details,
            state["status"],
            perf_counter() - started,
        )


def logged_stage(name, *, fields=()):
    def decorate(function):
        signature = inspect.signature(function)
        logger = logging.getLogger(function.__module__)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            details = {key: bound.arguments.get(key) for key in fields}
            with progress_stage(logger, name, **details) as state:
                result = function(*args, **kwargs)
                if result is False:
                    state["status"] = "partial"
                elif isinstance(result, dict) and result.get("status") in (
                    "complete",
                    "partial",
                    "failed",
                    "running",
                ):
                    state["status"] = result["status"]
                return result

        return wrapped

    return decorate


def log_progress(logger, name, completed, total, /, **fields):
    """Report completed task counts, without presenting them as a runtime ETA."""
    logger.info(
        "Progress: %s completed=%s/%s%s", name, completed, total, _details(fields)
    )
