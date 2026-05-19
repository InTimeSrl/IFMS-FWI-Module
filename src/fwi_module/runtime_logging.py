"""Runtime logging helpers for console and per-run log files."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .config import LoggingConfig
from .exceptions import ConfigError
from .utils import ensure_directory


PACKAGE_LOGGER_NAME = "fwi_module"
_LOG_CONTEXT: ContextVar[dict[str, str]] = ContextVar("fwi_module_log_context", default={})
_DEFAULT_CONTEXT = {
    "command": "-",
    "run_id": "-",
    "window": "-",
    "year": "-",
    "phase": "-",
}


@dataclass(frozen=True, slots=True)
class LoggingSession:
    logger: logging.Logger
    log_path: Path | None
    command: str
    run_id: str
    started_at: str


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = dict(_DEFAULT_CONTEXT)
        context.update(_LOG_CONTEXT.get({}))
        for key, value in context.items():
            setattr(record, key, value)
        return True


def configure_run_logging(logging_config: LoggingConfig, *, command: str) -> LoggingSession:
    """Configure package logging for a single CLI invocation."""

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _reset_logger(logger)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    started_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | cmd=%(command)s run=%(run_id)s "
            "window=%(window)s year=%(year)s phase=%(phase)s | %(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    context_filter = _ContextFilter()
    level = _parse_level(logging_config.level)

    if logging_config.enabled and logging_config.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        logger.addHandler(console_handler)

    log_path: Path | None = None
    if logging_config.enabled and logging_config.file:
        log_path = _build_log_path(logging_config, command=command, started_at=started_at, run_id=run_id)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

    return LoggingSession(logger=logger, log_path=log_path, command=command, run_id=run_id, started_at=started_at)


@contextmanager
def bind_log_context(**values: object) -> Iterator[None]:
    current = dict(_LOG_CONTEXT.get({}))
    updated = dict(current)
    for key, value in values.items():
        if value is None:
            updated.pop(key, None)
            continue
        updated[key] = str(value)
    token = _LOG_CONTEXT.set(updated)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def close_run_logging(logger: logging.Logger | None = None) -> None:
    target = logger or logging.getLogger(PACKAGE_LOGGER_NAME)
    _reset_logger(target)


def _build_log_path(logging_config: LoggingConfig, *, command: str, started_at: str, run_id: str) -> Path:
    if logging_config.directory is None:
        raise ConfigError("logging.directory must be configured when file logging is enabled")

    try:
        file_name = logging_config.filename_template.format(command=command, started_at=started_at, run_id=run_id)
    except KeyError as exc:
        raise ConfigError(f"invalid logging.filename_template placeholder: {exc.args[0]}") from exc

    return ensure_directory(logging_config.directory) / file_name


def _parse_level(value: str) -> int:
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ConfigError(f"invalid logging level: {value}")
    return level


def _reset_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
