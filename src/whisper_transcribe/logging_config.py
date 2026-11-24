"""Configure human-readable CLI logs and persistent JSONL execution logs."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TextIO

import structlog
from platformdirs import user_log_dir

APP_NAME = "whisper-transcribe"
LOG_FILE_NAME = "whisper-transcribe.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def get_log_file_path() -> Path:
    """Return the platform-specific per-user path for the active log file."""
    return Path(user_log_dir(APP_NAME)) / LOG_FILE_NAME


def _formatter(
    renderer: structlog.typing.Processor,
) -> structlog.stdlib.ProcessorFormatter:
    """Create a formatter that adds the shared fields before rendering."""
    shared_processors: list[structlog.typing.Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def configure_logging(*, verbose: bool, stderr: TextIO | None = None) -> OSError | None:
    """Write structlog events to stderr and a rotating per-user JSONL file.

    If the log directory or file cannot be opened, stderr logging stays active
    and the caller receives the filesystem error so it can show a warning.
    """
    stream = stderr if stderr is not None else sys.stderr
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(
        _formatter(structlog.dev.ConsoleRenderer(colors=stream.isatty())),
    )
    handlers: list[logging.Handler] = [stream_handler]

    file_error: OSError | None = None
    try:
        log_file = get_log_file_path()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        file_error = exc
    else:
        file_handler.setFormatter(_formatter(structlog.processors.JSONRenderer()))
        handlers.append(file_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        handlers=handlers,
        level=logging.DEBUG if verbose else logging.INFO,
        force=True,
    )
    return file_error
