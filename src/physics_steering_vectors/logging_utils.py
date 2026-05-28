"""Terminal logging helpers.

Local Function:
- Configures package logging and renders full text blocks consistently.

Global Role:
- Makes long experiment runs auditable from terminal output.
"""

import logging
import sys
from pathlib import Path
from typing import Any


PACKAGE_LOGGER_NAME = "physics_steering_vectors"
_HANDLER_MARKER = "_physics_steering_vectors_terminal_handler"
_FILE_HANDLER_MARKER = "_physics_steering_vectors_file_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(config: Any) -> logging.Logger:
    """Configure package logging for terminal and optional file output."""

    level = parse_log_level(config.log_level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)
    terminal_handler = _find_marked_handler(logger, _HANDLER_MARKER)
    if terminal_handler is None:
        terminal_handler = DynamicStdoutHandler()
        setattr(terminal_handler, _HANDLER_MARKER, True)
        logger.addHandler(terminal_handler)

    terminal_handler.setLevel(level)
    terminal_handler.setFormatter(formatter)

    file_handler = configure_file_handler(logger, config.log_file_path, level, formatter)
    logger.debug(
        "Configured logging level=%s full_text=%s log_file_path=%s",
        logging.getLevelName(level),
        config.log_full_text,
        getattr(file_handler, "baseFilename", None),
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a package logger for one module."""

    return logging.getLogger(name)


def parse_log_level(level: str | int) -> int:
    """Convert a configured level name or number into a logging level."""

    if isinstance(level, int):
        return level

    normalized = level.upper()
    numeric = getattr(logging, normalized, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unsupported log level: {level}")
    return numeric


def configure_file_handler(
    logger: logging.Logger,
    log_file_path: str | Path | None,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler | None:
    """Configure the package-owned file handler."""

    existing = _find_marked_handler(logger, _FILE_HANDLER_MARKER)
    if log_file_path is None:
        if existing is not None:
            logger.removeHandler(existing)
            existing.close()
        return None

    path = Path(log_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = str(path)

    if isinstance(existing, logging.FileHandler) and existing.baseFilename == str(path.resolve()):
        handler = existing
    else:
        if existing is not None:
            logger.removeHandler(existing)
            existing.close()
        handler = logging.FileHandler(resolved_path, mode="w", encoding="utf-8")
        setattr(handler, _FILE_HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


class DynamicStdoutHandler(logging.Handler):
    """Logging handler that writes to the current stdout at emit time."""

    terminator = "\n"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            sys.stdout.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass


def log_text_block(
    logger: logging.Logger,
    enabled: bool,
    title: str,
    text: str,
    level: int = logging.DEBUG,
) -> None:
    """Log a full text block with unambiguous terminal delimiters."""

    char_count = len(text)
    if not enabled:
        logger.log(level, "%s omitted from terminal log char_count=%d", title, char_count)
        return

    logger.log(
        level,
        "----- BEGIN %s char_count=%d -----\n%s\n----- END %s -----",
        title,
        char_count,
        text,
        title,
    )


def _find_marked_handler(logger: logging.Logger, marker: str) -> logging.Handler | None:
    """Find a package-owned handler, if configured."""

    for handler in logger.handlers:
        if getattr(handler, marker, False):
            return handler
    return None
