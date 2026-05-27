"""Terminal logging helpers.

Local Function:
- Configures package logging and renders full text blocks consistently.

Global Role:
- Makes long experiment runs auditable from terminal output.
"""

import logging
import sys
from typing import Any


PACKAGE_LOGGER_NAME = "physics_steering_vectors"
_HANDLER_MARKER = "_physics_steering_vectors_terminal_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(config: Any) -> logging.Logger:
    """Configure package logging for terminal output."""

    level = parse_log_level(config.log_level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    handler = _find_terminal_handler(logger)
    if handler is None:
        handler = DynamicStdoutHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.debug("Configured terminal logging level=%s full_text=%s", logging.getLevelName(level), config.log_full_text)
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


def _find_terminal_handler(logger: logging.Logger) -> logging.Handler | None:
    """Find the package-owned terminal handler, if configured."""

    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler
    return None
