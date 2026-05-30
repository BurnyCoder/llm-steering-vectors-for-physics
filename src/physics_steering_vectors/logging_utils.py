"""Terminal logging helpers.

Sources Used:
- Python logging docs: https://docs.python.org/3/library/logging.html

Local Function:
- Configures package logging and renders full text blocks consistently.

Global Role:
- Makes long experiment runs auditable from terminal output.
"""

import logging  # Source: Python logging docs. Local: levels, handlers, and formatting. Global: central experiment audit trail.
import sys  # Local: write to the active stdout stream. Global: keep terminal capture reliable in tests and notebooks.
from pathlib import Path  # Local: normalize log-file paths. Global: create stable artifact directories.
from typing import Any  # Local: accept config-like objects. Global: avoid coupling this utility to one dataclass.


PACKAGE_LOGGER_NAME = __package__ or __name__.partition(".")[0]  # Local: infer package logger namespace. Global: one logging root for all modules.
_HANDLER_MARKER_PREFIX = f"_{PACKAGE_LOGGER_NAME.replace('.', '_')}"  # Local: namespace handler markers. Global: avoid marker collisions.
_HANDLER_MARKER = f"{_HANDLER_MARKER_PREFIX}_terminal_handler"  # Local: identify owned stdout handler. Global: avoid duplicate terminal logs.
_FILE_HANDLER_MARKER = f"{_HANDLER_MARKER_PREFIX}_file_handler"  # Local: identify owned file handler. Global: safely reconfigure log files.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"  # Local: uniform line format. Global: readable multi-phase run logs.


def configure_logging(config: Any) -> logging.Logger:
    """Configure package logging for terminal and optional file output."""

    level = parse_log_level(config.log_level)  # Local: convert configured level. Global: consistent verbosity across handlers.
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)  # Local: get package root logger. Global: shared audit channel.
    logger.setLevel(level)  # Local: set logger threshold. Global: align module logs with protocol config.
    logger.propagate = False  # Local: prevent parent logger duplication. Global: keep terminal/file output single-counted.

    formatter = logging.Formatter(_LOG_FORMAT)  # Source: Python logging Formatter. Local: build shared formatter. Global: stable log schema.
    terminal_handler = _find_marked_handler(logger, _HANDLER_MARKER)  # Local: reuse stdout handler. Global: idempotent setup.
    if terminal_handler is None:  # Local: first logging configuration. Global: install terminal output once.
        terminal_handler = DynamicStdoutHandler()  # Local: defer stdout lookup until emit. Global: support capture-aware terminals.
        setattr(terminal_handler, _HANDLER_MARKER, True)  # Local: mark ownership. Global: distinguish package handler from user handlers.
        logger.addHandler(terminal_handler)  # Local: attach terminal output. Global: make experiment progress visible.

    terminal_handler.setLevel(level)  # Local: apply current verbosity. Global: keep reconfiguration effective.
    terminal_handler.setFormatter(formatter)  # Local: apply shared format. Global: match file and terminal logs.

    file_handler = configure_file_handler(logger, config.log_file_path, level, formatter)  # Local: optional persistent log. Global: preserve long-run output.
    logger.debug(
        "Configured logging level=%s full_text=%s log_file_path=%s",
        logging.getLevelName(level),
        config.log_full_text,
        getattr(file_handler, "baseFilename", None),
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a package logger for one module."""

    return logging.getLogger(name)  # Local: module-specific logger. Global: namespaced trace of experiment phases.


def parse_log_level(level: str | int) -> int:
    """Convert a configured level name or number into a logging level."""

    if isinstance(level, int):  # Local: numeric level already supplied. Global: allow advanced logging configuration.
        return level  # Local: preserve explicit integer. Global: no lossy conversion.

    normalized = level.upper()  # Local: accept lowercase config strings. Global: more forgiving protocol settings.
    numeric = getattr(logging, normalized, None)  # Source: Python logging level constants. Local: resolve name. Global: use standard levels.
    if not isinstance(numeric, int):  # Local: reject unknown names. Global: fail fast on misconfigured runs.
        raise ValueError(f"Unsupported log level: {level}")  # Local: explain bad setting. Global: avoid silent logging gaps.
    return numeric  # Local: return standard integer level. Global: feed logger/handler configuration.


def configure_file_handler(
    logger: logging.Logger,
    log_file_path: str | Path | None,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler | None:
    """Configure the package-owned file handler."""

    existing = _find_marked_handler(logger, _FILE_HANDLER_MARKER)  # Local: locate current file handler. Global: idempotent file logging.
    if log_file_path is None:  # Local: file logging disabled. Global: support terminal-only runs.
        if existing is not None:  # Local: stale file handler present. Global: stop writing old artifact files.
            logger.removeHandler(existing)  # Local: detach handler. Global: prevent future writes to disabled log path.
            existing.close()  # Local: release file descriptor. Global: flush and close artifact cleanly.
        return None  # Local: report no file handler. Global: debug message can show terminal-only setup.

    path = Path(log_file_path)  # Local: normalize configured path. Global: support string or Path settings.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: ensure directory exists. Global: make artifact logging turnkey.
    resolved_path = str(path)  # Local: FileHandler-compatible path. Global: preserve configured relative artifact location.

    if isinstance(existing, logging.FileHandler) and existing.baseFilename == str(path.resolve()):  # Local: same target already active. Global: avoid duplicate file logs.
        handler = existing  # Local: reuse current file handler. Global: preserve idempotent configuration.
    else:
        if existing is not None:  # Local: replace old file target. Global: respect changed config path.
            logger.removeHandler(existing)  # Local: detach old handler. Global: prevent split writes.
            existing.close()  # Local: close old file. Global: finish previous artifact cleanly.
        handler = logging.FileHandler(resolved_path, mode="w", encoding="utf-8")  # Source: Python logging FileHandler. Local: open run log. Global: persist terminal audit trail.
        setattr(handler, _FILE_HANDLER_MARKER, True)  # Local: mark ownership. Global: make future reconfiguration safe.
        logger.addHandler(handler)  # Local: attach file output. Global: write run logs to artifacts.

    handler.setLevel(level)  # Local: apply configured verbosity. Global: keep file and terminal thresholds aligned.
    handler.setFormatter(formatter)  # Local: apply shared format. Global: consistent log parsing.
    return handler  # Local: expose active file handler. Global: include resolved path in debug output.


class DynamicStdoutHandler(logging.Handler):
    """Logging handler that writes to the current stdout at emit time."""

    terminator = "\n"  # Local: mirror StreamHandler line ending. Global: readable terminal log lines.

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)  # Source: logging Handler API. Local: render record. Global: use configured log schema.
            sys.stdout.write(message + self.terminator)  # Local: write to current stdout. Global: cooperate with capture/redirection.
            self.flush()  # Local: push line immediately. Global: make long-running progress visible.
        except Exception:
            self.handleError(record)  # Source: logging Handler API. Local: standard error path. Global: avoid crashing experiment logging.

    def flush(self) -> None:
        try:
            sys.stdout.flush()  # Local: flush current stdout. Global: keep terminal/file capture timely.
        except Exception:
            pass  # Local: ignore flush failures. Global: logging problems should not stop experiment execution.


def log_text_block(
    logger: logging.Logger,
    enabled: bool,
    title: str,
    text: str,
    level: int = logging.DEBUG,
) -> None:
    """Log a full text block with unambiguous terminal delimiters."""

    char_count = len(text)  # Local: summarize block length. Global: audit omitted or included full-text logs.
    if not enabled:  # Local: respect privacy/noise setting. Global: allow lighter terminal output.
        logger.log(level, "%s omitted from terminal log char_count=%d", title, char_count)  # Local: note omission. Global: preserve traceability.
        return  # Local: skip raw block. Global: avoid leaking full prompts/responses when disabled.

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

    for handler in logger.handlers:  # Local: inspect attached handlers. Global: reuse package-owned outputs.
        if getattr(handler, marker, False):  # Local: marker identifies our handler. Global: leave external handlers untouched.
            return handler  # Local: found matching handler. Global: keep configuration idempotent.
    return None  # Local: no owned handler found. Global: caller can install one.
