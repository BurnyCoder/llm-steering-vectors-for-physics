import logging

import pytest

from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.logging_utils import configure_logging, log_text_block, parse_log_level


def test_configure_logging_prints_to_terminal_and_reuses_handler(capsys) -> None:
    config = ExperimentConfig(log_level="INFO")
    logger = configure_logging(config)

    logger.info("first message")
    configure_logging(config)
    logger.info("second message")

    output = capsys.readouterr().out
    assert output.count("first message") == 1
    assert output.count("second message") == 1


def test_parse_log_level_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        parse_log_level("NOPE")


def test_log_text_block_prints_full_text(capsys) -> None:
    logger = configure_logging(ExperimentConfig(log_level="DEBUG", log_full_text=True))

    log_text_block(logger, True, "sample_block", "line 1\nline 2", logging.DEBUG)

    output = capsys.readouterr().out
    assert "BEGIN sample_block char_count=13" in output
    assert "line 1\nline 2" in output
    assert "END sample_block" in output
