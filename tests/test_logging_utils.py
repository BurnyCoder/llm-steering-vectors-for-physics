import logging

import pytest

from physics_steering_vectors.config import ExperimentConfig
from physics_steering_vectors.logging_utils import configure_logging, log_text_block, parse_log_level


def test_configure_logging_prints_to_terminal_file_and_reuses_handlers(capsys, tmp_path) -> None:
    log_path = tmp_path / "run.log"
    config = ExperimentConfig(log_level="INFO", log_file_path=str(log_path))
    logger = configure_logging(config)

    logger.info("first message")
    configure_logging(config)
    logger.info("second message")

    output = capsys.readouterr().out
    assert output.count("first message") == 1
    assert output.count("second message") == 1
    file_output = log_path.read_text(encoding="utf-8")
    assert file_output.count("first message") == 1
    assert file_output.count("second message") == 1


def test_parse_log_level_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        parse_log_level("NOPE")


def test_log_text_block_prints_full_text_to_terminal_and_file(capsys, tmp_path) -> None:
    log_path = tmp_path / "run.log"
    logger = configure_logging(ExperimentConfig(log_level="DEBUG", log_full_text=True, log_file_path=str(log_path)))

    log_text_block(logger, True, "sample_block", "line 1\nline 2", logging.DEBUG)

    output = capsys.readouterr().out
    assert "BEGIN sample_block char_count=13" in output
    assert "line 1\nline 2" in output
    assert "END sample_block" in output
    file_output = log_path.read_text(encoding="utf-8")
    assert "BEGIN sample_block char_count=13" in file_output
    assert "line 1\nline 2" in file_output
    assert "END sample_block" in file_output


def test_configure_logging_can_disable_file_output(tmp_path) -> None:
    log_path = tmp_path / "run.log"
    logger = configure_logging(ExperimentConfig(log_level="INFO", log_file_path=str(log_path)))
    logger.info("file message")

    configure_logging(ExperimentConfig(log_level="INFO", log_file_path=None))
    logger.info("terminal only message")

    file_output = log_path.read_text(encoding="utf-8")
    assert "file message" in file_output
    assert "terminal only message" not in file_output
