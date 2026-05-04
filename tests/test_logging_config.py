"""Tests for but_with_subs.logging_config module."""

import io
import logging

from but_with_subs import logging_config


def _clean_root_handlers() -> None:
    """Remove all handlers from the root logger."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


class TestLoggerExists:
    """Tests for the module-level logger."""

    def test_logger_exists(self) -> None:
        assert logging_config.logger is not None

    def test_logger_has_correct_name(self) -> None:
        assert logging_config.logger.name == "but_with_subs"


class TestConfigureLoggingRootLogger:
    """Tests for configure_logging setting up the root logger."""

    def test_root_logger_level_is_info(self, caplog) -> None:
        _clean_root_handlers()
        logging_config.configure_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_root_logger_has_stream_handler(self, caplog) -> None:
        _clean_root_handlers()
        logging_config.configure_logging()
        root_logger = logging.getLogger()
        stream_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_handler_formatter_is_correct(self) -> None:
        _clean_root_handlers()
        logging_config.configure_logging()
        root_logger = logging.getLogger()
        stream_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        handler = stream_handlers[0]
        formatter = handler.formatter
        assert formatter._fmt == "%(asctime)s \u2022 %(message)s"

    def test_httpx_logger_set_to_warning(self) -> None:
        _clean_root_handlers()
        logging_config.configure_logging()
        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level == logging.WARNING


class TestLoggerCapturesMessages:
    """Tests that the logger captures log messages."""

    def test_logger_captures_info_messages(self, caplog) -> None:
        _clean_root_handlers()
        sink = io.StringIO()
        handler = logging.StreamHandler(stream=sink)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        logging_config.logger.addHandler(handler)
        logging_config.logger.info("test message")
        handler.flush()
        assert "test message" in sink.getvalue()

    def test_logger_captures_debug_messages_not_shown_at_info(self, caplog) -> None:
        _clean_root_handlers()
        sink = io.StringIO()
        handler = logging.StreamHandler(stream=sink)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        logging_config.logger.addHandler(handler)
        logging_config.logger.debug("debug message")
        handler.flush()
        assert "debug message" not in sink.getvalue()


class TestMultipleCallsSafe:
    """Tests that multiple configure_logging calls do not duplicate handlers."""

    def test_multiple_calls_dont_duplicate_handlers(self) -> None:
        _clean_root_handlers()
        logging_config.configure_logging()
        count_first = len(
            [
                h
                for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler)
            ]
        )
        logging_config.configure_logging()
        count_second = len(
            [
                h
                for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler)
            ]
        )
        assert count_first == count_second
