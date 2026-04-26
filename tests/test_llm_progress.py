"""Tests for the SharedProgress class and SharedProgressCallback factory.

This module contains tests for ``SharedProgress`` and
``SharedProgressCallback``, covering thread-safe updates, context manager
protocol, callback factory behaviour, and integration with LLM progress
events.
"""

import threading
from unittest.mock import MagicMock, patch

from but_with_subs.llm_progress import (
    LLMProgress,
    SharedProgress,
    SharedProgressCallback,
)

# ---------------------------------------------------------------------------
# SharedProgress tests
# ---------------------------------------------------------------------------


def test_shared_progress_initialises_with_total() -> None:
    """Test that SharedProgress initialises with the correct total."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm:
        SharedProgress(total=5, desc="Testing")

        mock_tqdm.assert_called_once_with(total=5, desc="Testing")


def test_shared_progress_initialises_without_desc() -> None:
    """Test that SharedProgress initialises with ``desc=None`` when omitted."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm:
        SharedProgress(total=10)

        mock_tqdm.assert_called_once_with(total=10, desc=None)


def test_shared_progress_update_calls_tqdm_update() -> None:
    """Test that SharedProgress.update forwards to tqdm.update."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=5)
        sp.update(3)

        mock_instance.update.assert_called_once_with(3)


def test_shared_progress_update_defaults_to_one() -> None:
    """Test that SharedProgress.update defaults to advancing by 1."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=5)
        sp.update()

        mock_instance.update.assert_called_once_with(1)


def test_shared_progress_set_description_calls_tqdm_set_description() -> None:
    """Test that SharedProgress.set_description forwards to tqdm."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=5)
        sp.set_description("Batch 2/3 complete")

        mock_instance.set_description.assert_called_once_with("Batch 2/3 complete")


def test_shared_progress_close_calls_tqdm_close() -> None:
    """Test that SharedProgress.close forwards to tqdm.close."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=5)
        sp.close()

        mock_instance.close.assert_called_once()


def test_shared_progress_context_manager_enter_returns_self() -> None:
    """Test that SharedProgress.__enter__ returns the instance itself."""
    with patch("but_with_subs.llm_progress.tqdm"):
        sp = SharedProgress(total=3)

        with sp as ctx:
            assert ctx is sp


def test_shared_progress_context_manager_exit_calls_close() -> None:
    """Test that SharedProgress.__exit__ calls close on the tqdm instance."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=3)

        with sp:
            pass

        mock_instance.close.assert_called_once()


def test_shared_progress_thread_safe_updates() -> None:
    """Test that SharedProgress.update is safe across threads."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=100)
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                for _ in range(n):
                    sp.update(1)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mock_instance.update.call_count == 100


# ---------------------------------------------------------------------------
# SharedProgressCallback factory tests
# ---------------------------------------------------------------------------


def test_shared_progress_callback_calls_update_on_complete() -> None:
    """Test that SharedProgressCallback calls update(1) on complete status."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=1)
        callback = SharedProgressCallback(sp)

        callback(LLMProgress(status="complete", elapsed_ms=0.0, message="OK"))

        mock_instance.update.assert_called_once_with(1)


def test_shared_progress_callback_calls_set_description_on_request_starting() -> None:
    """Test that SharedProgressCallback calls set_description on request_starting."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=1)
        callback = SharedProgressCallback(sp)

        callback(
            LLMProgress(
                status="request_starting", elapsed_ms=0.0, message="Sending request..."
            )
        )

        mock_instance.set_description.assert_called_once_with("Sending request...")


def test_shared_progress_callback_calls_set_description_on_request_sent() -> None:
    """Test that SharedProgressCallback calls set_description on request_sent."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=1)
        callback = SharedProgressCallback(sp)

        callback(
            LLMProgress(
                status="request_sent", elapsed_ms=100.0, message="Response received"
            )
        )

        mock_instance.set_description.assert_called_once_with("Response received")


def test_shared_progress_callback_calls_set_description_on_error() -> None:
    """Test that SharedProgressCallback calls set_description on error status."""
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=1)
        callback = SharedProgressCallback(sp)

        callback(
            LLMProgress(
                status="error", elapsed_ms=50.0, message="HTTP error from LLM API"
            )
        )

        mock_instance.set_description.assert_called_once_with("HTTP error from LLM API")


def test_shared_progress_callback_does_not_update_on_intermediate_statuses() -> None:
    """Test that SharedProgressCallback does not call update on intermediate statuses.

    Verifies that only ``complete`` triggers ``update`` while intermediate
    statuses trigger ``set_description`` instead.
    """
    with patch("but_with_subs.llm_progress.tqdm") as mock_tqdm_cls:
        mock_instance = MagicMock()
        mock_tqdm_cls.return_value = mock_instance

        sp = SharedProgress(total=1)
        callback = SharedProgressCallback(sp)

        callback(
            LLMProgress(
                status="request_starting", elapsed_ms=0.0, message="Sending request..."
            )
        )

        mock_instance.update.assert_not_called()
        mock_instance.set_description.assert_called_once()
