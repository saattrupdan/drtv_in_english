"""Tests for the device module."""

from unittest.mock import MagicMock, patch

import pytest

from but_with_subs.device import get_device

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_device_cache():
    """Clear the cached device result before each test.

    get_device() uses @cache, so a previous test that ran on this machine
    (which has MPS) would have cached 'mps'. Without clearing, all
    subsequent tests would return the cached value regardless of mocks.
    """
    from but_with_subs.device import get_device

    get_device.cache_clear()
    yield


@pytest.fixture
def mock_torch_cuda() -> MagicMock:
    """Mock torch.cuda module.

    Returns:
        Mocked torch.cuda module.
    """
    mock_cuda = MagicMock()
    return mock_cuda


@pytest.fixture
def mock_torch_backends() -> MagicMock:
    """Mock torch.backends module.

    Returns:
        Mocked torch.backends module.
    """
    mock_backends = MagicMock()
    mock_mps = MagicMock()
    mock_backends.mps = mock_mps
    return mock_backends


# =============================================================================
# Tests for CUDA device detection
# =============================================================================


class TestCUDADeviceDetection:
    """Tests for CUDA device detection."""

    def test_cuda_available_returns_cuda_device(
        self, mock_torch_cuda: MagicMock
    ) -> None:
        """Test that when CUDA is available, cuda device is returned."""
        with patch("but_with_subs.device.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True

            device = get_device()

            assert device.type == "cuda"
            mock_cuda.is_available.assert_called_once()

    def test_cuda_device_string_is_correct(self) -> None:
        """Test that the CUDA device string is correctly formatted."""
        with patch("but_with_subs.device.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True

            device = get_device()

            assert str(device) == "cuda:0" or str(device) == "cuda"


# =============================================================================
# Tests for MPS (Metal Performance Shaders) detection
# =============================================================================


class TestMPSDeviceDetection:
    """Tests for MPS (Apple Silicon) device detection."""

    def test_mps_available_when_cuda_not_available(self) -> None:
        """Test that MPS is used when CUDA is not available but MPS is."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends") as mock_backends,
        ):
            mock_cuda.is_available.return_value = False
            mock_backends.mps.is_available.return_value = True

            device = get_device()

            assert device.type == "mps"

    def test_mps_device_string_is_correct(self) -> None:
        """Test that the MPS device string is correctly formatted."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends") as mock_backends,
        ):
            mock_cuda.is_available.return_value = False
            mock_backends.mps.is_available.return_value = True

            device = get_device()

            assert str(device) == "mps"

    def test_mps_not_checked_when_cuda_available(self) -> None:
        """Test that MPS is not checked when CUDA is available."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends") as mock_backends,
        ):
            mock_cuda.is_available.return_value = True

            device = get_device()

            assert device.type == "cuda"
            # MPS should not be called when CUDA is available
            mock_backends.mps.is_available.assert_not_called()


# =============================================================================
# Tests for CPU fallback
# =============================================================================


class TestCPUFallback:
    """Tests for CPU fallback when no GPU is available."""

    def test_cpu_fallback_when_no_cuda_or_mps(self) -> None:
        """Test that CPU is used when neither CUDA nor MPS is available."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends") as mock_backends,
        ):
            mock_cuda.is_available.return_value = False
            mock_backends.mps.is_available.return_value = False

            device = get_device()

            assert device.type == "cpu"

    def test_cpu_fallback_returns_cpu_device_string(self) -> None:
        """Test that the CPU device string is correctly formatted."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends") as mock_backends,
        ):
            mock_cuda.is_available.return_value = False
            mock_backends.mps.is_available.return_value = False

            device = get_device()

            assert str(device) == "cpu"

    def test_cpu_fallback_when_cuda_and_mps_both_unavailable(self) -> None:
        """Test CPU fallback when both GPU options are unavailable."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.return_value = False

            device = get_device()

            assert device.type == "cpu"


# =============================================================================
# Tests for error handling
# =============================================================================


class TestDeviceDetectionErrorHandling:
    """Tests for error handling when device detection fails."""

    def test_cuda_is_available_raises_exception(self) -> None:
        """Test handling when torch.cuda.is_available() raises an exception."""
        with patch("but_with_subs.device.torch.cuda") as mock_cuda:
            mock_cuda.is_available.side_effect = RuntimeError(
                "CUDA initialization failed"
            )

            # Should raise the exception
            with pytest.raises(RuntimeError, match="CUDA initialization failed"):
                get_device()

    def test_mps_is_available_raises_exception_when_cuda_unavailable(self) -> None:
        """Test handling when torch.backends.mps.is_available() raises an exception."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.side_effect = RuntimeError(
                "MPS initialization failed"
            )

            # Should raise the exception
            with pytest.raises(RuntimeError, match="MPS initialization failed"):
                get_device()

    def test_torch_device_raises_exception(self) -> None:
        """Test handling when torch.device() raises an exception."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.device") as mock_device,
        ):
            mock_cuda.is_available.return_value = True
            mock_device.side_effect = RuntimeError("Device creation failed")

            # Should raise the exception
            with pytest.raises(RuntimeError, match="Device creation failed"):
                get_device()


# =============================================================================
# Tests for device selection priority
# =============================================================================


class TestDeviceSelectionPriority:
    """Tests for device selection priority order."""

    def test_cuda_has_highest_priority(self) -> None:
        """Test that CUDA has the highest priority when available."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = True
            mock_mps.is_available.return_value = True

            device = get_device()

            # Should select CUDA, not MPS
            assert device.type == "cuda"

    def test_mps_has_second_priority(self) -> None:
        """Test that MPS is selected when CUDA is not available."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.return_value = True

            device = get_device()

            # Should select MPS
            assert device.type == "mps"

    def test_cpu_has_lowest_priority(self) -> None:
        """Test that CPU is selected only when no GPU is available."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.return_value = False

            device = get_device()

            # Should select CPU
            assert device.type == "cpu"


# =============================================================================
# Tests for logging
# =============================================================================


class TestDeviceLogging:
    """Tests for device logging."""

    def test_device_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that the selected device is logged."""
        with patch("but_with_subs.device.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True

            with caplog.at_level("INFO"):
                get_device()

            # Check that device info was logged
            assert any("Using device" in record.message for record in caplog.records)

    def test_cuda_device_is_logged_correctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CUDA device is logged correctly."""
        with patch("but_with_subs.device.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True

            with caplog.at_level("INFO"):
                get_device()

            assert any("cuda" in record.message.lower() for record in caplog.records)

    def test_mps_device_is_logged_correctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that MPS device is logged correctly."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.return_value = True

            with caplog.at_level("INFO"):
                get_device()

            assert any("mps" in record.message.lower() for record in caplog.records)

    def test_cpu_device_is_logged_correctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CPU device is logged correctly."""
        with (
            patch("but_with_subs.device.torch.cuda") as mock_cuda,
            patch("but_with_subs.device.torch.backends.mps") as mock_mps,
        ):
            mock_cuda.is_available.return_value = False
            mock_mps.is_available.return_value = False

            with caplog.at_level("INFO"):
                get_device()

            assert any("cpu" in record.message.lower() for record in caplog.records)
