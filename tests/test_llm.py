"""Tests for the LLM inference module."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, Response

import but_with_subs.llm
from but_with_subs.data_models import LLMConfig, LLMServerType
from but_with_subs.llm import query_llm


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | list = None,
    text: str = "",
    headers: dict | None = None,
) -> AsyncMock:
    mock_response = AsyncMock(spec=Response)
    mock_response.status_code = status_code
    mock_response.is_error = status_code >= 400
    mock_response.json.return_value = json_data if json_data is not None else {}
    mock_response.text = text
    mock_response.headers = headers or {}
    return mock_response


@pytest.fixture
def llm_config() -> LLMConfig:
    """Provide a default LLMConfig for testing."""
    return LLMConfig(
        model="gpt-4",
        temperature=0.0,
        max_tokens=64,
        api_base="http://localhost:8000",
    )


@pytest.mark.asyncio
async def test_query_llm_no_callback_param_at_all(
    llm_config: LLMConfig,
) -> None:
    """Test query_llm with a mocked response."""
    llm_config.server_type = LLMServerType.OPENAI_COMPATIBLE
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": None}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello", config=llm_config, client=mock_client
    )
    assert result is None


@pytest.mark.asyncio
async def test_query_llm_string_response(
    llm_config: LLMConfig,
) -> None:
    """Test query_llm with a mocked string response."""
    llm_config.server_type = LLMServerType.OPENAI_COMPATIBLE
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "Hello, world!"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello", config=llm_config, client=mock_client
    )
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_detect_openai_compatible_via_models() -> None:
    """Test that an OpenAI-compatible server is detected via /models."""
    mock_client = AsyncMock(spec=AsyncClient)
    # Detection checks owned_by != "llamacpp" and presence of "data" key
    mock_response = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4", "owned_by": "openai"}]}
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_type(
        api_base="http://localhost:8000", api_key=None, client=mock_client
    )

    assert result == LLMServerType.OPENAI_COMPATIBLE
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/models",
        timeout=5.0,
        headers={"Authorization": "Bearer None"},
    )


@pytest.mark.asyncio
async def test_detect_llama_cpp_owned_by() -> None:
    """Test that llama.cpp is detected when owned_by=llamacpp."""
    llama_success = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gguf-model", "owned_by": "llamacpp"}]},
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=llama_success)

    result = await but_with_subs.llm._detect_server_type(
        api_base="http://localhost:8000", api_key=None, client=mock_client
    )

    assert result == LLMServerType.LLAMA_CPP
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/models",
        timeout=5.0,
        headers={"Authorization": "Bearer None"},
    )


@pytest.mark.asyncio
async def test_detect_unknown_when_no_data_field() -> None:
    """Test that UNKNOWN is returned when /models returns no data field."""
    mock_response = _make_mock_response(status_code=200, json_data={"models": []})

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_type(
        api_base="http://localhost:8000", api_key=None, client=mock_client
    )

    assert result == LLMServerType.UNKNOWN
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_detect_unknown_on_error() -> None:
    """Test that UNKNOWN is returned when /models returns a non-200 status."""
    mock_response = _make_mock_response(status_code=500)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_type(
        api_base="http://localhost:8000", api_key=None, client=mock_client
    )

    assert result == LLMServerType.UNKNOWN
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_server_type_cache() -> None:
    """Test that detected server types are cached via query_llm."""
    but_with_subs.llm._server_capabilities_cache.clear()

    query_resp = _make_mock_response(
        json_data={"choices": [{"message": {"content": "cached test"}}]}
    )

    query_client = AsyncMock(spec=AsyncClient)
    query_client.post = AsyncMock(return_value=query_resp)

    config = LLMConfig(
        model="test",
        temperature=0.0,
        max_tokens=100,
        api_base="http://cache-test:8000",
        server_type=LLMServerType.OPENAI_COMPATIBLE,
    )

    # Pre-populate the cache
    but_with_subs.llm._server_capabilities_cache["http://cache-test:8000"] = (
        LLMServerType.OPENAI_COMPATIBLE
    )

    result = await query_llm(prompt="test", config=config, client=query_client)

    assert result == "cached test"
    assert (
        but_with_subs.llm._server_capabilities_cache.get("http://cache-test:8000")
        == LLMServerType.OPENAI_COMPATIBLE
    )

    # Second call with same api_base should use cache (no detection probing)
    with patch.object(AsyncClient, "aclose", AsyncMock()):
        result2 = await query_llm(prompt="test", config=config, client=query_client)

    assert result2 == "cached test"

    but_with_subs.llm._server_capabilities_cache.clear()


@pytest.mark.asyncio
async def test_query_llm_uses_cached_server_type(
    llm_config: LLMConfig,
) -> None:
    """Test that query_llm uses cached server type without re-probing."""
    but_with_subs.llm._server_capabilities_cache.clear()
    but_with_subs.llm._server_capabilities_cache[llm_config.api_base] = (
        LLMServerType.OPENAI_COMPATIBLE
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "cached test"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="test", config=llm_config, client=mock_client
    )

    assert result == "cached test"
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_query_llm_auto_detects_on_first_call(
    llm_config: LLMConfig,
) -> None:
    """Test that query_llm auto-detects server type on first call."""
    but_with_subs.llm._server_capabilities_cache.clear()

    detection_resp = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gpt-4", "owned_by": "llamacpp"}]},
    )
    query_resp = _make_mock_response(
        json_data={"choices": [{"message": {"content": "auto detected"}}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.post = AsyncMock(return_value=query_resp)

    async def mock_get(self, *args, **kwargs) -> AsyncMock:
        return detection_resp

    async def mock_aclose(self) -> None:
        pass

    with patch.object(AsyncClient, "get", mock_get):
        with patch.object(AsyncClient, "aclose", mock_aclose):
            result = await query_llm(
                prompt="test", config=llm_config, client=mock_client
            )

    assert result == "auto detected"
    cached = but_with_subs.llm._server_capabilities_cache.get(llm_config.api_base)
    assert cached == LLMServerType.LLAMA_CPP

    assert result == "auto detected"
    cached = but_with_subs.llm._server_capabilities_cache.get(llm_config.api_base)
    assert cached == LLMServerType.LLAMA_CPP


@pytest.mark.asyncio
async def test_llm_config_default_server_type_none() -> None:
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
