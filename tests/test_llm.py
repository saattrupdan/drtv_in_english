"""Tests for the LLM inference module."""

import pytest
from httpx import AsyncClient, Response
from unittest.mock import AsyncMock, patch
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
def llm_config():
    return LLMConfig(
        model="gpt-4", temperature=0.0, max_tokens=64, api_base="http://localhost:8000"
    )


@pytest.mark.asyncio
async def test_query_llm_no_callback_param_at_all(llm_config):
    """Test query_llm with a mocked response that simulates an empty content response."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={
            "choices": [{"message": {"content": None}}]
        }
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello", config=llm_config, client=mock_client
    )
    assert result is None


@pytest.mark.asyncio
async def test_query_llm_string_response(llm_config):
    """Test query_llm with a mocked response that simulates a string content response."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={
            "choices": [{"message": {"content": "Hello, world!"}}]
        }
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello", config=llm_config, client=mock_client
    )
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_detect_openai_compatible_via_v1_models():
    """Test that an OpenAI-compatible server is detected via /v1/models."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_type(
        "http://localhost:8000", mock_client
    )

    assert result == LLMServerType.OPENAI_COMPATIBLE
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/v1/models", timeout=5.0
    )


@pytest.mark.asyncio
async def test_detect_llama_cpp_fallback():
    """Test that llama.cpp is detected when /v1/models fails and /models succeeds."""
    vllm_fail = _make_mock_response(status_code=404)
    llama_success = _make_mock_response(
        status_code=200, json_data={"models": [{"id": "gguf-model"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_success])

    result = await but_with_subs.llm._detect_server_type(
        "http://localhost:8000", mock_client
    )

    assert result == LLMServerType.LLAMA_CPP
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_unknown_when_both_fail():
    """Test that UNKNOWN is returned when neither endpoint succeeds."""
    vllm_fail = _make_mock_response(status_code=404)
    llama_fail = _make_mock_response(status_code=404)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_fail])

    result = await but_with_subs.llm._detect_server_type(
        "http://localhost:8000", mock_client
    )

    assert result == LLMServerType.UNKNOWN
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_llama_cpp_on_vllm_exception():
    """Test that detection falls through to llama.cpp when /v1/models raises."""
    llama_success = _make_mock_response(
        status_code=200, json_data={"models": [{"id": "gguf-model"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(
        side_effect=[Exception("connection refused"), llama_success]
    )

    result = await but_with_subs.llm._detect_server_type(
        "http://localhost:8000", mock_client
    )

    assert result == LLMServerType.LLAMA_CPP
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_server_type_cache():
    """Test that detected server types are cached via query_llm."""
    but_with_subs.llm._server_capabilities_cache.clear()

    detection_resp = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )
    query_resp = _make_mock_response(
        json_data={"choices": [{"message": {"content": "cached test"}}]}
    )

    query_client = AsyncMock(spec=AsyncClient)
    query_client.post = AsyncMock(return_value=query_resp)

    detect_client = AsyncMock(spec=AsyncClient)
    detect_client.get = AsyncMock(return_value=detection_resp)
    detect_client.aclose = AsyncMock()

    with patch.object(AsyncClient, "__init__", lambda self, *a, **kw: None):
        with patch.object(AsyncClient, "aclose", AsyncMock()):
            with patch("but_with_subs.llm.AsyncClient.get", side_effect=lambda *a, **kw: detection_resp):
                config = LLMConfig(
                    model="test", temperature=0.0, max_tokens=100,
                    api_base="http://cache-test:8000"
                )
                result = await query_llm(prompt="test", config=config, client=query_client)

    assert result == "cached test"
    assert but_with_subs.llm._server_capabilities_cache.get(
        "http://cache-test:8000"
    ) == LLMServerType.OPENAI_COMPATIBLE

    # Second call with same api_base should use cache (no detection probing)
    with patch.object(AsyncClient, "__init__", lambda self, *a, **kw: None):
        with patch.object(AsyncClient, "aclose", AsyncMock()):
            with patch("but_with_subs.llm.AsyncClient.get") as mock_get:
                result2 = await query_llm(prompt="test", config=config, client=query_client)

    assert result2 == "cached test"
    # get() should NOT have been called since cache was hit
    mock_get.assert_not_called()

    but_with_subs.llm._server_capabilities_cache.clear()


@pytest.mark.asyncio
async def test_query_llm_uses_cached_server_type(llm_config):
    """Test that query_llm uses cached server type without re-probing."""
    but_with_subs.llm._server_capabilities_cache.clear()
    but_with_subs.llm._server_capabilities_cache[
        llm_config.api_base
    ] = LLMServerType.OPENAI_COMPATIBLE

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
async def test_query_llm_auto_detects_on_first_call(llm_config):
    """Test that query_llm auto-detects server type on first call."""
    but_with_subs.llm._server_capabilities_cache.clear()

    detection_resp = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )
    query_resp = _make_mock_response(
        json_data={"choices": [{"message": {"content": "auto detected"}}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.post = AsyncMock(return_value=query_resp)

    async def mock_get(self, *args, **kwargs):
        return detection_resp

    async def mock_aclose(self):
        pass

    with patch.object(AsyncClient, "get", mock_get):
        with patch.object(AsyncClient, "aclose", mock_aclose):
            result = await query_llm(
                prompt="test", config=llm_config, client=mock_client
            )

    assert result == "auto detected"
    cached = but_with_subs.llm._server_capabilities_cache.get(llm_config.api_base)
    assert cached == LLMServerType.OPENAI_COMPATIBLE


@pytest.mark.asyncio
async def test_llm_config_default_server_type_none():
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
