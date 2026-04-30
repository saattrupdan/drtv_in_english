"""Tests for the LLM module.

This module contains tests for ``query_llm``, covering unmodified behaviour
without a callback and the raw string response path.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, Request, Response
from pydantic import BaseModel

from but_with_subs.llm import LLMConfig, LLMServerType, query_llm


class TranslationResponse(BaseModel):
    """Simple Pydantic model for testing structured LLM responses.

    Attributes:
        text:
            The translated text.
    """

    text: str


@pytest.fixture()
def llm_config() -> LLMConfig:
    """Create an LLMConfig for testing.

    Returns:
        An LLMConfig instance.
    """
    return LLMConfig(
        model="gpt-4", temperature=0.0, max_tokens=64, api_base="http://localhost:8000"
    )


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | list = {},
    content: str = '{"text": "translation"}',
) -> Response:
    """Create a fake httpx.Response.

    Args:
        status_code:
            The HTTP status code for the response.
        json_data:
            The JSON body content to return.
        content:
            The LLM response content string.

    Returns:
        A Response instance.
    """
    if not json_data:
        json_data = {"choices": [{"message": {"content": content}}]}
    return Response(
        status_code=status_code,
        json=json_data,
        request=Request("POST", "http://localhost:8000/chat/completions"),
    )


@pytest.mark.asyncio
async def test_query_llm_no_callback_param_at_all(llm_config: LLMConfig) -> None:
    """Test that omitting the callback parameter entirely works."""
    client = AsyncClient()
    client.post = AsyncMock(return_value=_make_mock_response())

    result = await query_llm(prompt="translate hello", config=llm_config, client=client)

    assert result is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_query_llm_string_response(llm_config: LLMConfig) -> None:
    """Test that the raw string response path works without a callback."""
    client = AsyncClient()
    client.post = AsyncMock(
        return_value=_make_mock_response(
            json_data={"choices": [{"message": {"content": "raw translation"}}]}
        )
    )

    result = await query_llm(prompt="translate hello", config=llm_config, client=client)

    assert result == "raw translation"
    await client.aclose()


@pytest.mark.asyncio
async def test_detect_vllm_server():
    """Test that a vLLM server is correctly detected via /v1/models endpoint."""
    from but_with_subs.llm import _detect_server_type

    mock_resp = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.VLLM
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/v1/models", timeout=5.0
    )


@pytest.mark.asyncio
async def test_detect_llama_cpp_fallback():
    """Test that llama.cpp is detected when /v1/models fails and /models succeeds."""
    from but_with_subs.llm import _detect_server_type

    vllm_fail = _make_mock_response(status_code=404)
    llama_success = _make_mock_response(
        status_code=200, json_data={"models": [{"id": "gguf-model"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_success])

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.LLAMA_CPP
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_unknown_when_both_fail():
    """Test that UNKNOWN is returned when neither endpoint succeeds."""
    from but_with_subs.llm import _detect_server_type

    vllm_fail = _make_mock_response(status_code=404)
    llama_fail = _make_mock_response(status_code=404)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_fail])

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.UNKNOWN
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_unknown_on_vllm_exception():
    """Test that detection falls through to llama.cpp when /v1/models raises."""
    from but_with_subs.llm import _detect_server_type

    llama_success = _make_mock_response(
        status_code=200, json_data={"models": [{"id": "gguf-model"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[Exception("connection refused"), llama_success])

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.LLAMA_CPP
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_server_type_cache():
    """Test that detected server types are cached per api_base."""
    from but_with_subs import llm

    llm._server_type_cache.clear()

    mock_resp = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    result1 = await llm._detect_server_type("http://cache-test:8000", mock_client)
    assert result1 == LLMServerType.VLLM
    call_count_after_first = mock_client.get.call_count

    result2 = await llm._detect_server_type("http://cache-test:8000", mock_client)
    assert result2 == LLMServerType.VLLM
    assert mock_client.get.call_count == call_count_after_first

    mock_client2 = AsyncMock(spec=AsyncClient)
    mock_client2.get = AsyncMock(return_value=mock_resp)
    result3 = await llm._detect_server_type("http://other-server:8000", mock_client2)
    assert result3 == LLMServerType.VLLM
    mock_client2.get.assert_called_once()

    llm._server_type_cache.clear()


@pytest.mark.asyncio
async def test_query_llm_uses_cached_server_type(llm_config):
    """Test that query_llm uses cached server type without re-probing."""
    from but_with_subs import llm

    llm._server_type_cache.clear()
    llm._server_type_cache[llm_config.api_base] = LLMServerType.VLLM

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
    from but_with_subs import llm

    llm._server_type_cache.clear()

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
    assert llm._server_type_cache.get(llm_config.api_base) == LLMServerType.VLLM


@pytest.mark.asyncio
async def test_llm_config_default_server_type_none():
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
