"""Tests for the LLM module.

This module contains tests for ``query_llm``, covering unmodified behaviour
without a callback and the raw string response path.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, Request, Response
from pydantic import BaseModel

from but_with_subs.llm import (
    LLMConfig,
    LLMServerType,
    _detect_server_type,
    query_llm,
)

import but_with_subs.llm


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
async def test_detect_vllm_server() -> None:
    """Test that a vLLM server is correctly detected via /v1/models endpoint."""
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
async def test_detect_llama_cpp_fallback() -> None:
    """Test that llama.cpp is detected when /v1/models fails and /models succeeds."""
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
async def test_detect_unknown_when_both_fail() -> None:
    """Test that UNKNOWN is returned when neither endpoint succeeds."""
    vllm_fail = _make_mock_response(status_code=404)
    llama_fail = _make_mock_response(status_code=404)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_fail])

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.UNKNOWN
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_unknown_on_vllm_exception() -> None:
    """Test that detection falls through to llama.cpp when /v1/models raises."""
    llama_success = _make_mock_response(
        status_code=200, json_data={"models": [{"id": "gguf-model"}]}
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(
        side_effect=[Exception("connection refused"), llama_success]
    )

    result = await _detect_server_type("http://localhost:8000", mock_client)

    assert result == LLMServerType.LLAMA_CPP
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_server_type_cache() -> None:
    """Test that detected server types are cached per api_base via query_llm."""
    but_with_subs.llm._server_type_cache.clear()

    detection_resp = _make_mock_response(
        status_code=200, json_data={"data": [{"id": "gpt-4"}]}
    )
    query_resp = _make_mock_response(
        json_data={"choices": [{"message": {"content": "cached test"}}]}
    )

    query_client = AsyncMock(spec=AsyncClient)
    query_client.post = AsyncMock(return_value=query_resp)

    # First call: patch AsyncClient constructor to return our detection client,
    # and patch the internal detection get() to return vLLM response
    detect_client = AsyncMock(spec=AsyncClient)
    detect_client.get = AsyncMock(return_value=detection_resp)
    detect_client.aclose = AsyncMock()

    def mock_async_client_init(self, *args, **kwargs) -> None:
        pass

    with patch.object(AsyncClient, "__init__", mock_async_client_init):
        with patch.object(AsyncClient, "aclose", AsyncMock()):
            with patch(
                "but_with_subs.llm.AsyncClient.get",
                side_effect=lambda *a, **kw: detection_resp,
            ):
                config = LLMConfig(
                    model="test",
                    temperature=0.0,
                    max_tokens=100,
                    api_base="http://cache-test:8000",
                )
                result = await query_llm(
                    prompt="test", config=config, client=query_client
                )

    assert result == "cached test"
    assert but_with_subs.llm._server_type_cache.get(
        "http://cache-test:8000"
    ) == LLMServerType.VLLM

    # Second call with same api_base should use cache (no extra detection calls)
    with patch.object(AsyncClient, "__init__", mock_async_client_init):
        with patch.object(AsyncClient, "aclose", AsyncMock()):
            with patch(
                "but_with_subs.llm.AsyncClient.get",
                side_effect=lambda *a, **kw: detection_resp,
            ) as mock_get:
                result2 = await query_llm(
                    prompt="test", config=config, client=query_client
                )

    assert result2 == "cached test"
    # get() should NOT have been called since cache was hit
    mock_get.assert_not_called()

    but_with_subs.llm._server_type_cache.clear()


@pytest.mark.asyncio
async def test_query_llm_uses_cached_server_type(llm_config) -> None:
    """Test that query_llm uses cached server type without re-probing."""
    but_with_subs.llm._server_type_cache.clear()
    but_with_subs.llm._server_type_cache[llm_config.api_base] = LLMServerType.VLLM

    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "cached test"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(prompt="test", config=llm_config, client=mock_client)

    assert result == "cached test"
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_query_llm_auto_detects_on_first_call(llm_config) -> None:
    """Test that query_llm auto-detects server type on first call."""
    but_with_subs.llm._server_type_cache.clear()

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

    async def mock_aclose(self) -> None:
        pass

    with patch.object(AsyncClient, "get", mock_get):
        with patch.object(AsyncClient, "aclose", mock_aclose):
            result = await query_llm(
                prompt="test", config=llm_config, client=mock_client
            )

    assert result == "auto detected"
    assert but_with_subs.llm._server_type_cache.get(
        llm_config.api_base
    ) == LLMServerType.VLLM


@pytest.mark.asyncio
async def test_llm_config_default_server_type_none() -> None:
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
