"""Tests for the LLM inference module."""

import pytest
from httpx import AsyncClient, Response
from unittest.mock import AsyncMock, patch
import but_with_subs.llm
from but_with_subs.llm import LLMConfig, query_llm, LLMServerType, ServerCapabilities


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
async def test_detect_openai_via_header():
    """Test that OpenAI is detected via Server header."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gpt-4"}]},
        headers={"Server": "openai-api"},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.OPENAI
    assert result.supports_json_schema is True
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/v1/models", timeout=5.0
    )


@pytest.mark.asyncio
async def test_detect_ollama_via_header():
    """Test that Ollama is detected via Server header."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "llama3"}]},
        headers={"Server": "ollama"},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.OLLAMA
    assert result.supports_json_schema is False
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/v1/models", timeout=5.0
    )


@pytest.mark.asyncio
async def test_detect_lm_studio_via_header():
    """Test that LM Studio is detected via Server header."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "local-model"}]},
        headers={"Server": "lm-studio"},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.LM_STUDIO
    assert result.supports_json_schema is True


@pytest.mark.asyncio
async def test_detect_groq_via_header():
    """Test that Groq is detected via Server header."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "mixtral-8x7b"}]},
        headers={"Server": "groq"},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.GROQ
    assert result.supports_json_schema is True


@pytest.mark.asyncio
async def test_detect_vllm_via_header():
    """Test that vLLM is detected via Server header."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "meta-llama/Llama-3"}]},
        headers={"Server": "vLLM"},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.VLLM
    assert result.supports_json_schema is True


@pytest.mark.asyncio
async def test_detect_llama_cpp_fallback():
    """Test that llama.cpp is detected when /v1/models fails and /models succeeds."""
    vllm_fail = _make_mock_response(status_code=404)
    llama_success = _make_mock_response(
        status_code=200,
        json_data={"models": [{"id": "gguf-model"}]},
        headers={},
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_success])

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.LLAMA_CPP
    assert result.supports_json_schema is True
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_unknown_when_both_fail():
    """Test that UNKNOWN is returned when neither endpoint succeeds."""
    vllm_fail = _make_mock_response(status_code=404)
    llama_fail = _make_mock_response(status_code=404)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(side_effect=[vllm_fail, llama_fail])

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.UNKNOWN
    assert result.supports_json_schema is False
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_detect_via_model_id_fallback():
    """Test that servers are detected via model ID patterns when headers are absent."""
    mock_client = AsyncMock(spec=AsyncClient)
    # No Server header, but model ID reveals the provider
    mock_response = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gemini-pro"}]},
        headers={},
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await but_with_subs.llm._detect_server_capabilities(
        "http://localhost:8000", mock_client
    )

    assert result.server_type == LLMServerType.GEMINI
    assert result.supports_json_schema is True


@pytest.mark.asyncio
async def test_server_capabilities_cache():
    """Test that detected server capabilities are cached per api_base."""
    # Clear cache for isolation
    but_with_subs.llm._server_capabilities_cache.clear()

    mock_resp = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gpt-4"}]},
        headers={"Server": "openai-api"},
    )

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    # First call should probe
    result1 = await but_with_subs.llm._detect_server_capabilities(
        "http://cache-test:8000", mock_client
    )
    assert result1.server_type == LLMServerType.OPENAI
    call_count_after_first = mock_client.get.call_count

    # Second call with same api_base should use cache
    result2 = await but_with_subs.llm._detect_server_capabilities(
        "http://cache-test:8000", mock_client
    )
    assert result2.server_type == LLMServerType.OPENAI
    # No additional calls — cache hit
    assert mock_client.get.call_count == call_count_after_first

    # Different api_base should probe again
    mock_client2 = AsyncMock(spec=AsyncClient)
    mock_client2.get = AsyncMock(return_value=mock_resp)
    result3 = await but_with_subs.llm._detect_server_capabilities(
        "http://other-server:8000", mock_client2
    )
    assert result3.server_type == LLMServerType.OPENAI
    mock_client2.get.assert_called_once()

    # Clean up
    but_with_subs.llm._server_capabilities_cache.clear()


@pytest.mark.asyncio
async def test_query_llm_uses_cached_capabilities(llm_config):
    """Test that query_llm uses cached capabilities without re-probing."""
    but_with_subs.llm._server_capabilities_cache.clear()
    but_with_subs.llm._server_capabilities_cache[
        llm_config.api_base
    ] = ServerCapabilities(
        server_type=LLMServerType.OPENAI,
        supports_json_schema=True,
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
    # get() should NOT have been called — cache was used
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_query_llm_auto_detects_on_first_call(llm_config):
    """Test that query_llm auto-detects server capabilities on first call."""
    but_with_subs.llm._server_capabilities_cache.clear()

    detection_resp = _make_mock_response(
        status_code=200,
        json_data={"data": [{"id": "gpt-4"}]},
        headers={"Server": "openai-api"},
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
    # Cache should be populated after detection
    cached = but_with_subs.llm._server_capabilities_cache.get(llm_config.api_base)
    assert cached is not None
    assert cached.server_type == LLMServerType.OPENAI
    assert cached.supports_json_schema is True


@pytest.mark.asyncio
async def test_llm_config_default_server_type_none():
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
