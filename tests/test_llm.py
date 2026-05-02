"""Tests for the LLM inference module."""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, Response
from pydantic import BaseModel

import but_with_subs.llm
from but_with_subs.data_models import LLMConfig, LLMServerType
from but_with_subs.llm import query_llm


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
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
    """Provide a default LLMConfig for testing.

    Returns:
        The default LLMConfig for testing.
    """
    return LLMConfig(
        model="gpt-4", temperature=0.0, max_tokens=64, api_base="http://localhost:8000"
    )


@pytest.mark.asyncio
async def test_query_llm_no_callback_param_at_all(llm_config: LLMConfig) -> None:
    """Test query_llm with a mocked response."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": None}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello",
        config=llm_config,
        client=mock_client,
        server_type=LLMServerType.OPENAI_COMPATIBLE,
    )
    assert result is None


@pytest.mark.asyncio
async def test_query_llm_string_response(llm_config: LLMConfig) -> None:
    """Test query_llm with a mocked string response."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "Hello, world!"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello",
        config=llm_config,
        client=mock_client,
        server_type=LLMServerType.OPENAI_COMPATIBLE,
    )
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_query_llm_string_response_works_without_server_type(
    llm_config: LLMConfig,
) -> None:
    """Test query_llm string response works without server_type (no response_model)."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "Hello, world!"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="translate hello",
        config=llm_config,
        client=mock_client,
        server_type=None,
    )
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_query_llm_raises_valueerror_without_server_type_and_response_model(
    llm_config: LLMConfig,
) -> None:
    """Test that query_llm raises ValueError when server_type is None.

    and response_model is set.
    """

    class MyResponse(BaseModel):
        answer: str

    llm_config.response_model = MyResponse
    mock_client = AsyncMock(spec=AsyncClient)

    with pytest.raises(ValueError, match="requires a known server type"):
        await query_llm(
            prompt="test", config=llm_config, client=mock_client, server_type=None
        )


@pytest.mark.asyncio
async def test_query_llm_passes_explicit_server_type(llm_config: LLMConfig) -> None:
    """Test that query_llm accepts and uses an explicit server_type param."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_response = _make_mock_response(
        json_data={"choices": [{"message": {"content": "explicit type"}}]}
    )
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await query_llm(
        prompt="test",
        config=llm_config,
        client=mock_client,
        server_type=LLMServerType.OPENAI_COMPATIBLE,
    )

    assert result == "explicit type"
    mock_client.get.assert_not_called()


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
async def test_llm_config_default_server_type_none() -> None:
    """Test that LLMConfig defaults server_type to None."""
    config = LLMConfig(
        model="test", temperature=0.5, max_tokens=100, api_base="http://localhost:8000"
    )
    assert config.server_type is None
