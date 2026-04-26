"""Tests for the LLM module.

This module contains tests for ``LLMProgress``, ``_emit_progress``, and
``query_llm``, covering progress callback emission for successful requests,
HTTP errors, validation errors, unmodified behaviour without a callback,
and the raw string response path.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, Request, Response
from pydantic import BaseModel

from but_with_subs.llm import LLMConfig, query_llm
from but_with_subs.llm_progress import LLMProgress, _emit_progress


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
    json_body: dict | None = None,
    content: str = '{"text": "translation"}',
) -> Response:
    """Create a fake httpx.Response.

    Args:
        status_code:
            The HTTP status code for the response.
        json_body:
            The JSON body content to return.
        content:
            The LLM response content string.

    Returns:
        A Response instance.
    """
    if json_body is None:
        json_body = {"choices": [{"message": {"content": content}}]}
    return Response(
        status_code=status_code,
        json=json_body,
        request=Request("POST", "http://localhost:8000/chat/completions"),
    )


def test_llm_progress_frozen_immutability() -> None:
    """Test that LLMProgress is read-only and cannot be mutated."""
    progress = LLMProgress(status="complete", elapsed_ms=0.0, message="OK")

    with pytest.raises(Exception):
        progress.status = "error"  # type: ignore[assignment]

    with pytest.raises(Exception):
        progress.message = "changed"  # type: ignore[assignment]


def test_emit_progress_calls_callback_once() -> None:
    """Test that _emit_progress invokes the callback with the progress."""

    def callback(p: LLMProgress) -> None:
        received.append(p)

    received: list[LLMProgress] = []

    _emit_progress(callback=callback, status="complete", elapsed_ms=0.0, message="OK")

    assert len(received) == 1
    assert received[0].status == "complete"
    assert received[0].message == "OK"


@pytest.mark.asyncio
async def test_query_llm_no_callback_param_at_all(llm_config: LLMConfig) -> None:
    """Test that omitting the callback parameter entirely works."""
    client = AsyncClient()
    client.post = AsyncMock(return_value=_make_mock_response())

    result = await query_llm(prompt="translate hello", config=llm_config, client=client)

    assert result is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_query_llm_string_response_emits_progress() -> None:
    """Test that the raw string path also emits progress events."""
    config_no_model = LLMConfig(
        model="gpt-4", temperature=0.0, max_tokens=64, api_base="http://localhost:8000"
    )

    received: list[LLMProgress] = []

    def callback(p: LLMProgress) -> None:
        received.append(p)

    client = AsyncClient()
    client.post = AsyncMock(
        return_value=_make_mock_response(
            json_body={"choices": [{"message": {"content": "raw translation"}}]}
        )
    )

    result = await query_llm(
        prompt="translate hello",
        config=config_no_model,
        client=client,
        progress_callback=callback,
    )

    assert len(received) == 3
    assert received[0].status == "request_starting"
    assert received[1].status == "request_sent"
    assert received[2].status == "complete"
    assert result == "raw translation"
    await client.aclose()
