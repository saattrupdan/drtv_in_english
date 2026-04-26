"""LLM inference module for querying local and remote LLM APIs.

This module provides a generic interface for querying LLM APIs, with a focus on
compatibility with llama.cpp servers. It supports arbitrary response models defined
as Pydantic BaseModels.
"""

import collections.abc as c
import time
import typing as t

from httpx import AsyncClient, Response
from pydantic import BaseModel, ValidationError

from .llm_progress import LLMProgress, _noop_callback
from .logging_config import logger
from .types import ChatCompletionRequest, ChatCompletionResponse, InputMessage


class LLMConfig(BaseModel):
    """Configuration for an LLM API call.

    Attributes:
        model:
            The name of the LLM model to use.
        temperature:
            The temperature to use for the LLM API. Required.
        max_tokens:
            The maximum number of tokens to generate.
        api_base:
            The base URL of the LLM API. Required.
        api_key:
            The API key to use for the LLM API. Not required for local LLMs.
        response_model:
            A Pydantic BaseModel subclass that will be used to parse the response. Can
            be None if no structured generation is used.
    """

    model: str
    temperature: float
    max_tokens: int
    api_base: str
    api_key: str | None = None
    response_model: type[BaseModel] | None = None


async def query_llm[ResponseModel: BaseModel](
    prompt: str,
    config: LLMConfig,
    client: AsyncClient | None = None,
    progress_callback: c.Callable[[LLMProgress], None] | None = None,
) -> ResponseModel | str | None:
    """Query an LLM API with a prompt and return a parsed response.

    Sends the prompt to the specified LLM API endpoint and parses the response
    using the provided response model.

    Args:
        prompt:
            The prompt text to send to the LLM.
        config:
            Configuration for the LLM API call.
        client (optional):
            An optional httpx AsyncClient to use for the request. If not provided,
            a new client will be created.
        progress_callback (optional):
            An optional callback function that receives LLMProgress events.

    Returns:
        The parsed response as an instance of the response model, a string,
        or None if the LLM returns null content.

    Raises:
        ValueError:
            If the response cannot be parsed according to the response model.
    """
    message: InputMessage = {"role": "user", "content": prompt}
    if progress_callback is None:
        progress_callback = _noop_callback

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload: ChatCompletionRequest = {
        "model": config.model,
        "messages": [message],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.response_model is not None:
        payload["response_format"] = {
            "type": "json_object",
            "schema": config.response_model.model_json_schema(),
        }

    url = f"{config.api_base}/chat/completions"

    close_after = False
    if client is None:
        client = AsyncClient()
        close_after = True

    try:
        start_time = time.monotonic()
        _emit_progress(
            callback=progress_callback,
            status="request_starting",
            elapsed_ms=0.0,
            message="Sending request...",
        )

        response: Response = await client.post(
            url=url, json=payload, headers=headers, timeout=600
        )
        if response.is_error:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"LLM API error {response.status_code}: {response.text}")
            _emit_progress(
                callback=progress_callback,
                status="error",
                elapsed_ms=elapsed_ms,
                message="HTTP error from LLM API",
                error=response.text,
            )
            response.raise_for_status()

        response_data: ChatCompletionResponse = response.json()
        elapsed_ms = (time.monotonic() - start_time) * 1000
        _emit_progress(
            callback=progress_callback,
            status="request_sent",
            elapsed_ms=elapsed_ms,
            message="Response received",
        )

        content: str = response_data["choices"][0]["message"]["content"]

        # Guard against null content from the LLM
        if content is None:
            logger.warning("LLM returned null content, returning raw response")
            _emit_progress(
                callback=progress_callback,
                status="complete",
                elapsed_ms=elapsed_ms,
                message="Prompt formatted successfully",
                model=config.model,
            )
            return content

        if config.response_model is None:
            _emit_progress(
                callback=progress_callback,
                status="complete",
                elapsed_ms=elapsed_ms,
                message="Prompt formatted successfully",
                model=config.model,
            )
            return content

        try:
            parsed: ResponseModel = t.cast(
                ResponseModel, config.response_model.model_validate_json(content)
            )
        except ValidationError as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                f"Failed to parse LLM response with {config.response_model.__name__}: "
                f"{exc}"
            )
            _emit_progress(
                callback=progress_callback,
                status="error",
                elapsed_ms=elapsed_ms,
                message=f"Failed to parse LLM response with "
                f"{config.response_model.__name__}",
                error=str(exc),
            )
            raise ValueError(
                f"Failed to parse LLM response with {config.response_model.__name__}"
            ) from exc

        _emit_progress(
            callback=progress_callback,
            status="complete",
            elapsed_ms=elapsed_ms,
            message="Prompt formatted successfully",
            model=config.model,
        )
        return parsed
    finally:
        if close_after:
            await client.aclose()


def _emit_progress(
    callback: c.Callable[[LLMProgress], None] | None,
    status: str,
    elapsed_ms: float,
    message: str,
    model: str | None = None,
    error: str | None = None,
) -> None:
    """Emit a progress event if a callback is provided.

    Args:
        callback:
            The callback function to invoke, or None.
        status:
            The status string for the progress event.
        elapsed_ms:
            Elapsed time in milliseconds since the call started.
        message:
            A human-readable description of the current state.
        model (optional):
            The name of the model being queried.
        error (optional):
            An error message, if the call failed.
    """
    if callback is not None:
        callback(
            LLMProgress(
                status=status,
                elapsed_ms=elapsed_ms,
                message=message,
                model=model,
                error=error,
            )
        )
