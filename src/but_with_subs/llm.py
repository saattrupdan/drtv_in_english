"""LLM inference module for querying local and remote LLM APIs.

This module provides a generic interface for querying LLM APIs, with a focus on
compatibility with llama.cpp servers. It supports arbitrary response models defined
as Pydantic BaseModels.
"""

import time
import typing as t
from enum import Enum

from httpx import AsyncClient, Response
from pydantic import BaseModel, ValidationError

from .logging_config import logger
from .types import ChatCompletionRequest, ChatCompletionResponse, InputMessage


class LLMServerType(str, Enum):
    """Detected type of LLM backend server."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    UNKNOWN = "unknown"


# Cache detected server types per api_base to avoid repeated probing
_server_type_cache: dict[str, LLMServerType] = {}


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
        server_type:
            The detected type of LLM backend server. Auto-detected on first
            query if not explicitly set. Not required for operation.
        response_model:
            A Pydantic BaseModel subclass that will be used to parse the response. Can
            be None if no structured generation is used.
    """

    model: str
    temperature: float
    max_tokens: int
    api_base: str
    api_key: str | None = None
    server_type: LLMServerType | None = None
    response_model: type[BaseModel] | None = None


async def _detect_server_type(api_base: str, client: AsyncClient) -> LLMServerType:
    """Detect whether the LLM backend is vLLM or llama.cpp by probing endpoints.

    vLLM exposes /v1/models with an OpenAI-style response containing a "data" key.
    llama.cpp exposes /models with a simpler response containing a "models" key.

    Args:
        api_base: The base URL of the LLM API.
        client: An httpx AsyncClient to use for the request.

    Returns:
        LLMServerType indicating the detected backend, or UNKNOWN if neither matches.
    """
    # Try vLLM endpoint first
    try:
        resp = await client.get(f"{api_base}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                logger.debug("Detected vLLM server at %s", api_base)
                return LLMServerType.VLLM
    except Exception:
        pass

    # Fall back to llama.cpp endpoint
    try:
        resp = await client.get(f"{api_base}/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "models" in data:
                logger.debug("Detected llama.cpp server at %s", api_base)
                return LLMServerType.LLAMA_CPP
    except Exception:
        pass

    logger.warning(
        "Could not detect LLM server type at %s, defaulting to UNKNOWN", api_base
    )
    return LLMServerType.UNKNOWN


async def query_llm[ResponseModel: BaseModel](
    prompt: str, config: LLMConfig, client: AsyncClient | None = None
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

    Returns:
        The parsed response as an instance of the response model, a string,
        or None if the LLM returns null content.

    Raises:
        ValueError:
            If the response cannot be parsed according to the response model.
    """
    message: InputMessage = {"role": "user", "content": prompt}

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
            "type": "json_schema",
            "json_schema": {
                "name": config.response_model.__name__,
                "schema": config.response_model.model_json_schema(),
            },
        }

    url = f"{config.api_base}/chat/completions"

    close_after = False
    if client is None:
        client = AsyncClient()
        close_after = True

    try:
        start_time = time.monotonic()

        # Auto-detect server type if not already known
        if config.server_type is None:
            cached = _server_type_cache.get(config.api_base)
            if cached is not None:
                effective_server_type = cached
            else:
                # Create a temporary client for detection
                detect_client = AsyncClient()
                try:
                    effective_server_type = await _detect_server_type(
                        config.api_base, detect_client
                    )
                    _server_type_cache[config.api_base] = effective_server_type
                finally:
                    await detect_client.aclose()
        else:
            effective_server_type = config.server_type

        logger.info(
            "Using LLM server type: %s for %s",
            effective_server_type.value,
            config.api_base,
        )

        response: Response = await client.post(
            url=url, json=payload, headers=headers, timeout=600
        )

        if response.is_error:
            logger.error(f"LLM API error {response.status_code}: {response.text}")
            response.raise_for_status()

        response_data: ChatCompletionResponse = response.json()

        # Log raw response for diagnosing null content issues
        logger.debug("Raw LLM response: %s", response_data)

        # Defensive extraction of content with full structure logging on failure
        choices = response_data.get("choices")
        if not choices:
            logger.warning(
                "LLM response has no choices. Full response: %s", response_data
            )
            return None

        first_choice = choices[0]
        message = first_choice.get("message")
        if not message:
            logger.warning(
                "LLM response choice has no message. Full response: %s", response_data
            )
            return None

        content: str | None = message.get("content")

        # Guard against null content from the LLM
        if content is None:
            logger.warning("LLM returned null content, returning raw response")
            return content

        if config.response_model is None:
            return content

        try:
            parsed: ResponseModel = t.cast(
                ResponseModel, config.response_model.model_validate_json(content)
            )

            return parsed
        except ValidationError as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                f"Failed to parse LLM response with {config.response_model.__name__} "
                f"(took {elapsed_ms:.0f}ms): {exc}"
            )
            raise ValueError(
                f"Failed to parse LLM response with {config.response_model.__name__}"
            ) from exc
    finally:
        if close_after:
            await client.aclose()
