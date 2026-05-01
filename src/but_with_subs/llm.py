"""LLM inference module for querying local and remote LLM APIs.

This module provides a generic interface for querying LLM APIs, with a focus on
compatibility with OpenAI-compatible APIs. It supports arbitrary response models defined
as Pydantic BaseModels.
"""

import asyncio
import typing as t
from dataclasses import dataclass

from httpx import AsyncClient, Response
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .data_models import LLMServerType, LLMConfig, QueryLLMBatchItem
from .logging_config import logger
from .types import ChatCompletionRequest, ChatCompletionResponse, InputMessage


@dataclass
class ServerCapabilities:
    """Capabilities of a detected LLM server."""

    server_type: LLMServerType


# Cache detected server capabilities per api_base to avoid repeated probing
_server_capabilities_cache: dict[str, LLMServerType] = {}


async def _detect_server_type(
    api_base: str, api_key: str | None, client: AsyncClient
) -> LLMServerType:
    """Detect server type by probing endpoints.

    Args:
        api_base:
            The base URL of the LLM API.
        api_key:
            The API key to use for the LLM API.
        client:
            An httpx AsyncClient to use for the request.

    Returns:
        LLMServerType indicating the detected server type.
    """
    resp = await client.get(
        f"{api_base}/models",
        timeout=5.0,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if resp.status_code == 200:
        payload = resp.json()
        if (
            isinstance(payload, dict)
            and "data" in payload
            and isinstance(payload["data"], list)
            and len(payload["data"]) > 0
            and "owned_by" in payload["data"][0]
        ):
            if payload["data"][0]["owned_by"] == "llamacpp":
                logger.info("Detected Llama.cpp server at %s", api_base)
                return LLMServerType.LLAMA_CPP
            else:
                logger.info("Detected OpenAI-compatible server at %s", api_base)
                return LLMServerType.OPENAI_COMPATIBLE

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
        # Auto-detect server type if not already known
        if config.server_type is None:
            cached = _server_capabilities_cache.get(config.api_base)
            if cached is not None:
                server_type = cached
            else:
                detect_client = AsyncClient()
                try:
                    server_type = await _detect_server_type(
                        api_base=config.api_base,
                        api_key=config.api_key,
                        client=detect_client,
                    )
                    _server_capabilities_cache[config.api_base] = server_type
                finally:
                    await detect_client.aclose()
        else:
            server_type = config.server_type

        if config.response_model is not None:
            match server_type:
                case LLMServerType.LLAMA_CPP:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "schema": config.response_model.model_json_schema(),
                    }
                case LLMServerType.OPENAI_COMPATIBLE:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": config.response_model.__name__,
                            "schema": config.response_model.model_json_schema(),
                        },
                    }
                case LLMServerType.UNKNOWN:
                    raise ValueError(
                        "Using `response_format` is not supported when the LLM server "
                        "type is unknown."
                    )

        response: Response = await client.post(
            url=url, json=payload, headers=headers, timeout=600
        )

        if response.is_error:
            logger.error(f"LLM API error {response.status_code}: {response.text}")
            response.raise_for_status()

        response_data: ChatCompletionResponse = response.json()

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
            logger.error(
                f"Failed to parse LLM response with {config.response_model.__name__}: {exc}"
            )
            raise ValueError(
                f"Failed to parse LLM response with {config.response_model.__name__}"
            ) from exc
    finally:
        if close_after:
            await client.aclose()


async def query_llm_batch(
    items: list[QueryLLMBatchItem],
    client: AsyncClient | None = None,
    desc: str | None = None,
) -> list[t.Any]:
    """Query an LLM API with multiple prompts concurrently.

    Sends all prompts to the LLM API in parallel and returns results
    in the same order as the input list.

    Args:
        items:
            A list of (prompt, config) tuples to send to the LLM.
        client (optional):
            An optional httpx AsyncClient to share across requests. If not provided,
            a new client will be created for each request.
        desc (optional):
            Description string for the progress bar.

    Returns:
        A list of responses in the same order as the input items.
    """

    async def _run(item: QueryLLMBatchItem) -> t.Any:
        return await query_llm(prompt=item.prompt, config=item.config, client=client)

    coroutines = [_run(item) for item in items]
    with tqdm(total=len(items), desc=desc, disable=len(items) == 0) as pbar:
        results = await asyncio.gather(*coroutines)
        pbar.update(len(items))

    return results
