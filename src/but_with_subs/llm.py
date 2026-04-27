"""LLM inference module for querying local and remote LLM APIs.

This module provides a generic interface for querying LLM APIs, with a focus on
compatibility with llama.cpp servers. It supports arbitrary response models defined
as Pydantic BaseModels.
"""

import time
import typing as t

from httpx import AsyncClient, Response
from pydantic import BaseModel, ValidationError

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

        response: Response = await client.post(
            url=url, json=payload, headers=headers, timeout=600
        )

        if response.is_error:
            (time.monotonic() - start_time) * 1000
            logger.error(f"LLM API error {response.status_code}: {response.text}")
            response.raise_for_status()

        response_data: ChatCompletionResponse = response.json()
        (time.monotonic() - start_time) * 1000

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
            (time.monotonic() - start_time) * 1000
            logger.error(
                f"Failed to parse LLM response with {config.response_model.__name__}: "
                f"{exc}"
            )
            raise ValueError(
                f"Failed to parse LLM response with {config.response_model.__name__}"
            ) from exc
    finally:
        if close_after:
            await client.aclose()
