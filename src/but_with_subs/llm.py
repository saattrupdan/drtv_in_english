"""LLM inference module for querying local and remote LLM APIs.

This module provides a generic interface for querying LLM APIs, with a focus on
compatibility with OpenAI-compatible APIs. It supports arbitrary response models defined
as Pydantic BaseModels.
"""

import time
import typing as t
from dataclasses import dataclass
from enum import Enum

from httpx import AsyncClient, Response
from pydantic import BaseModel, ValidationError

from .logging_config import logger
from .types import ChatCompletionRequest, ChatCompletionResponse, InputMessage


class LLMServerType(str, Enum):
    """Detected type of LLM backend server."""

    OPENAI = "openai"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    GROQ = "groq"
    TOGETHER = "together"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    LLAMA_CPP = "llama_cpp"
    UNKNOWN = "unknown"


@dataclass
class ServerCapabilities:
    """Capabilities of a detected LLM server."""

    server_type: LLMServerType
    supports_json_schema: bool


# Cache detected server capabilities per api_base to avoid repeated probing
_server_capabilities_cache: dict[str, ServerCapabilities] = {}


def _is_known_groq_model(model_id: str) -> bool:
    """Check if a model ID looks like a known Groq model."""
    groq_models = {
        "mixtral",
        "llama3",
        "gemma",
        "llama-3",
        "llama3-70b",
        "llama3-8b",
        "mixtral-8x7b",
        "gemma-7b",
        "gemma2-9b",
    }
    mid = model_id.lower()
    return any(gm in mid for gm in groq_models)


async def _detect_server_capabilities(
    api_base: str, client: AsyncClient
) -> ServerCapabilities:
    """Detect server type and capabilities by probing endpoints.

    Step 1: Probe /v1/models (OpenAI-compatible servers).
    Step 2: Probe /models (llama.cpp fallback).
    Step 3: Determine json_schema support based on server type.

    Args:
        api_base: The base URL of the LLM API.
        client: An httpx AsyncClient to use for the request.

    Returns:
        ServerCapabilities with the detected server type and json_schema support.
    """
    # Step 1: Probe /v1/models
    try:
        resp = await client.get(f"{api_base}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                # OpenAI-compatible endpoint found. Identify the provider.
                headers_lower = {k.lower(): v for k, v in resp.headers.items()}
                server_header = headers_lower.get("server", "").lower()
                model_ids = []
                for item in data.get("data", []):
                    if isinstance(item, dict):
                        mid = item.get("id", "")
                        if mid:
                            model_ids.append(mid)
                first_model_id = model_ids[0] if model_ids else ""

                # Prioritize HTTP response headers for identification
                if "openai" in server_header:
                    server_type = LLMServerType.OPENAI
                elif "ollama" in server_header:
                    server_type = LLMServerType.OLLAMA
                elif "lm-studio" in server_header:
                    server_type = LLMServerType.LM_STUDIO
                elif "groq" in server_header:
                    server_type = LLMServerType.GROQ
                elif "together" in server_header:
                    server_type = LLMServerType.TOGETHER
                elif "mistral" in server_header:
                    server_type = LLMServerType.MISTRAL
                elif "vllm" in server_header:
                    server_type = LLMServerType.VLLM
                elif "gemini" in server_header or first_model_id.startswith("gemini-"):
                    server_type = LLMServerType.GEMINI
                elif _is_known_groq_model(first_model_id):
                    server_type = LLMServerType.GROQ
                else:
                    server_type = LLMServerType.OPENAI

                logger.debug(
                    "Detected %s server at %s", server_type.value, api_base
                )
                return ServerCapabilities(
                    server_type=server_type,
                    supports_json_schema=_json_schema_supported(server_type),
                )
    except Exception:
        pass

    # Step 2: Fall back to llama.cpp endpoint
    try:
        resp = await client.get(f"{api_base}/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "models" in data:
                logger.debug("Detected llama.cpp server at %s", api_base)
                return ServerCapabilities(
                    server_type=LLMServerType.LLAMA_CPP,
                    supports_json_schema=True,
                )
    except Exception:
        pass

    logger.warning(
        "Could not detect LLM server type at %s, defaulting to UNKNOWN", api_base
    )
    return ServerCapabilities(
        server_type=LLMServerType.UNKNOWN,
        supports_json_schema=False,
    )


def _json_schema_supported(server_type: LLMServerType) -> bool:
    """Return whether a server type supports json_schema response_format."""
    supported = {
        LLMServerType.OPENAI,
        LLMServerType.VLLM,
        LLMServerType.LM_STUDIO,
        LLMServerType.GROQ,
        LLMServerType.TOGETHER,
        LLMServerType.MISTRAL,
        LLMServerType.GEMINI,
        LLMServerType.LLAMA_CPP,
    }
    return server_type in supported


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
            The type of LLM backend server. Auto-detected on first query
            if not explicitly set. Can be manually set to override autodetection.
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

        # Auto-detect server capabilities if not already known
        if config.server_type is None:
            cached = _server_capabilities_cache.get(config.api_base)
            if cached is not None:
                caps = cached
            else:
                detect_client = AsyncClient()
                try:
                    caps = await _detect_server_capabilities(
                        config.api_base, detect_client
                    )
                    _server_capabilities_cache[config.api_base] = caps
                finally:
                    await detect_client.aclose()
        else:
            caps = ServerCapabilities(
                server_type=config.server_type,
                supports_json_schema=True,  # Assume true if user set it manually
            )

        logger.info(
            "Using LLM server: %s at %s",
            caps.server_type.value,
            config.api_base,
        )

        # Warn if structured output may not work
        if not caps.supports_json_schema and config.response_model is not None:
            logger.warning(
                "Server %s (%s) may not support json_schema response_format. "
                "Structured output may not work as expected.",
                config.api_base, caps.server_type.value,
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
