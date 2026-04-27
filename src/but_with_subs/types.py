"""TypedDict definitions for LLM API request and response payloads.

This module provides strongly-typed definitions for chat completion requests
and responses, replacing the previous use of `dict[str, t.Any]` with proper
TypedDict classes for better type safety and IDE support.
"""

import typing as t


class InputMessage(t.TypedDict):
    """A single message to input to an LLM.

    Attributes:
        content:
            The content of the message.
        role:
            The role of the message sender (e.g. "user", "system").
    """

    content: str
    role: t.Literal["user", "system"]


class OutputMessage(t.TypedDict):
    """A response message from an LLM.

    Attributes:
        content:
            The content of the message.
        role:
            The role of the message sender (e.g. "assistant").
    """

    content: str | None
    role: t.Literal["assistant"]


class Choice(t.TypedDict):
    """A single choice in a chat completion response.

    Attributes:
        message:
            The message contained in this choice.
    """

    message: OutputMessage


class ChatCompletionResponse(t.TypedDict):
    """A full chat completion response from an LLM API.

    Attributes:
        choices:
            The list of choices returned by the API.
    """

    choices: list[Choice]


class JsonSchema(t.TypedDict):
    """A JSON schema for a response format.

    Attributes:
        name:
            The name of the schema.
        schema:
            The JSON schema for the response format.
    """

    name: str
    schema: str


class ResponseFormat(t.TypedDict):
    """The format in which the LLM should respond.

    Attributes:
        type:
            The type of response format (e.g. "json").
        schema:
            The schema for the response format.
    """

    type: str
    json_schema: JsonSchema


class ChatCompletionRequest(t.TypedDict):
    """A chat completion request payload sent to an LLM API.

    Attributes:
        model:
            The name of the LLM model to use.
        messages:
            The list of messages in the conversation.
        temperature:
            The sampling temperature for the LLM.
        max_tokens:
            The maximum number of tokens to generate.
        response_format:
            The format in which the LLM should respond.
    """

    model: str
    messages: list[InputMessage]
    temperature: float
    max_tokens: int
    response_format: t.NotRequired[ResponseFormat]
