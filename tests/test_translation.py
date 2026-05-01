"""Tests for the translation module.

This module contains tests for the ``translate`` function, covering
extraction of translated text from a ``TranslatedText`` Pydantic model,
direct string returns from ``query_llm``, and various target languages.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from but_with_subs import translation
from but_with_subs.data_models import LLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_config(
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 64,
    api_base: str = "http://localhost:8000",
) -> LLMConfig:
    """Create a minimal LLMConfig for testing.

    Args:
        model: Model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        api_base: API base URL.

    Returns:
        An LLMConfig instance.
    """
    return LLMConfig(
        model=model, temperature=temperature, max_tokens=max_tokens, api_base=api_base
    )


# ---------------------------------------------------------------------------
# TranslatedText model tests
# ---------------------------------------------------------------------------


def test_translated_text_model_creation() -> None:
    """Test that TranslatedText stores and returns text correctly."""
    model = translation.TranslatedText(text="Bonjour le monde")

    assert model.text == "Bonjour le monde"


def test_translated_text_default_text_is_empty() -> None:
    """Test that TranslatedText defaults text to an empty string."""
    model = translation.TranslatedText()

    assert model.text == ""


# ---------------------------------------------------------------------------
# translate() with TranslatedText response
# ---------------------------------------------------------------------------


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_extracts_text_from_model(mock_query_llm: AsyncMock) -> None:
    """Test that translate extracts text when query_llm returns a TranslatedText.

    Verifies that translating English text to French correctly returns the
    ``text`` field of the ``TranslatedText`` model returned by ``query_llm``.
    """
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Bonjour le monde")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="French", llm_config=config
        )
    )

    assert result == "Bonjour le monde"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_calls_query_llm_with_correct_prompt(
    mock_query_llm: AsyncMock,
) -> None:
    """Test that translate builds the correct prompt for query_llm.

    Verifies that the prompt includes the target language and the source text.
    """
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Hola mundo")

    asyncio.run(
        translation.translate(
            text="Hello world", target_language="Spanish", llm_config=config
        )
    )

    call_args = mock_query_llm.call_args
    prompt = call_args[0][0]  # First positional arg is prompt

    assert "Spanish" in prompt
    assert "Hello world" in prompt
    assert "Translate the following text to Spanish" in prompt


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_passes_response_model_to_config(mock_query_llm: AsyncMock) -> None:
    """Test that translate sets response_model on the config passed to query_llm.

    Verifies that the TranslatedText model is propagated via the config.
    """
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Ciao mondo")

    asyncio.run(
        translation.translate(
            text="Hello world", target_language="Italian", llm_config=config
        )
    )

    call_args = mock_query_llm.call_args
    passed_config: LLMConfig = call_args[0][1]  # Second positional arg is config

    assert passed_config.response_model is translation.TranslatedText


# ---------------------------------------------------------------------------
# translate() with raw string response
# ---------------------------------------------------------------------------


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_returns_string_directly(mock_query_llm: AsyncMock) -> None:
    """Test that translate returns the raw string when query_llm does not.

    If ``query_llm`` returns a plain string (not a ``TranslatedText``),
    ``translate`` should return it via ``str()``.
    """
    config = _make_llm_config()
    mock_query_llm.return_value = "Hola mundo"

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="Spanish", llm_config=config
        )
    )

    assert result == "Hola mundo"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_stringifies_non_string_response(mock_query_llm: AsyncMock) -> None:
    """Test that translate converts non-string responses via str().

    If ``query_llm`` returns some object that is not ``TranslatedText``,
    ``translate`` should still call ``str()`` on it.
    """
    config = _make_llm_config()

    class CustomResult:
        def __str__(self) -> str:
            return "custom output"

    mock_query_llm.return_value = CustomResult()

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="German", llm_config=config
        )
    )

    assert result == "custom output"


# ---------------------------------------------------------------------------
# Various target languages
# ---------------------------------------------------------------------------


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_spanish(mock_query_llm: AsyncMock) -> None:
    """Test translation to Spanish."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Hola mundo")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="Spanish", llm_config=config
        )
    )

    assert result == "Hola mundo"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_french(mock_query_llm: AsyncMock) -> None:
    """Test translation to French."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Bonjour le monde")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="French", llm_config=config
        )
    )

    assert result == "Bonjour le monde"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_german(mock_query_llm: AsyncMock) -> None:
    """Test translation to German."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="Hallo Welt")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="German", llm_config=config
        )
    )

    assert result == "Hallo Welt"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_japanese(mock_query_llm: AsyncMock) -> None:
    """Test translation to Japanese."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="こんにちは世界")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="Japanese", llm_config=config
        )
    )

    assert result == "こんにちは世界"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_mandarin(mock_query_llm: AsyncMock) -> None:
    """Test translation to Mandarin."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="你好世界")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="Mandarin", llm_config=config
        )
    )

    assert result == "你好世界"


@patch("but_with_subs.translation.query_llm", new_callable=AsyncMock)
def test_translate_arabic(mock_query_llm: AsyncMock) -> None:
    """Test translation to Arabic."""
    config = _make_llm_config()
    mock_query_llm.return_value = translation.TranslatedText(text="مرحبا بالعالم")

    result = asyncio.run(
        translation.translate(
            text="Hello world", target_language="Arabic", llm_config=config
        )
    )

    assert result == "مرحبا بالعالم"
