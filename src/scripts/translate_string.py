"""Translate a text string using an LLM."""

import asyncio

import click

from but_with_subs.data_models import LLMConfig
from but_with_subs.translation import translate


@click.command()
@click.option("--text", required=True, help="Text to translate")
@click.option("--language", required=True, help="Target language to translate to")
@click.option("--api-base", default="http://localhost:8080", help="API base URL")
@click.option("--api-key", required=True, help="LLM API key")
@click.option("--llm-model", required=True, help="LLM model name")
def main(text: str, language: str, api_base: str, api_key: str, llm_model: str) -> None:
    """Translate a string to a target language using an LLM.

    Args:
        text:
            Text to translate.
        language:
            Target language to translate to.
        api_base:
            API base URL. Defaults to ``http://localhost:8080``.
        api_key:
            LLM API key.
        llm_model:
            LLM model name.
    """
    config = LLMConfig(
        model=llm_model,
        temperature=0.0,
        max_tokens=1000,
        api_base=api_base,
        api_key=api_key,
    )

    result = asyncio.run(translate(text, language, config))
    print(result)


if __name__ == "__main__":
    main()
