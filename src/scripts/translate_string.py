import click
import os

from but_with_subs.llm import LLMConfig
from but_with_subs.translation import translate


@click.command()
@click.option("--text", required=True, help="Text to translate")
@click.option("--language", required=True, help="Target language to translate to")
@click.option("--api-base", default=None, help="Override default API base")
def main(text: str, language: str, api_base: str | None) -> None:
    """Translate a string to a target language using an LLM."""
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]

    config = LLMConfig(
        model=model,
        temperature=0.0,
        max_tokens=1000,
        api_base=api_base or os.environ.get("LLM_API_BASE", "http://localhost:8080"),
        api_key=api_key,
    )

    result = translate(text, language, config)
    print(result)


if __name__ == "__main__":
    main()
