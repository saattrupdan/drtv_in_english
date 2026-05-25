"""Watch DR videos with English subtitles."""

from .llm import build_client, correct_and_translate
from .logging_config import configure_logging
from .resolver import resolve
from .vtt import parse_external_vtt, parse_vtt_text, write_vtt_file
