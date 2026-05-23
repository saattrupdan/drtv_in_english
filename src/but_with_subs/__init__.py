"""Watch anything online... but with subs."""

from .audio_chunking import diarize, load_diarization_pipeline
from .downloading import download
from .llm import correct_and_translate
from .logging_config import configure_logging
from .subtitling import generate_subtitles
