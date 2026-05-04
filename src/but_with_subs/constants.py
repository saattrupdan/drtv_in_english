"""Constants used in the project."""

# We ignore chunks that have this length
MIN_CHUNK_LENGTH_SECONDS = 0.05

# We ensure that subtitles for chunks are at least this long
MIN_CHUNK_DISPLAY_LENGTH_SECONDS = 0.5

# Colorblind-friendly palette with maximum contrast (WCAG AA compliant)
# Hex colors optimized for distinguishing overlapping speakers
OVERLAPPING_SPEAKER_COLORS = [
    "#E69F00",  # Safety orange - highly visible
    "#56B4E9",  # Sky blue - distinct from orange
    "#009E73",  # Bluish green - colorblind safe
    "#CC79A7",  # Reddish purple - high contrast
]

DEFAULT_TRANSLATION_MODEL = "alirezamsh/small100"
DEFAULT_TARGET_LANGUAGE = "en"
DEFAULT_BATCH_SIZE = 16

# ASR model for transcription
ASR_MODEL_ID = "CoRal-project/roest-v3-wav2vec2-315m"

TARGET_SAMPLE_RATE = 16_000

# Language codes (ISO 639-1, 2-letter format)
DA = "da"

# Max words per text segment for chunking
MAX_WORDS = 12

# Maximum total audio duration (seconds) per batch for transcription
MAX_DURATION = 60.0

# Audio validation duration bounds
MIN_DURATION_SECONDS = 0.1
MAX_DURATION_SECONDS = 3600.0

# Default data directory
DATA_DIR = "./data"
