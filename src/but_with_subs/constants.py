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

TRANSLATION_MODEL = "alirezamsh/small100"

ASR_MODEL_ID = "CoRal-project/roest-v3-wav2vec2-315m"
TARGET_SAMPLE_RATE = 16_000

# Max words per text segment for chunking
MAX_WORDS = 12

# Default data directory
DATA_DIR = "./data"

# Language codes supported by the translation model (ISO 639-1, 2-letter format)
FAIRSEQ_LANGUAGE_CODES = [
    "af",
    "am",
    "ar",
    "ast",
    "az",
    "ba",
    "be",
    "bn",
    "br",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "ilo",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lb",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
]
