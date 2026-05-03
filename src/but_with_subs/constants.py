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
