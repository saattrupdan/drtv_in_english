# Phase 2: Chunking Audio - Implementation Plan

## Overview

Phase 2 implements audio chunking functionality that splits audio
files into chunks based on natural breaks (silence detection). This
builds on Phase 1's downloading and audio extraction modules.

## Repository Structure

The repository is organized as follows:

```text
.
├── src/
│   ├── but_with_subs/          # Code modules (imported)
│   │   ├── __init__.py         # Package init, exports download, configure_logging
│   │   ├── api.py              # FastAPI app
│   │   ├── audio_extraction.py # Phase 1: extract audio via ffmpeg
│   │   ├── downloading.py      # Phase 1: download video/audio via yt-dlp
│   │   └── logging_config.py   # Central logging configuration
│   ├── scripts/                # Executable scripts (uv run)
│   │   ├── download_video.py   # Script to download from URL
│   │   ├── extract_audio.py    # Script to extract audio from video
│   │   └── fix_dot_env_file.py
│   └── frontend/               # Vue.js frontend
├── tests/                      # Test files
│   ├── test_audio_extraction.py
│   └── test_downloading.py
├── pyproject.toml              # Dependency management
├── makefile                    # Build/test commands
└── data/                       # Downloaded files directory
```

## Dependencies

Phase 2 requires audio processing libraries. The following packages need to be added:

- `numpy` - For audio data manipulation (arrays, shape operations)
- `scipy` - For signal processing (silence detection, resampling)

These will be added via `uv add numpy scipy`.

## Implementation Components

### 1. Chunk Model (Pydantic)

A new `Chunk` Pydantic model will be created in the `chunking` module. It will have:

- `start_time: float` - Start time in seconds from beginning of audio
- `end_time: float` - End time in seconds from beginning of audio
- `audio: numpy.ndarray` - Mono audio data at 16kHz

### 2. Chunking Module (`src/but_with_subs/chunking.py`)

The module will contain:

**`chunk_audio` function:**

- Takes `audio_path: Path` argument
- Yields `Chunk` models
- Uses silence detection to find natural breaks
- Resamples audio to 16kHz mono
- Returns numpy arrays of shape `(audio_length,)`

**Helper functions (low-level, ordered after high-level):**

- `_detect_silence_breaks` - Finds silence thresholds and gap locations
- `_load_and_resample_audio` - Loads WAV file and converts to 16kHz mono
- `_split_audio_into_chunks` - Splits audio into chunks based on break points

### 3. Logging

All logging will use the existing `logger` from
`logging_config.py` (configured via
`logging.getLogger(__package__)`).

### 4. Script for Testing

A new script `src/scripts/chunk_audio.py` will be created to test
the chunking function on sample files.

## Implementation Steps

### Step 1: Add audio processing dependencies

- [x] Add `numpy` and `scipy` to the project using `uv add`. These
are the core audio processing libraries needed for silence detection,
resampling, and array manipulation.

### Step 2: Create the Chunk model

- [x] Create a `Chunk` Pydantic model in `src/but_with_subs/chunking.py` with:
  - `start_time: float` - Start time in seconds
  - `end_time: float` - End time in seconds
  - `audio: numpy.ndarray` - Mono audio data at 16kHz

### Step 3: Implement the chunking module

- [x] Create `src/but_with_subs/chunking.py` with:
  - `chunk_audio(audio_path: Path) -> Generator[Chunk, None, None]`
    - Main function that yields Chunk models
  - `_load_audio(path: Path) -> tuple[int, numpy.ndarray]`
    - Loads audio file using scipy.io.wavfile
  - `_resample_to_16k_mono(audio: numpy.ndarray, original_sr: int) -> numpy.ndarray`
    - Resamples to 16kHz mono using scipy
  - `_detect_silence_breaks(
    audio: numpy.ndarray,
    sr: int,
    threshold_db: float,
    min_gap_seconds: float,
  ) -> list[float]`
    - Finds silence gaps using scipy signal processing
  - `_split_audio_into_chunks(
    audio: numpy.ndarray,
    break_times: list[float],
  ) -> tuple[list[float], list[float], list[numpy.ndarray]]`
    - Splits audio into chunks based on break points

All functions should be ordered from highest-level to lowest-level.
All imports should use relative imports. All function calls should use
keyword arguments.

### Step 4: Create test script

- [x] Create `src/scripts/chunk_audio.py` that:
  - Accepts an audio file path as a CLI argument via click
  - Calls `chunk_audio` and prints the number of chunks and their durations
  - Uses the same logger as the rest of the project

### Step 5: Create tests

- [x] Create `tests/test_chunking.py` with tests for:
  - Chunk model creation
  - `chunk_audio` function with mocked audio data
  - Silence detection logic
  - Audio resampling logic
  - Edge cases (short audio, no silence, continuous silence)

### Step 6: Run checks and tests

- Run `make check` to ensure formatting, linting, and type checking passes
- Run `make test` to ensure all tests pass
- Update any relevant documentation
