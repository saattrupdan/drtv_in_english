# Phase 1 Implementation Plan: Downloading Video and Audio

## Repository Structure

```
.
├── src/
│   ├── but_with_subs/          # Core Python modules (imported)
│   │   ├── __init__.py         # Package init + logging config import
│   │   ├── api.py              # FastAPI application
│   │   ├── downloading.py      # NEW: download function + File model
│   │   └── logging_config.py   # NEW: central logging configuration
│   ├── scripts/                # Executable scripts (run with uv run)
│   │   ├── fix_dot_env_file.py
│   │   └── download_video.py   # NEW: script to test downloading
│   └── frontend/               # Vue.js frontend (not in scope)
├── tests/
│   ├── __init__.py
│   └── test_downloading.py     # NEW: tests for downloading module
├── data/
│   └── .gitkeep                # Existing placeholder
├── pyproject.toml              # Dependencies (yt-dlp to be added)
├── makefile                    # make check, make test
└── .gitignore                  # data/* to be added
```

## Detailed Design

### 1. Dependencies

- Add `yt-dlp` via `uv add yt-dlp` (for downloading video/audio from URLs)
- `pydantic` is already available as a transitive dependency of `fastapi[standard]`

### 2. Logging Configuration (`src/but_with_subs/logging_config.py`)

- Create a `configure_logging()` function that sets up logging for the `but_with_subs` package
- Use `logger = logging.getLogger(__package__)` pattern throughout
- Configure handlers and formatters centrally so all modules share the same logging setup

### 3. `__init__.py` Update

- Import `configure_logging` from `logging_config` at package level
- Call `configure_logging()` during package initialisation so logging is set up before any module uses it

### 4. Downloading Module (`src/but_with_subs/downloading.py`)

#### `File` Pydantic Model

```python
class File(BaseModel):
    url: str
    video_path: Path | None
    audio_path: Path | None
```

- `url`: The original URL that was downloaded
- `video_path`: Path to the downloaded video file (first one if multiple), or `None`
- `audio_path`: Path to the downloaded audio file (first one if multiple), or `None`

#### `DownloadProgress` Pydantic Model

```python
class DownloadProgress(BaseModel):
    percentage: float
    status: str
    current_file: str | None = None
```

- `percentage`: Download progress as a float from 0.0 to 100.0
- `status`: Human-readable status string (e.g., "downloading", "finished", "error")
- `current_file`: Name of the file currently being downloaded, or `None`

#### `download` Function

```python
def download(url: str) -> c.Generator[DownloadProgress, None, File]:
    """Download video and audio from a URL using yt-dlp.

    Args:
        url:
            The URL to download from.

    Yields:
        Progress updates as the download progresses.

    Returns:
        A File model with the URLs and paths of the downloaded files.
    """
```

Implementation approach:

- Use `yt_dlp.YoutubeDL` with custom options
- Set `outtmpl` to `./data/%(title)s.%(ext)s` to store files in `./data/`
- Use `format` option to select best video and best audio formats separately
- Set up progress hooks to monitor yt-dlp output and yield `DownloadProgress`
- After download, return a `File` model with the paths
- If multiple video/audio files are produced, pick the first one

The function will:

1. Ensure the `./data/` directory exists
2. Configure yt-dlp options for video and audio extraction
3. Set up progress hooks that map yt-dlp progress events to `DownloadProgress`
4. Call yt-dlp's `download()` method
5. Yield progress updates during download
6. Return a `File` model with the resulting paths

### 5. Test Script (`src/scripts/download_video.py`)

- A script that demonstrates the `download` function in action
- Uses one of the sample URLs from the requirements
- Iterates over the generator to display progress
- Prints the final `File` model
- Uses the package logger (never `print`)

### 6. Tests (`tests/test_downloading.py`)

- Tests for the `File` Pydantic model (construction, validation)
- Tests for the `DownloadProgress` Pydantic model
- Tests for the `download` function using `unittest.mock` to mock yt-dlp
- Tests cover: successful download, multiple files (first one picked), progress yielding

### 7. `.gitignore` Update

- Add `data/*` to `.gitignore` (keeping `data/.gitkeep` via negation if needed)

## Todo List

- [x] Add `yt-dlp` dependency using `uv add yt-dlp`
- [x] Create `src/but_with_subs/logging_config.py` with central logging setup
- [x] Update `src/but_with_subs/__init__.py` to import and call `configure_logging()`
- [x] Create `src/but_with_subs/downloading.py` with `File`, `DownloadProgress`, and `download`
- [x] Create `src/scripts/download_video.py` test script
- [x] Create `tests/test_downloading.py` with comprehensive tests
- [ ] Add `data/*` to `.gitignore` (preserving `.gitkeep`)
- [ ] Run `make check` (formatters, linters, type checkers)
- [ ] Run `make test` (pytest)
- [ ] Update documentation if necessary
