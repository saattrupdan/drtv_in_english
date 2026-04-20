# Plan: Add Transcription Testing Script

## Context

The project has three existing test scripts in `src/scripts/`:
- `download_video.py` - tests the download function
- `extract_audio.py` - tests audio extraction from video
- `chunk_audio.py` - tests audio chunking

All follow the same pattern:
- Docstring with description and usage instructions
- `click` for CLI argument parsing
- `logger = logging.getLogger(__package__)` for logging
- `main()` function with `@click.command()` decorator
- Input validation (file existence checks)
- Call the relevant function from the `but_with_subs` package

The user wants a `transcribe_audio.py` script that:
- Takes a WAV audio file path as input
- Chunks the audio using `chunk_audio` from `but_with_subs.chunking`
- Creates a Wav2Vec2 ASR pipeline via `transformers`
- Transcribes each chunk using `transcribe` from `but_with_subs.transcribing`
- Logs the transcriptions to the terminal (no disk output)

## Implementation Steps

- [x] Create `src/scripts/transcribe_audio.py` with click CLI, Wav2Vec2 pipeline setup, chunking loop, and transcription output logging
- [ ] Run the script against a sample audio file to verify it works end-to-end
- [ ] Run `make check` to ensure linting and formatting pass
- [ ] Run `make test` to ensure existing tests still pass
