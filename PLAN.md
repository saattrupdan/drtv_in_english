# Phase 3 Plan: Transcribing Audio

## Summary

Phase 3 adds speech-to-text transcription to the pipeline. We create a `transcribing` module that uses Hugging Face's `transformers` library with the `CoRal-project/roest-v3-wav2vec2-315m` model (a Danish ASR model) to transcribe audio chunks into text segments.

## What Phase 3 entails

- A new `Transcription` Pydantic model with `start_time`, `end_time`, and `text` fields
- A `transcribe` function that takes a numpy audio array and an `AutomaticSpeechRecognitionPipeline` object, and returns a list of `Transcription` models
- The `transcribe` function tracks global time offsets so that `start_time` and `end_time` refer to the full audio timeline, not just the individual chunk
- A new `transformers` and `torch` dependency for the ASR pipeline
- Tests for the transcribing module

## Implementation Steps

- [x] Add `transformers` and `torch` dependencies to `pyproject.toml`
- [ ] Create `src/but_with_subs/transcribing.py` with `Transcription` model and `transcribe` function
- [ ] Create `tests/test_transcribing.py` with unit tests for the Transcription model and transcribe function
- [ ] Run tests and verify they pass with `make test`
- [ ] Run linting and formatting checks with `make check`
