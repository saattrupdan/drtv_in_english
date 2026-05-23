# PLAN.md

Replace the small100 translation model with an LLM-based "correct + translate" step,
and fix the diarization / chunking bugs that produced unusable timestamps and `(N/A)`
speakers in the Tingbjerg run.

This plan is scoped to the changes required. It is **not** a refactor — leave
untouched anything that isn't listed.

---

## 1. Motivation

Running the pipeline on
`https://www.dr.dk/drtv/se/tingbjerg_eksperimentet_-de-udvalgte_594476` revealed three
problems in the output VTT:

1. **Garbage cue timestamps.** ~250 of 530 cues had end-times spanning minutes
   (e.g. cue 42: `00:00:10.689 --> 00:36:37.129`). Dozens of cues all started at the
   same timestamp (`00:00:10.689`, the first `jeg` in the audio).
2. **Speaker labels are always `(N/A)`.** Diarization is never wired into the live
   pipeline — `transcribing.transcribe_audio` produces chunks with `speaker=None`
   and `audio_chunking.chunk_by_audio` (the only code that calls
   `speaker-diarization-community-1`) is dead code from the pipeline's perspective.
3. **small100 translation quality is poor.** Examples:
   - `"Otte kilometer fugleflugt fra Rådhuspladsen"` → `"Eight miles of birds escape
     from Councilhusplatz"` (km→miles, idiom literalised, proper noun re-spelled).
   - `"Tingbjerg"` → `"Tingberg"` (proper noun mangled).
   - small100 also has no access to surrounding context, so ASR-level errors (e.g.
     missing `af` in `"Et Danmarkshistoriens største..."`) are translated faithfully
     instead of being corrected.

We will replace small100 with an LLM call against an OpenAI-compatible API that does
**both** ASR correction and translation in one pass, using window-of-context, and
we will drop the dual `.da.vtt` + `.<lang>.vtt` output in favour of one VTT in the
target language.

---

## 2. Bug 1 — Cue timestamps in `text_chunking.group_word_chunks`

### Root cause

`text_chunking._find_matching_chunks(target, word_chunks)` performs a **global**
fuzzy match of `target` against every `Chunk` in `word_chunks`. For very common
words ("jeg", "det", "er", "her", ...), many word chunks throughout the entire
audio match with ratio 1.0; Python's stable sort then deterministically returns
whichever occurs first in `word_chunks`. As a result:

- Every segment whose first word is `"jeg"` is anchored to the **first** `jeg` in
  the audio (`00:00:10.689`), regardless of where it actually occurs.
- For the last word, the code filters to `end_time > start_time` then picks the
  first match — which can be tens of minutes later in the file.

The current code has no positional anchor at all.

### Fix

Walk the punctuated text and `word_chunks` in lockstep with a moving pointer.

In `src/but_with_subs/text_chunking.py`:

- Remove `_find_matching_chunks`, the `difflib`/`SequenceMatcher` import, and the
  `re`/`string` punctuation hacks tied to it. (Keep `PUNCTUATION_PATTERN` only if
  still used by `group_word_chunks`'s lowercase pass into PunctFixer.)
- Replace the global-match logic in `group_word_chunks` with a sequential aligner:

  ```python
  def group_word_chunks(word_chunks, punctuation_model, max_words):
      # 1. Build the cleaned word stream that PunctFixer sees, preserving the
      #    1:1 mapping cleaned_words[i] <-> word_chunks[i].
      cleaned_words: list[str] = []
      indexed_chunks: list[Chunk] = []
      for wc in word_chunks:
          if not wc.text:
              continue
          cleaned = re.sub(PUNCTUATION_PATTERN, "", wc.text.lower()).strip()
          if not cleaned:
              continue
          cleaned_words.append(cleaned)
          indexed_chunks.append(wc)

      punctuated = punctuation_model.punctuate(text=" ".join(cleaned_words))

      # 2. Split the punctuated text into readable segments.
      #    PunctFixer preserves word order and count for non-empty inputs, so
      #    iterating segments and consuming N cleaned words per segment is
      #    sufficient.
      result: list[Chunk] = []
      cursor = 0
      for segment in _split_text(text=punctuated, max_words=max_words):
          tokens = [
              re.sub(PUNCTUATION_PATTERN, "", t).strip().lower()
              for t in segment.split()
          ]
          tokens = [t for t in tokens if t]
          if not tokens:
              continue
          n = len(tokens)
          if cursor + n > len(indexed_chunks):
              logger.warning(
                  f"Word/segment alignment exhausted at cursor={cursor} for "
                  f"segment={segment!r}; skipping."
              )
              break
          window = indexed_chunks[cursor : cursor + n]
          cursor += n
          start = window[0].start_time
          end = window[-1].end_time
          if end - start < MIN_CHUNK_LENGTH_SECONDS:
              continue
          result.append(
              Chunk(
                  start_time=start,
                  end_time=end,
                  audio=np.concatenate([w.audio for w in window], axis=0),
                  text=segment,
                  speaker=window[0].speaker,
              )
          )
      return result
  ```

- If PunctFixer turns out to ever drop/insert tokens (audit by running on the
  Tingbjerg .wav and asserting `len(out.split()) == len(in.split())`), fall back
  to a token-by-token alignment using `difflib.SequenceMatcher` on the **cleaned
  sequences themselves** (not the global candidate list), which gives an
  order-preserving alignment. Encode this fallback only if the audit fails.

### Tests

- `tests/test_text_chunking.py`: add a case where the same word appears multiple
  times (e.g. `"jeg ... jeg ... jeg"`) and assert each output `Chunk.start_time`
  matches the **positional** occurrence in `word_chunks`, not the first.
- Add a regression case asserting that for `N` synthetic word_chunks with
  monotonically increasing timestamps, every produced segment satisfies
  `start_time < end_time` **and** `end_time <= next_segment.start_time` (or the
  next segment is missing).

---

## 3. Bug 2 — Diarization is never run

### Root cause

`pipeline.run_pipeline` calls `transcribe_audio` (VAD-only) and never invokes
`audio_chunking.chunk_by_audio` (the only path that runs
`speaker-diarization-community-1`). `Chunk.speaker` therefore stays `None`.

### Fix

Wire diarization into the canonical pipeline. Diarization output is a list of
`(start_s, end_s, speaker_label)` turns. We need to assign each word chunk a
speaker so the existing colour logic in `subtitling._detect_overlapping_speakers`
can do its job.

Concrete changes:

- **`src/but_with_subs/audio_chunking.py`**
  - Split the current `chunk_by_audio` into two functions:
    - `load_diarization_pipeline()` — returns the loaded `Pipeline` (so the
      FastAPI lifespan can load it once at startup, matching how ASR /
      PunctFixer are handled).
    - `diarize(audio, pipeline) -> list[tuple[float, float, str]]` — returns
      raw `(start, end, speaker)` turns. No `Chunk` construction.
  - Keep the existing `chunk_by_audio` only if a script depends on it (check
    `src/scripts/chunk_audio.py`); otherwise delete it.

- **`src/but_with_subs/transcribing.py`**
  - Add a helper `assign_speakers(word_chunks, turns)` that, for each word chunk,
    picks the speaker whose `(start, end)` turn has the largest temporal overlap
    with the word's `[start_time, end_time]`. Set `wc.speaker = label`. If no
    turn overlaps, leave `wc.speaker = None`.
  - Do **not** intertwine diarization with VAD segmentation — they remain
    independent passes over the same audio.

- **`src/but_with_subs/pipeline.py`**
  - Accept a `diarization_model` argument in `run_pipeline`.
  - After `transcribe_audio` returns word chunks, call
    `turns = diarize(audio, diarization_model)`, then
    `assign_speakers(word_chunks, turns)` **before** `group_word_chunks` (so the
    grouped segments inherit speaker labels from `window[0].speaker`).
  - Emit a `ProgressEvent(stage="transcribing", message="Identifying
    speakers...")` at the start of the diarization step; budget ~10 % of the
    transcribe range for it (e.g. shift `TRANSCRIBE_END` math so diarization
    occupies a clear sub-band, or just emit one update before and after).

- **`src/but_with_subs/api.py`**
  - In `lifespan`, load the diarization pipeline once via
    `load_diarization_pipeline()`; store on `AppState`.
  - Pass `state.diarization_model` into `run_pipeline`.

- **`src/scripts/run_pipeline.py`**
  - Load the diarization pipeline alongside the ASR model and pass it through.

### HF token

`speaker-diarization-community-1` requires accepting the licence + a
`HF_TOKEN`. `fix_dot_env_file.py` already prompts for it (commit `b8256cb`), so
no env wiring is needed beyond confirming `HF_TOKEN` is read in
`load_diarization_pipeline()` — pyannote picks it up automatically from the env.

### Tests

- Mock `Pipeline.from_pretrained` and the diarization output the way existing
  tests mock `vad_segment_audio` (see `AGENTS.md`, "Heavy model downloads"
  guidance).
- `tests/test_transcribing.py` (or new `tests/test_speaker_assignment.py`): given
  fake turns `[(0,5,"S1"), (5,10,"S2")]` and word chunks straddling the
  boundary, assert each chunk gets the speaker with the largest overlap.

---

## 4. Replace small100 with an LLM-based corrector + translator

### Behaviour

A new module `src/but_with_subs/llm.py` exposes:

```python
def correct_and_translate(
    chunks: list[Chunk],
    target_language: str,
    *,
    client: openai.OpenAI,
    model: str,
    context_window: int = 6,
    on_progress: Callable[[float], None] | None = None,
) -> list[Chunk]:
    """Rewrite each chunk's `text` so it is both ASR-corrected and translated
    into `target_language`.

    Operates over sliding windows of `context_window` chunks before and after
    the target chunk so the LLM can fix ASR errors using surrounding context
    (proper nouns, agreement, missing function words). Returns a new list of
    Chunk objects with the same timing/audio/speaker but updated `text`.
    """
```

Key design points:

- **Single LLM pass per chunk, with context.** Each request sends the LLM:
  - `target_language` (ISO-639-1, e.g. `"en"`).
  - The full window of `2 * context_window + 1` chunks as a numbered list, each
    with start/end timestamp.
  - An instruction to return the corrected + translated text for **only** the
    centre chunk, as a JSON object `{"text": "..."}`.
  - The system prompt explains: this is Danish TV speech transcribed by an ASR
    model; correct ASR errors (missing words, mangled proper nouns) using
    surrounding context; preserve speaker meaning; keep length similar so it
    fits as a subtitle; do not translate proper nouns; output only the
    requested JSON.
- **Batching for throughput.** Group adjacent chunks into a single request when
  feasible (e.g. ask the LLM to return text for chunks N..N+k together) — but
  only as a v2 optimisation. v1 should be the simple per-chunk loop, since
  correctness matters more than latency for a first cut. Note this in code as
  a TODO.
- **OpenAI-compatible.** Use the `openai` Python SDK
  (`openai>=1.0`) with a configurable base URL so any compatible endpoint
  (vLLM, llama.cpp, OpenRouter, Together, Groq, Mistral, Anthropic via a proxy,
  etc.) works. Read three env vars:
  - `LLM_BASE_URL` — required, e.g. `https://api.openai.com/v1` or
    `http://localhost:8000/v1`.
  - `LLM_API_KEY` — required; pass through verbatim.
  - `LLM_MODEL` — required, e.g. `gpt-4o-mini` or `qwen2.5-7b-instruct`.

  Build the client once in `llm.py:build_client()`; do not re-create per call.
- **Structured output.** Use the SDK's JSON mode
  (`response_format={"type": "json_object"}`) and parse with `json.loads` plus
  Pydantic validation. If the model returns malformed JSON or an empty `text`,
  fall back to the original (uncorrected, untranslated) Danish text for that
  chunk and emit a warning. Never let one bad chunk kill the whole run.
- **Progress reporting.** Call `on_progress(i / n)` after each chunk so the
  pipeline can yield `ProgressEvent`s. Keep the existing translate-stage
  percentage band (`TRANSCRIBE_END` is the current end of transcription; the
  LLM stage should sit between transcription and subtitling — see §5).
- **Retries.** Wrap the API call with exponential backoff (max 3 attempts) for
  transient errors (HTTP 429 / 5xx, network errors). The `openai` SDK supports
  automatic retries via `max_retries=` on the client; configure
  `max_retries=3` and rely on that rather than rolling our own retry loop.
- **No streaming.** Subtitle text is short; non-streaming JSON responses are
  simpler and avoid partial-parse failures.

### Files to delete

- `src/but_with_subs/translation.py` — entire file.
- `src/but_with_subs/tokenization_small100.py` — entire file.
- `tests/test_translation.py` — if present.
- `src/scripts/translate_string.py` — entire file; it's a thin wrapper around
  small100. (Replace with a thin CLI wrapper around `llm.py` only if needed;
  default = drop it.)

### Files to update

- **`src/but_with_subs/constants.py`**
  - Remove `TRANSLATION_MODEL` and `FAIRSEQ_LANGUAGE_CODES`. The LLM accepts
    any natural-language target spec, so a hard-coded list is unnecessary;
    validate at the API layer with a simple ISO-639-1 regex if we want to
    reject obvious garbage.
- **`src/but_with_subs/__init__.py`**
  - Drop any re-exports of `translate_*` and `SMALL100Tokenizer`.
  - Add `correct_and_translate` re-export.
- **`src/but_with_subs/api.py`**
  - Remove `M2M100ForConditionalGeneration` / `SMALL100Tokenizer` imports and
    fields on `AppState`. Remove the small100 load block in `lifespan`.
  - Remove the `set_tgt_lang_special_tokens` call in `process`.
  - Build the LLM client once in `lifespan` via `llm.build_client()` and store
    on `AppState` as `llm_client` + `llm_model`.
  - Forward `llm_client` and `llm_model` into `run_pipeline`.
- **`src/but_with_subs/pipeline.py`**
  - Replace `translation_model` / `translation_tokenizer` parameters with
    `llm_client` and `llm_model`.
  - Replace the `_translate_batch` block with a call to
    `correct_and_translate(chunks, target_language=language, client=...,
    model=..., on_progress=...)`.
  - The LLM stage always runs when `language` is set (which is now required —
    see §5).
- **`src/scripts/run_pipeline.py`**
  - Read `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` from env; fail fast with
    a clear error message if missing.
  - Pass the client + model name into the pipeline.
- **`src/scripts/transcribe_audio.py`**
  - Same env-driven LLM client construction; drop small100 imports.
- **`pyproject.toml`**
  - Add `openai>=1.50` to `dependencies`.
  - Optionally drop `sentencepiece` if no other module pulls it in (small100's
    tokenizer was its primary consumer). Verify with `uv pip tree` before
    removing.
- **`Dockerfile.backend` / `docker-compose.yaml`**
  - Drop any small100 model warm-up step from `model-cache-init`. Pass
    `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` into the backend container via
    `environment:` (referencing `.env`).
- **`fix_dot_env_file.py`**
  - Prompt for `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` alongside the
    existing `HF_TOKEN` prompt.
- **`AGENTS.md`**
  - Update the "Stack" and "End-to-end flow" sections: small100/M2M100/
    SMALL100Tokenizer references go away; add a line about the LLM
    correct-and-translate stage and the three env vars.

### Tests

- `tests/test_llm.py` — new. Mock the `openai.OpenAI` client (use
  `unittest.mock` or `respx`/`httpx` mock); assert:
  - Each chunk is sent with the configured context window before and after.
  - On malformed JSON, the original text is preserved and a warning is logged.
  - `on_progress` fires once per chunk with monotonically increasing ratios.
- `tests/test_pipeline.py` — update fakes so `run_pipeline` is called with a
  mocked LLM client; assert the final `Chunk.text` values came from the LLM.
- Delete tests asserting small100 behaviour.

---

## 5. Single-language output (drop `.da.vtt`)

### Behaviour change

The CLI `run_pipeline.py` currently always emits **two** VTTs (`.da.vtt` source
and `.<lang>.vtt` translation). The new behaviour:

- The user must specify a target language; the pipeline emits **one** VTT in
  that language, named `<basename>.<lang>.vtt`.
- If the user wants Danish subs (uncorrected), they can pass `--language da`;
  the LLM stage will then just correct ASR errors and re-emit Danish.

Concrete changes:

- **`src/scripts/run_pipeline.py`** — make `--language` a required option (no
  default). Remove any code that writes a separate Danish VTT.
- **`src/but_with_subs/pipeline.py`** — `language` becomes non-optional in the
  `run_pipeline` signature (`language: str`). Update `api.py` to validate that
  the request body includes `language`.
- **`src/but_with_subs/api.py`** — `ProcessRequest.language: str` (no default).
  Update the OpenAPI schema implicitly via the type change.
- **`src/but_with_subs/subtitling.py`** — `generate_subtitles` no longer needs
  to know about source-vs-translation; its current single-VTT output is fine.
  Just confirm `output_path` derivation includes the language suffix
  (`audio_path.with_suffix(f".{language}.vtt")`); thread `language` through if
  it isn't already.
- **`src/frontend/views/LandingPageView.vue`** — confirm the UI already
  requires a language selector. If today it has an optional language, make it
  required and update the `consumeStream` contract docs.

### Tests

- Update any test that asserted two output files; expect one.
- Update API tests to expect a 422 (validation error) when `language` is
  missing from `POST /process`.

---

## 6. Order of work

1. **Bug 1 (chunking timestamps)** — smallest blast radius, biggest visible
   win. Land + verify on the Tingbjerg .wav by re-running the pipeline with
   the existing small100 translator still in place. Assert no cue spans more
   than ~30 s.
2. **Bug 2 (diarization)** — adds a new dependency on `HF_TOKEN` actually
   working; verify on Tingbjerg that the resulting VTT has ≥2 distinct
   speakers and colour styling appears in the output.
3. **LLM module + remove small100** — biggest diff. Build with mocks first;
   run end-to-end against a real OpenAI-compatible endpoint last.
4. **Single-language output** — trivial once steps 1–3 are in; pure plumbing.

Each step is a separate commit (per the in-repo `commit-after-changes`
preference).

---

## 7. Verification on the Tingbjerg URL

After all four steps, re-run:

```
uv run python src/scripts/run_pipeline.py \
  "https://www.dr.dk/drtv/se/tingbjerg_eksperimentet_-de-udvalgte_594476" \
  --language en
```

Expected outcome:

- A single `Tingbjerg-eksperimentet... [00922217210].en.vtt` file.
- No cue with `end - start > 30 s` (sanity-check with a one-liner over the
  VTT after the run).
- Multiple distinct speaker labels present (not all `(N/A)`).
- Spot-checks:
  - `"Otte kilometer fugleflugt fra Rådhuspladsen"` translates to something
    using "as the crow flies" / "from City Hall Square", **not** "birds
    escape" / "Councilhusplatz".
  - `"Tingbjerg"` stays as `"Tingbjerg"` in the English subs.
  - `"Et Danmarkshistoriens største..."` is corrected to insert the missing
    `af` before translation.

---

## 8. Out of scope (explicitly not changing)

- ASR model (`CoRal Roest`) — keep as-is for now. The LLM corrector should
  patch over most word-level errors.
- PunctFixer — still used between ASR and segment-splitting. The LLM stage
  produces final cue text but PunctFixer's punctuated output drives where the
  cue boundaries fall, which we want to keep.
- Frontend rendering of overlapping speakers — already works; new diarization
  data will flow into it unchanged.
- Database schema — `FileRecord` still stores one `subtitles_path`; that
  matches the new single-VTT-per-run reality.
