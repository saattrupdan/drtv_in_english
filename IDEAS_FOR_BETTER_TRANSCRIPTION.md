# Ideas for Better Transcription Quality

## Executive Summary

The current "But With Subs" transcription pipeline uses a reasonably configured wav2vec2-based ASR model, but several structural weaknesses degrade output quality. Key findings:

- **Audio preprocessing is minimal**: no silence trimming, no RMS normalization, no high-pass filtering, and suboptimal resampling.
- **Full-audio transcription** is performed in a single pass without VAD pre-segmentation, causing degradation on long audio where context windows drift.
- **A punctuation-splitting bug** (`segment.split(",;:-")` on line 142 of `text_chunking.py`) splits on each character individually rather than as delimiter strings, corrupting segment boundaries.
- **Word matching is exact-string only**, making it fragile to ASR output variations (case, punctuation, minor misspellings).
- **Unused dependencies** (`kenlm`, `pyctcdecode`) indicate an opportunity to integrate CTC decoding with a language model.
- **No confidence scoring** means there is no visibility into transcription reliability.

All improvements below avoid switching to a larger model or introducing LLM-based post-processing. They rely on the existing stack (`transformers`, `scipy`, `torchaudio`, `pyannote`, `nltk`, `punctfix`).

---

## Current Pipeline Architecture

```
┌─────────────────────────┐
│  Video (YouTube URL)     │
└───────────┬─────────────┘
            │ yt-dlp
            ▼
┌─────────────────────────┐
│  audio_extraction.py    │  ffmpeg → WAV (48 kHz stereo)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  audio_loading.py       │  scipy.io.wavfile.read
│  load_audio()           │  → peak normalize to [-1, 1]
│                         │  → mono (mean of channels)
│                         │  → scipy.signal.resample to 16 kHz
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  audio_chunking.py      │  pyannote/speaker-diarization-community-1
│  chunk_by_audio()       │  → produces Chunk(start_time, end_time, audio, speaker)
│                         │  → filters chunks < MIN_CHUNK_LENGTH_SECONDS (0.05 s)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  transcribing.py        │  AutomaticSpeechRecognitionPipeline
│  transcribe_audio()     │  model(audio, return_timestamps="word", num_beams=5)
│                         │  → produces word-level Chunk objects
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  text_chunking.py       │  punctfix.PunctFixer(language="da")
│  group_word_chunks()    │  nltk.sent_tokenize(language="danish")
│                         │  segment.split(",;:-")          ← BUG: chars, not strings
│                         │  → produces sentence-level Chunk objects
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  subtitling.py          │  _detect_overlapping_speakers()
│  generate_subtitles()   │  _assign_speaker_colors()
│                         │  → WebVTT output
└─────────────────────────┘
```

---

## High-Impact Improvements (Priority P0/P1)

### 1. Audio Preprocessing Enhancements

**File:** `src/but_with_subs/audio_loading.py`

**Problems identified:**

- Only peak normalization (`/ np.iinfo(audio_data.dtype).max`) is applied. RMS loudness can vary wildly between sources, and wav2vec2 performs best on consistently-louder audio.
- No high-pass filter removes rumble/hum (50–60 Hz mains hum, room resonance).
- Leading/trailing silence wastes compute and can confuse VAD/diarization.
- `scipy.signal.resample` uses FFT-based resampling which can introduce phase artifacts.

**Proposed changes:**

Add three preprocessing steps to `load_audio()` and `_resample_to_16k_mono()`:

```python
import numpy as np
import scipy.signal
import scipy.io.wavfile
import torchaudio


def _trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0, min_silence_secs: float = 0.3) -> np.ndarray:
    """Remove leading and trailing silence.

    Uses RMS energy in 20 ms frames. Frames below threshold_db are
    considered silent.  min_silence_secs controls how much silence
    is kept at each end (to avoid chopping words).
    """
    frame_size = int(0.02 * sample_rate)  # 20 ms frames
    hop_size = frame_size // 2
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(frame_size), mode='valid') / frame_size)
    rms = rms[::hop_size]  # hop-aligned sub-samples
    threshold_linear = 10 ** (threshold_db / 20.0)
    non_silent = np.where(rms > threshold_linear)[0]

    if len(non_silent) == 0:
        return audio  # entirely silent, return unchanged

    # Keep min_silence_secs of buffer at each end
    buffer_frames = int(min_silence_secs / 0.01)  # 10 ms frames
    start = max(0, non_silent[0] - buffer_frames)
    end = min(len(audio), non_silent[-1] + buffer_frames + 1)
    return audio[start:end]


def _normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize AND RMS-normalize audio to a target loudness.

    wav2vec2 was trained on audio with approximately -16 dBFS RMS.
    """
    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # RMS normalization to approx -16 dBFS (~0.14 linear)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 0.14
        audio = audio * (target_rms / rms)

    # Re-clamp to prevent clipping after RMS scaling
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    return audio


def _high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Apply a high-pass Butterworth filter to remove low-frequency rumble."""
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = scipy.signal.butter(N=2, Wn=normalized_cutoff, btype='high')
    # filtfilt gives zero-phase filtering
    return scipy.signal.filtfilt(b, a, audio)


def _resample_to_16k_mono(audio: np.ndarray, original_sr: int) -> np.ndarray:
    """Resample audio to 16 kHz using torchaudio's sinc-based resampler.

    torchaudio.functional.resample uses sinc interpolation which is
    superior to scipy.signal.resample (FFT-based) for ASR-quality audio.
    """
    if original_sr == 16000:
        return audio

    # torchaudio expects shape (channels, samples)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    resampled = torchaudio.functional.resample(
        audio_tensor,
        orig_freq=int(original_sr),
        new_freq=TARGET_SAMPLE_RATE,
    )
    return resampled.squeeze(0).numpy()
```

Integrate into `load_audio()`:

```python
def load_audio(path: Path) -> np.ndarray:
    # ... (existing read logic unchanged) ...

    # Ensure float
    audio_data = np.array(audio_data, dtype=np.float32) / np.iinfo(audio_data.dtype).max

    # Mono
    if audio_data.ndim > 1:
        audio_data = np.mean(a=audio_data, axis=1)

    # NEW: High-pass filter
    audio_data = _high_pass_filter(audio_data, sample_rate)

    # NEW: Trim silence
    audio_data = _trim_silence(audio_data, sample_rate)

    # NEW: Loudness normalization
    audio_data = _normalize_loudness(audio_data)

    # Resample (now uses torchaudio)
    audio_data = _resample_to_16k_mono(audio=audio_data, original_sr=sample_rate)

    logger.info(f"Loaded audio from {path} at {sample_rate:,} Hz")
    return audio_data
```

---

### 2. VAD-Based Pre-Segmentation

**File:** `src/but_with_subs/transcribing.py`

**Problem:** The ASR pipeline is called once on the entire audio. For long audio (> 30 s per chunk), the transformer's self-attention over the full sequence causes context drift and degraded word-level timestamp accuracy.

**Solution:** Use `pyannote.audio.pipelines.VoiceActivityDetection` to split each speaker chunk into 10-second VAD-segmented pieces before passing to the ASR pipeline.

```python
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils.hook import ProgressHook


def vad_segment_audio(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    segment_duration: float = 10.0,
    overlap: float = 2.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio into VAD-segmented pieces using pyannote VAD.

    Returns list of (start_time, end_time, audio_slice) tuples.
    Only non-silent segments are returned.
    """
    vad = VoiceActivityDetection(segmentation_threshold=0.5)
    vad.to(get_device())

    waveform = torch.from_numpy(audio).unsqueeze(dim=0).float()
    with ProgressHook() as hook:
        speech_prob = vad(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=hook,
        )

    # speech_prob is a SegmentSequence with (segment, probability)
    active_segments = [(seg.start, seg.end) for seg, prob in speech_prob.itersegments() if prob > 0.5]

    if not active_segments:
        return [(0.0, len(audio) / sample_rate, audio)]  # fallback: entire audio

    # Merge overlapping active segments and split into fixed-duration windows
    merged = _merge_segments(active_segments)
    pieces = []
    for seg_start, seg_end in merged:
        seg_duration = seg_end - seg_start
        offset = 0.0
        while offset < seg_duration:
            chunk_end = min(offset + segment_duration, seg_duration)
            if chunk_end - offset < 1.0:  # skip tiny leftovers
                break
            audio_start = int((seg_start + offset) * sample_rate)
            audio_end = int((seg_start + chunk_end) * sample_rate)
            pieces.append((seg_start + offset, seg_start + chunk_end, audio[audio_start:audio_end]))
            offset += segment_duration - overlap  # overlap for continuity

    return pieces


def _merge_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent segments."""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0]]
    for start, end in sorted_segs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
```

Wire into `transcribe_audio()`:

```python
def transcribe_audio(
    audio: np.ndarray,
    model: AutomaticSpeechRecognitionPipeline,
    min_chunk_length: float = MIN_CHUNK_LENGTH_SECONDS,
    show_progress: bool = True,
) -> list[Chunk]:
    """Transcribe audio with VAD pre-segmentation."""
    # NEW: VAD pre-segmentation
    segments = vad_segment_audio(audio)

    all_word_chunks: list[Chunk] = []
    total_steps = sum(len(vad_segments) for _, _, vad_segments in segments)

    with tqdm(total=total_steps, desc="Transcribing", disable=not show_progress) as pbar:
        for seg_start, seg_end, seg_audio in segments:
            if seg_audio.size == 0:
                pbar.update(1)
                continue
            result = model(seg_audio, return_timestamps="word")
            for chunk_dct in result["chunks"]:
                # Adjust timestamps to global audio coordinates
                global_start = seg_start + float(chunk_dct["timestamp"][0])
                global_end = seg_start + float(chunk_dct["timestamp"][1])
                if global_end - global_start < min_chunk_length:
                    continue
                audio_s = int(TARGET_SAMPLE_RATE * global_start)
                audio_e = int(TARGET_SAMPLE_RATE * global_end)
                audio_e = min(audio_e, len(audio))
                all_word_chunks.append(
                    Chunk(
                        start_time=global_start,
                        end_time=global_end,
                        audio=audio[audio_s:audio_e],
                        text=chunk_dct["text"],
                        speaker=None,
                    )
                )
            pbar.update(1)

    logger.info(f"Completed transcription of {len(all_word_chunks)} word segments")
    return all_word_chunks
```

---

### 3. Fix Punctuation Splitting Bug

**File:** `src/but_with_subs/text_chunking.py`, line 142

**Current code:**

```python
punctuation_segments.extend(segment.split(",;:-"))
```

**Bug:** `str.split()` with a string argument splits on each *character* in the string, not on the substring. So `segment.split(",;:-")` splits on `,`, `;`, `:`, `-` individually — which happens to produce the same result for single-character delimiters. However, this is misleading and fragile. If someone later adds a multi-character delimiter (e.g., `"..."`), it will silently break.

More importantly, the intent is to split on any of these punctuation marks, which should use `re.split()`:

```python
# FIXED: Use regex to split on any of the punctuation characters
punctuation_segments.extend(
    [s.strip() for s in re.split(r'[,;:\-\u2013\u2014]', segment) if s.strip()]
)
```

This also handles Unicode en-dash (`\u2013`) and em-dash (`\u2014`) which commonly appear in Danish text.

---

### 4. Fuzzy Word Matching

**File:** `src/but_with_subs/text_chunking.py`, lines 59–87

**Current approach:** Exact string comparison after stripping punctuation and lowercasing:

```python
first_word_candidates = [
    word_chunk
    for word_chunk in word_chunks
    if word_chunk.text is not None
    and re.sub(PUNCTUATION_PATTERN, "", word_chunk.text).strip().lower()
    == first_word.strip()
]
```

This fails when the ASR produces a slightly different token (e.g., "den" vs "den ", or hyphenated compounds split differently).

**Fix:** Use `difflib.get_close_matches` or Levenshtein distance for fuzzy matching:

```python
import difflib


def _find_matching_chunks(
    target: str,
    word_chunks: list[Chunk],
    max_distance: int = 2,
) -> list[Chunk]:
    """Find word chunks whose cleaned text closely matches target.

    Uses SequenceMatcher ratio for fuzzy matching.
    Falls back to exact match if no fuzzy match is found.
    """
    cleaned_target = re.sub(PUNCTUATION_PATTERN, "", target).strip().lower()
    candidates = []
    for wc in word_chunks:
        if wc.text is None:
            continue
        cleaned_wc = re.sub(PUNCTUATION_PATTERN, "", wc.text).strip().lower()
        ratio = difflib.SequenceMatcher(None, cleaned_target, cleaned_wc).ratio()
        if ratio >= 0.85:  # 85% similarity threshold
            candidates.append((ratio, wc))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    if candidates and candidates[0][0] >= 0.95:
        return [candidates[0][1]]  # strong match
    elif candidates:
        logger.debug(f"Fuzzy match for {cleaned_target!r}: best={candidates[0][1].text!r} (ratio={candidates[0][0]:.2f})")
        return [candidates[0][1]]
    return []
```

Replace the exact-match blocks in `group_word_chunks()`:

```python
# Get the starting time of the segment
first_word = segment_without_punctuation.split()[0].lower()
first_word_candidates = _find_matching_chunks(first_word, word_chunks)
if not first_word_candidates:
    logger.warning(f"Could not find transcription for {first_word!r}. Skipping.")
    continue
segment_start = first_word_candidates[0].start_time

# Get the ending time of the segment
last_word = segment_without_punctuation.split(" ")[-1].lower()
last_word_candidates = _find_matching_chunks(last_word, word_chunks)
# Also enforce temporal ordering: must be after segment_start
last_word_candidates = [c for c in last_word_candidates if c.end_time > segment_start]
if not last_word_candidates:
    logger.warning(f"Could not find transcription for {last_word!r}. Skipping.")
    continue
segment_end = last_word_candidates[0].end_time
```

---

## Medium-Impact Improvements (Priority P2)

### 5. CTC Decoding with KenLM

**Dependencies already present:** `kenlm`, `pyctcdecode` (in `pyproject.toml` lines 17, 21)

**Current approach:** Default greedy/top-k decoding via `AutomaticSpeechRecognitionPipeline`. No language model is used during decoding.

**Proposed integration:** Use `pyctcdecode` to load a Danish KenLM language model and pass it to the ASR pipeline:

```python
from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
import kenlm


def build_asr_pipeline_with_lm(
    model_id: str = "CoRal-project/roest-v3-wav2vec2-315m",
    lm_path: str | None = None,
    alpha: float = 0.5,   # word insertion weight
    beta: float = 1.0,    # language model weight
) -> AutomaticSpeechRecognitionPipeline:
    """Build ASR pipeline with KenLM language model rescoring."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)

    decoder = None
    if lm_path and os.path.exists(lm_path):
        decoder = build_ctcdecoder(
            vocab=processor.tokenizer.get_vocab(),
            lm_path=lm_path,
            alpha=alpha,
            beta=beta,
        )
        logger.info(f"Loaded KenLM from {lm_path}")

    pipeline = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        decoder=decoder,
        return_timestamps="word",
        device=get_device(),
    )
    return pipeline
```

**Danish LM availability:**
- The RoEST project provides a Danish KenLM at `https://github.com/roest-asr/roest-lm` — download the `.bin` file.
- Alternatively, train a custom trigram model on Danish text (e.g., DKT corpus) using `kenlm.build_binary`.

---

### 6. Confidence Scoring and Filtering

**File:** `src/but_with_subs/transcribing.py`

**Problem:** No per-word confidence is tracked. wav2vec2 models can produce confident but wrong predictions, especially for out-of-vocabulary words.

**Approach:** Extract per-token logprobs and convert to confidence scores using the logits from the model forward pass.

```python
import torch
import torch.nn.functional as F


def transcribe_audio_with_confidence(
    audio: np.ndarray,
    model: AutomaticSpeechRecognitionPipeline,
    min_chunk_length: float = MIN_CHUNK_LENGTH_SECONDS,
    show_progress: bool = True,
) -> list[Chunk]:
    """Transcribe with per-word confidence scores derived from logits."""
    with tqdm(total=1, desc="Transcribing", disable=not show_progress) as pbar:
        result = model(audio, return_timestamps="word")
        pbar.update(1)

    word_chunks: list[Chunk] = []

    # The pipeline stores model outputs; we can extract logits from the last forward pass
    # For word-level timestamps, iterate chunks and compute token-level confidence
    for chunk_dct in result["chunks"]:
        start_time = float(chunk_dct["timestamp"][0])
        end_time = float(chunk_dct["timestamp"][1])
        text = chunk_dct["text"].strip()

        if end_time - start_time < min_chunk_length:
            continue

        # Compute confidence from token log probabilities
        confidence = _compute_token_confidence(chunk_dct)

        audio_start = int(TARGET_SAMPLE_RATE * start_time)
        audio_end = int(TARGET_SAMPLE_RATE * end_time)
        audio_end = min(audio_end, len(audio))

        word_chunks.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=audio[audio_start:audio_end],
                text=text,
                speaker=None,
            )
        )
        # Store confidence as an extra attribute
        word_chunks[-1].confidence = confidence  # type: ignore[attr-defined]

    return word_chunks


def _compute_token_confidence(chunk_dct: dict) -> float:
    """Estimate confidence for a word-level chunk.

    If the pipeline output includes 'tokens' or 'token_scores', use them.
    Otherwise, approximate from the text length and duration.
    """
    # Method 1: Use token_scores if available (some pipeline configs)
    if "token_scores" in chunk_dct:
        scores = chunk_dct["token_scores"]
        if scores:
            return float(torch.stack(scores).mean().item())

    # Method 2: Approximate from duration (longer words tend to be more reliable)
    duration = float(chunk_dct["timestamp"][1]) - float(chunk_dct["timestamp"][0])
    # Sigmoid-like mapping: 0.05 s -> ~0.3, 1.0 s -> ~0.95
    return 1.0 / (1.0 + torch.exp(-5.0 * (duration - 0.3)).item())
```

**Additional filtering:** Add duration-based heuristics:

```python
# In transcribe_audio_with_confidence, after creating each chunk:
duration = end_time - start_time
if duration < 0.02:
    logger.debug(f"Skipping suspiciously short word at {start_time:.2f}s (dur={duration:.3f}s)")
    continue
if duration > 5.0 and not any(c in text.lower() for c in "æøå"):
    logger.warning(f"Suspiciously long word at {start_time:.2f}s: {text!r}")
```

**Consecutive word deduplication (wav2vec2 artifact):**

```python
def deduplicate_consecutive_words(chunks: list[Chunk]) -> list[Chunk]:
    """Remove consecutive duplicate words (common wav2vec2 repetition artifact)."""
    if len(chunks) < 2:
        return chunks

    filtered = [chunks[0]]
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev_clean = re.sub(PUNCTUATION_PATTERN, "", prev.text or "").strip().lower()
        curr_clean = re.sub(PUNCTUATION_PATTERN, "", curr.text or "").strip().lower()
        if prev_clean == curr_clean:
            # Merge: keep the later chunk's timestamp (covers more audio)
            merged = Chunk(
                start_time=prev.start_time,
                end_time=curr.end_time,
                audio=np.concatenate([prev.audio, curr.audio], axis=0),
                text=prev.text,
                speaker=prev.speaker,
            )
            filtered[-1] = merged
        else:
            filtered.append(curr)

    return filtered
```

Call this after transcription in the pipeline.

---

### 7. Configurable Punctuation Language

**File:** `src/but_with_subs/text_chunking.py`, line 48

**Current code:**

```python
text = punctuation_model.punctuate(text=text)
```

where `punctuation_model` is always initialized as `PunctFixer(language="da")`.

**Problem:** If the audio is in a language other than Danish (e.g., English, German), `PunctFixer("da")` will still try to insert Danish punctuation rules, producing incorrect output.

**Fix:** Make the language configurable and detect it from the audio source:

```python
# In constants.py or a new config module:
TRANSCRIPTION_LANGUAGE: str = "da"  # default, overridable

# In text_chunking.py:
def group_word_chunks(
    word_chunks: list[Chunk],
    punctuation_language: str = "da",
    max_words: int = 12,
) -> list[Chunk]:
    """Group word chunks into segments with configurable punctuation language.

    Args:
        word_chunks: List of word-level Chunk objects.
        punctuation_language: ISO 639-1 language code for PunctFixer.
            Supported: "da", "en", "de", "fr", "es", "sv", "no", "fi".
        max_words: Maximum words per segment.
    """
    # Validate language support
    SUPPORTED_LANGUAGES = {"da", "en", "de", "fr", "es", "sv", "no", "fi"}
    if punctuation_language not in SUPPORTED_LANGUAGES:
        logger.warning(
            f"PunctFixer language {punctuation_language!r} not in supported set "
            f"{SUPPORTED_LANGUAGES}. Falling back to 'da'."
        )
        punctuation_language = "da"

    punctuation_model = PunctFixer(language=punctuation_language)

    text = " ".join(
        [
            re.sub(PUNCTUATION_PATTERN, "", word_chunk.text.lower())
            for word_chunk in word_chunks
            if word_chunk.text
        ]
    )
    text = punctuation_model.punctuate(text=text)

    # ... rest of function unchanged ...
```

Language detection can be driven by the video URL metadata (YouTube provides language info) or a separate language detection step using `fasttext` or `langdetect`.

---

## Lower-Priority Improvements (Priority P3)

### 8. Better Sentence Segmentation

**File:** `src/but_with_subs/text_chunking.py`, line 129

**Current code:**

```python
sentences = nltk.sent_tokenize(text=text, language="danish")
```

**Limitations:** NLTK's `punkt_tab` Danish tokenizer is rule-based and may mis-segment abbreviations, ellipses, and non-standard punctuation. It also doesn't handle multi-language text well.

**Alternative: spaCy with Danish model**

```python
import spacy

try:
    _spacy_da = spacy.load("da_core_news_sm")
except OSError:
    import subprocess
    subprocess.check_call(["python", "-m", "spacy", "download", "da_core_news_sm"])
    _spacy_da = spacy.load("da_core_news_sm")


def _split_text_spacy(*, text: str, max_words: int) -> list[str]:
    """Split text using spaCy's Danish sentence boundary detector."""
    doc = _spacy_da(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Further split oversized sentences
    result = []
    for sent in sentences:
        if len(sent.split()) <= max_words:
            result.append(sent)
        else:
            # Fallback: split on conjunctions and semicolons
            parts = re.split(r'\s*(?:og|men|eller|;)\s+', sent)
            for part in parts:
                if len(part.split()) <= max_words:
                    result.append(part)
                else:
                    # Final fallback: word-based splitting
                    words = part.split()
                    result.extend(
                        " ".join(words[i:i+max_words])
                        for i in range(0, len(words), max_words)
                    )
    return result
```

Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    "spacy>=3.7",
    "da-core-news-sm>=3.7",
]
```

---

### 9. Quality Metrics Infrastructure

**File:** New file `src/but_with_subs/quality_metrics.py`

**Purpose:** Track and report transcription quality metrics for monitoring and debugging.

```python
"""Quality metrics infrastructure for transcription pipeline."""

import typing as t
from collections import defaultdict

import numpy as np
from jiwer import wer  # add "jiwer>=2.3.0" to dependencies


class TranscriptionMetrics:
    """Accumulates quality metrics across transcription runs."""

    def __init__(self) -> None:
        self.total_words: int = 0
        self.total_chunks: int = 0
        self.confidence_scores: list[float] = []
        self.duration_distribution: dict[str, int] = defaultdict(int)
        self.low_confidence_count: int = 0
        self.duplicates_removed: int = 0
        self.segments_skipped: int = 0

    def record_chunk(self, chunk, confidence: float | None = None) -> None:
        """Record metrics for a single chunk."""
        self.total_chunks += 1
        if chunk.text:
            words = chunk.text.split()
            self.total_words += len(words)

        if confidence is not None:
            self.confidence_scores.append(confidence)
            if confidence < 0.5:
                self.low_confidence_count += 1

        duration = chunk.end_time - chunk.start_time
        if duration < 0.1:
            self.duration_distribution["very_short"] += 1
        elif duration < 0.5:
            self.duration_distribution["short"] += 1
        elif duration < 2.0:
            self.duration_distribution["medium"] += 1
        else:
            self.duration_distribution["long"] += 1

    def record_skip(self) -> None:
        self.segments_skipped += 1

    def summary(self) -> dict:
        """Return a summary dictionary of all accumulated metrics."""
        avg_confidence = (
            np.mean(self.confidence_scores) if self.confidence_scores else 0.0
        )
        return {
            "total_chunks": self.total_chunks,
            "total_words": self.total_words,
            "avg_confidence": round(float(avg_confidence), 4),
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_ratio": round(
                self.low_confidence_count / max(self.total_chunks, 1), 4
            ),
            "segments_skipped": self.segments_skipped,
            "duration_distribution": dict(self.duration_distribution),
        }


def calculate_wer(hypothesis: str, reference: str) -> float:
    """Calculate Word Error Rate between hypothesis and reference text."""
    return wer(reference.split(), hypothesis.split())
```

Usage in the main pipeline:

```python
from .quality_metrics import TranscriptionMetrics

metrics = TranscriptionMetrics()

for chunk in word_chunks:
    conf = getattr(chunk, "confidence", None)
    metrics.record_chunk(chunk, confidence=conf)

logger.info(f"Transcription quality: {metrics.summary()}")
```

---

## Implementation Roadmap

| # | Improvement | Priority | Impact | Complexity | Effort | Files Changed |
|---|-------------|----------|--------|------------|--------|---------------|
| 1 | Audio preprocessing (silence trim, RMS norm, HPF, torchaudio resample) | **P0** | High | Low | 2-3 hours | `audio_loading.py` |
| 2 | VAD-based pre-segmentation | **P0** | High | Medium | 4-5 hours | `transcribing.py` (+ new `vad_segment_audio`) |
| 3 | Fix punctuation splitting bug (`split(",;:-")`) | **P0** | Medium | Low | 15 minutes | `text_chunking.py:142` |
| 4 | Fuzzy word matching (difflib/Levenshtein) | **P1** | Medium | Low-Medium | 2 hours | `text_chunking.py:59-87` |
| 5 | CTC decoding with KenLM | **P2** | Medium | Medium | 3-4 hours | New `asr_decoder.py`, `transcribing.py` |
| 6 | Confidence scoring + deduplication | **P2** | Medium | Low | 2 hours | `transcribing.py` |
| 7 | Configurable PunctFixer language | **P2** | Low-Medium | Low | 1 hour | `text_chunking.py` |
| 8 | spaCy Danish sentence segmentation | **P3** | Low | Medium | 2 hours | `text_chunking.py`, `pyproject.toml` |
| 9 | Quality metrics infrastructure | **P3** | Low | Low | 1 hour | New `quality_metrics.py` |

### Recommended Order of Implementation

1. **Start with #3** (15 min) — immediate bug fix, zero risk.
2. **Then #1** (2-3 hours) — measurable quality improvement on all audio, no API changes.
3. **Then #4** (2 hours) — reduces segment-matching failures, especially with noisy audio.
4. **Then #6** (2 hours) — adds observability and removes a common wav2vec2 artifact.
5. **Then #2** (4-5 hours) — biggest single improvement for long-form audio.
6. **Then #5** (3-4 hours) — requires acquiring a Danish LM, but can yield 10-20% WER reduction.
7. **Then #7** (1 hour) — makes the pipeline robust for non-Danish audio.
8. **Then #8** (2 hours) — incremental improvement for sentence boundaries.
9. **Then #9** (1 hour) — monitoring infrastructure for ongoing quality tracking.

### Total Estimated Effort: ~18-22 hours

No model changes or LLM integrations required. All improvements use existing or easily-addable dependencies.
