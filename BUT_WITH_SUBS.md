# But With Subs Project

The core idea of this project is to make it possible to watch a movie or a TV show with
subtitles in whatever language you want, even if it's on a streaming service.

## Stack

- FastAPI for the API backend
- Vue.js and Vite for the frontend
- Postgres database for storing references between URLs and files

## Development plan

### Phase 1: Downloading video and audio

Implement `download` function in a separate `downloading` module that takes a URL and
downloads the video and audio using the `yt-dlp` library - docs for that library is
here:

<https://raw.githubusercontent.com/yt-dlp/yt-dlp/refs/heads/master/README.md>

The downloaded files should be stored in the `./data/` directory.

When the download is running, it will output progress to stdout or stderr, and you need
to monitor this, and yield the progress along the way. When the download has finished,
the function should return a Pydantic `File` model with the following fields:

- `url`: The URL of the video
- `video_path`: The path to the downloaded video file, if any
- `audio_path`: The path to the downloaded audio file, if any

If there are multiple video and/or audio paths, then we pick the first one (we might
change this later if necessary).

Some URLs to check if it works:

- <https://www.dr.dk/drtv/serie/kommissionen_589959>
- <https://www.dr.dk/drtv/program/doeden-paa-nilen_575463>
- <https://viaplay.dk/player/default//serier/the-good-doctor/saeson-1/afsnit-1>

Maybe we need a login to make it work, but we need to figure this out.

Ensure good logging along the way (use `logger = logging.getLogger(__package__)`), and
make sure that logging is configured somewhere central (not just in the module where it
is used).

### Phase 2: Chunking audio

The audio file should be split into chunks. We should chunk based on natural breaks in
the audio (silence below a certain threshold, or other such heuristics). This should
result in a `chunk_audio` function in a separate `chunking` module.

The function should take an `audio_path` argument (`pathlib` Path object) and yield
Pydantic `Chunk` models with the following fields:

- `start_time`: The start time of the chunk, in seconds from the beginning of the audio
- `end_time`: The end time of the chunk, in seconds from the beginning of the audio
- `audio`: The mono audio data of the chunk, as a `numpy` array in 16 kHz, of shape
  `(audio_length,)`

Check that it works on the sample URLs above.

Ensure good logging along the way, with the same logger as defined in Phase 1.

### Phase 3: Transcribing audio

We should use `pipeline` function from the `transformer` library with Wav2Vec2 models to
transcribe the audio chunks. This should result in a `transcribe` function in a separate
`transcribing` module, which takes a `numpy` array of audio data as input (same format
as in the `Chunk` model above) and a `AutomaticSpeechRecognitionPipeline` object from
`transformers`, and returns a list of `Transcription` models with the following fields:

- `start_time`: The start time of the transcription, in seconds from the beginning of the
  full audio (not just the chunk)
- `end_time`: The end time of the transcription, in seconds from the beginning of the
  full audio (not just the chunk)
- `text`: The transcribed text

Check that it works on the sample URLs above.

Ensure good logging along the way, with the same logger as defined in Phase 1.

### Phase 4: Generating subtitle files

We should have a convenience function that converts a list of `Transcription` models
into a `vtt` file.

This should result in a `generate_subtitles` function in a separate `subtitling` module.
The input should be a list of `Transcription` models defined above. The function should
yield progress along the way, and at the end return a `pathlib` Path object pointing to
the generated `vtt` file. The generated `vtt` file should be named the same as the input
audio file, but with a `.vtt` extension.

Check that it works on the sample URLs above.

Ensure good logging along the way, with the same logger as defined in Phase 1.

### Phase 5: Creating UI

The UI should look very professional and sleek, and should include the title "... But
With Subs" and logo (found in `./public/but-with-subs-logo.jpg`) and the favicon (found
in `./public/favicon.png`).

Don't include a gradient background, that's quite tacky.

The UI should include a text field for entering a URL, and a button for submitting the
URL, with a "Watch with Subs" label (pressing the Enter key should also submit the URL).
The user should also be able to drag-and-drop a file onto the URL field or browse for a
file using a file picker.

When the user submits a URL, the UI should change to a loading screen, where there is a
pretty progress bar that shows the progress of the download and transcription process.
0-50% should be used for the download progress, and 50-100% should be used for the
transcription progress (transcription really also includes chunking, but we don't need
to show this to the user).

When the progress reaches 100%, the progress bar should move up near the top of the page
in a fluent motion, and a message should appear saying "Ready to watch!" (not an alert
though, never use alerts). Could be a toast notification or something. Below the 100%
progress bar, a HTML5 video element should be displayed, which is ready to play the
video along with the generated subtitles. There should be no auto play, it should just
be ready to play.

When the user plays the video, the progress bar and the "Ready to watch!" message should
disappear, and the video should play with the generated subtitles, in full screen mode.
The player should have the usual controls, and if the user exits full screen mode, then
we only see the video player (along with the title+logo) and the controls.

Note that in this phase, there's no calling the API yet, we just want to make sure that
the UI works. The progress bar should just slowly move up to 100%, imitating the
download and transcription process. The video player should just play an empty video.

### Phase 6: API
