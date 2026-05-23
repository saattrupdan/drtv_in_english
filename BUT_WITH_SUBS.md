# But With Subs Project

The core idea of this project is to make it possible to watch a movie or a TV show with
subtitles in whatever language you want, even if it's on a streaming service.

## Stack

- FastAPI for the API backend
- Vue.js and Vite for the frontend
- Postgres database for storing references between URLs and files

## Development plan

### Phase 1: Downloading video and audio ✅

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

### Phase 2: Chunking audio ✅

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

### Phase 3: Transcribing audio ✅

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

### Phase 4: Generating subtitle files ✅

We should have a convenience function that converts a list of `Transcription` models
into a `vtt` file.

This should result in a `generate_subtitles` function in a separate `subtitling` module.
The input should be a list of `Transcription` models defined above. The function should
yield progress along the way, and at the end return a `pathlib` Path object pointing to
the generated `vtt` file. The generated `vtt` file should be named the same as the input
audio file, but with a `.vtt` extension.

Check that it works on the sample URLs above.

Ensure good logging along the way, with the same logger as defined in Phase 1.

### Phase 5: Translating subtitles ✅

We should have a `translate` function in a separate `translation` module that takes
transcribed text, a target language, and an LLM configuration, and returns the
translated text. It uses the LLM module (`llm.py`) to perform the translation via an
LLM API.

A `--language` command-line parameter should be added to the workflow so the user
can specify the desired output language. When provided, each transcription segment is
translated into the target language before subtitle generation. The translation happens
after transcription and before subtitling, replacing the original text in each
`Transcription` model with its translated counterpart.

### Phase 6: Creating UI ✅

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

There should also be a button in the UI to go back to home, to process another video.

### Phase 7: API ✅

The API should be a FastAPI app that exposes the POST endpoint `/process` that takes a
URL and does the following:

1. Downloads the video and audio using the `download` function from Phase 1. The yielded
   progress updates should be captured in a generator, which will be the
   `StreamingResponse` to the endpoint.
2. When the video and audio finish downloading we get the output `File` model from the
   function, and these should be stored in the database. The database should have a
   `files` table with the following columns:

   - `url (str)`: The URL of the video (primary key)
   - `video_path (str)`: The absolute path to the downloaded video file, if any
   - `audio_path (str)`: The absolute path to the downloaded audio file, if any
   - `subtitles_path (str)`: The absolute path to the generated subtitles file, if any
   - `created_at (datetime)`: The timestamp when the file was created

   Note that we don't set the `subtitles_path` yet, since we haven't generated the
   subtitles yet.
3. Next, we chunk the audio using the `chunk_audio` function from Phase 2. The yielded
   chunks are passed through to the `transcribe` function from Phase 3, and we capture
   the progress updates in the generator. When the transcription finishes, we get a list
   of `Transcription` models.
4. We next call the `generate_subtitles` function from Phase 4, and capture the progress
   updates in the generator. When the subtitles are generated, we get a `Path` object
   pointing to the generated `vtt` file, and we store this in the database.
5. Finally, we add a final 'completed' progress update to the generator, which also
   contains a `VideoWithSubs` model with the following fields:

   - `video_path (str)`: The absolute path to the downloaded video file, with audio
  merged
   - `subtitles_path (str)`: The absolute path to the generated subtitles file

### Phase 8: Integrate UI with API ✅

When the submission is triggered, the frontend should make a POST request to the API
endpoint `/process` with the URL as the request body. The frontend should then display
a progress bar with the progress updates from the API, in the same progress bar as was
created in Phase 5.

When the progress is finished, the frontend receives the `VideoWithSubs` model defined
in Phase 6, and the values of these are then displayed in the HTML5 video element,
defined in Phase 5.

### Phase 9: Set up Docker ✅

Docker Compose is already set up, but you should add a postgres service (use
`18.3-trixie` as the version) to the `docker-compose.yaml` file. You have to ensure that
the backend installs `ffmpeg`, as that's required for the downloading to work. Also
ensure that the NGINX config in `docker-compose.nginx.conf` is set up properly to handle
the streaming of the API responses, and also in terms of timeouts, as the processing can
take a long time (maybe have the timeout be 1 hour).

### Phase 10: Documentation ✅

Update the readme with the following sections:

- "Quick Start", which uses the `docker compose up --build --remove-orphans` command to
  start the app
- "Local Development", which shows how to install the repo (`make install`) and to run
  the app in development mode (`npm run dev` and `uv run fastapi dev
  src/but_with_subs/api.py`)
- "Workflow", describing the workflow from the user sends a request, with diagrams
- "Stack", describing the stack used in the project
- "Contribute", on how to contribute to the project. Refer to the `CONTRIBUTING.md` file

The readme should also feature the logo in `./public/but-with-subs-logo.jpg` at the top.

### Phase 11: Backend tests

Create tests for all the backend modules, and ensure they all pass with `make test`.
Include both simple unit tests as well as full integration tests that test the entire
flow from downloading the video to generating the subtitles (including the API). We
should use a short dummy clip for this: ./data/test.mp4
