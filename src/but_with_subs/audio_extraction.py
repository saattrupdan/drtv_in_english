"""Audio extraction module for extracting audio from video files.

This module provides a function to extract audio from video files using
ffmpeg, outputting the result as a WAV file.
"""

import subprocess
from pathlib import Path

from .logging_config import logger


def _run_ffmpeg_extract(input_path: Path, output_path: Path) -> None:
    """Run ffmpeg to extract audio from a video file.

    Args:
        input_path:
            Path to the input video file.
        output_path:
            Path to the output WAV file.
    """
    logger.info("Extracting audio from %s to %s", input_path, output_path)

    command: list[str] = ["ffmpeg", "-i", str(input_path), "-vn", str(output_path)]

    subprocess.run(args=command, check=True)


def extract_audio(video_path: Path) -> Path:
    """Extract audio from a video file using ffmpeg.

    Runs ffmpeg to extract audio from the specified video file and saves
    it as a WAV file in the same directory.

    Args:
        video_path:
            Path to the input video file.

    Returns:
        Path to the extracted WAV file.

    """
    output_path = video_path.with_suffix(suffix=".wav")

    logger.info("Extracting audio from %s to %s", video_path, output_path)

    _run_ffmpeg_extract(input_path=video_path, output_path=output_path)

    return output_path
