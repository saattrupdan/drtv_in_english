"""Audio extraction module for extracting audio from video files.

This module provides a function to extract audio from video files using
ffmpeg, outputting the result as a WAV file.
"""

import subprocess
from pathlib import Path

from .logging_config import logger


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
    logger.info(f"Extracting audio from '{video_path}' to '{output_path}'")
    command: list[str] = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-n",
        "-v",
        "error",
        str(output_path),
    ]
    subprocess.run(args=command, check=True)
    return output_path
