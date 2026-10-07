"""Configuration module for STT pipeline.

Loads environment variables from .env and defines typed settings for audio
validation, preprocessing, chunking, and STT engines.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def resolve_ffmpeg_path() -> str:
    """Find the best available ffmpeg binary."""
    custom = os.getenv("FFMPEG_PATH")
    if custom and Path(custom).is_file():
        return str(Path(custom).resolve())

    # Check Gyan.FFmpeg winget location if on Windows
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        winget_pattern = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pattern.is_dir():
            matches = list(winget_pattern.glob("**/ffmpeg.exe"))
            if matches:
                return str(matches[0].resolve())

    # Check imageio_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(Path(exe).resolve())
    except ImportError:
        pass

    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    return "ffmpeg"


def resolve_ffprobe_path() -> str:
    """Find the best available ffprobe binary."""
    custom = os.getenv("FFPROBE_PATH")
    if custom and Path(custom).is_file():
        return str(Path(custom).resolve())

    # Check Gyan.FFmpeg winget location if on Windows
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        winget_pattern = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pattern.is_dir():
            matches = list(winget_pattern.glob("**/ffprobe.exe"))
            if matches:
                return str(matches[0].resolve())

    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path

    return "ffprobe"


class Settings(BaseSettings):
    # API Keys
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")

    # Engine selection and order
    stt_engines: str = Field(default="elevenlabs,whisper", alias="STT_ENGINES")

    # Whisper Engine Settings
    whisper_model_size: str = Field(default="large-v3", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="auto", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="default", alias="WHISPER_COMPUTE_TYPE")

    # Supported audio extensions (allowlist)
    supported_formats: tuple[str, ...] = (
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".ogg",
        ".flac",
        ".webm",
        ".aac",
    )

    # Audio validation constraints
    min_duration_s: float = Field(default=2.0, alias="AUDIO_MIN_DURATION_S")
    max_duration_s: float = Field(default=10800.0, alias="AUDIO_MAX_DURATION_S")  # 3 hours

    # Audio chunking
    chunk_max_s: float = Field(default=600.0, alias="AUDIO_CHUNK_MAX_S")  # 10 minutes
    chunk_overlap_s: float = Field(default=1.5, alias="AUDIO_CHUNK_OVERLAP_S")
    enable_denoising: bool = Field(default=False, alias="AUDIO_ENABLE_DENOISING")

    # Binary tools
    ffmpeg_path: str = Field(default_factory=resolve_ffmpeg_path)
    ffprobe_path: str = Field(default_factory=resolve_ffprobe_path)

    @property
    def engines_list(self) -> List[str]:
        return [e.strip() for e in self.stt_engines.split(",") if e.strip()]


settings = Settings()
