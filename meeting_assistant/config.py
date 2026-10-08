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
original_environ = os.environ.copy()
load_dotenv(BASE_DIR / ".env", override=True)


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
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # LLM Models
    llm1_model: str = Field(default="", alias="LLM1_MODEL")
    llm2_model: str = Field(default="", alias="LLM2_MODEL")
    fallback_models: str = Field(default="", alias="FALLBACK_MODELS")
    llm_rate_limit_rpm: int = Field(default=20, alias="LLM_RATE_LIMIT_RPM")

    @property
    def fallback_models_list(self) -> List[str]:
        return [m.strip() for m in self.fallback_models.split(",") if m.strip()]

    # Engine selection and order
    stt_engines: str = Field(default="whisper,groq_whisper,elevenlabs", alias="STT_ENGINES")
    stt_engine_weights: str = Field(default="elevenlabs=1.2,groq_whisper=1.0,whisper=1.0", alias="STT_ENGINE_WEIGHTS")

    @property
    def engines_list(self) -> List[str]:
        return [e.strip() for e in self.stt_engines.split(",") if e.strip()]

    @property
    def engine_priors(self) -> dict[str, float]:
        priors = {}
        for pair in self.stt_engine_weights.split(","):
            if "=" in pair:
                eng, w = pair.split("=", 1)
                try:
                    priors[eng.strip()] = float(w.strip())
                except ValueError:
                    pass
        return priors

    # Whisper Engine Settings
    whisper_model_size: str = Field(default="large-v3", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="auto", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="default", alias="WHISPER_COMPUTE_TYPE")
    groq_stt_model: str = Field(default="whisper-large-v3-turbo", alias="GROQ_STT_MODEL")

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

def is_placeholder(val: str) -> bool:
    if not val:
        return True
    val = val.lower()
    if "<" in val and ">" in val:
        return True
    if "your_" in val and "_here" in val:
        return True
    if "model-id" in val or "free-model-id" in val or "replace_me" in val:
        return True
    return False

def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse provider:model_id splitting ONLY on the FIRST colon."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty model spec")
    if ":" not in spec:
        raise ValueError(f"Invalid model spec (missing colon): {spec}")
    parts = spec.split(":", 1)
    return parts[0].strip(), parts[1].strip()

# Pre-process environment variables (strip quotes/whitespace, handle placeholders, and streamlit secrets)
try:
    import streamlit as st
    secrets = st.secrets
except ImportError:
    secrets = {}

def _clean_env(val: str | None) -> str:
    if not val:
        return ""
    if is_placeholder(val):
        return ""
    
    # Auto-strip
    original = val
    val = val.strip()
    val = val.strip("'\"")
    if val.startswith("Bearer "):
        val = val[7:].strip()
    # Remove newlines
    val = val.replace("\n", "").replace("\r", "")
    # Remove non-ascii
    val = "".join(c for c in val if ord(c) < 128)
    
    return val

os.environ["GROQ_API_KEY"] = _clean_env(secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")))
os.environ["ELEVENLABS_API_KEY"] = _clean_env(secrets.get("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY")))
os.environ["OPENROUTER_API_KEY"] = _clean_env(secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")))

settings = Settings()

try:
    import ctranslate2
    has_gpu = ctranslate2.get_cuda_device_count() > 0
except Exception:
    has_gpu = False

if settings.whisper_device == "auto":
    settings.whisper_device = "cuda" if has_gpu else "cpu"
elif settings.whisper_device == "cuda" and not has_gpu:
    settings.whisper_device = "cpu"
    print("Notice: CUDA requested but no GPU found. Falling back to CPU.", flush=True)

if settings.whisper_device == "cpu":
    if settings.whisper_compute_type != "int8":
        settings.whisper_compute_type = "int8"
        print("Notice: Whisper on CPU must use int8 compute type. Forcing int8.", flush=True)


def mask(key: str) -> str:
    if not key or len(key) < 6:
        return "***"
    return f"{key[:4]}...{key[-2:]}"

import sys

# Validations
if not settings.groq_api_key:
    if "pytest" not in sys.modules:
        print("\n[FAIL] Missing required key: GROQ_API_KEY. Please set it in .env", file=sys.stderr)
        sys.exit(1)

if not settings.elevenlabs_api_key:
    print("Warning: ELEVENLABS_API_KEY is missing. Using Whisper-only STT fallback.", flush=True)

if not settings.openrouter_api_key:
    print("Warning: OPENROUTER_API_KEY is missing. LLM fallback options using OpenRouter will fail.", flush=True)
