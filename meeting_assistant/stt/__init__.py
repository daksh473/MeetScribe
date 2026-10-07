"""Speech-To-Text module for Meeting Assistant."""

from .errors import (
    CorruptAudioError,
    EmptyFileError,
    FileNotFoundAudioError,
    NoAudioStreamError,
    NoSpeechError,
    STTError,
    TooLongError,
    TooShortError,
    UnsupportedFormatError,
)
from .schema import (
    AudioInfo,
    EngineResult,
    Segment,
    STTMetadata,
    STTResult,
    UncertainSpan,
    Word,
)
from .pipeline import run_stt

__all__ = [
    "run_stt",
    "Word",
    "Segment",
    "EngineResult",
    "UncertainSpan",
    "STTMetadata",
    "STTResult",
    "AudioInfo",
    "STTError",
    "UnsupportedFormatError",
    "EmptyFileError",
    "FileNotFoundAudioError",
    "CorruptAudioError",
    "NoAudioStreamError",
    "NoSpeechError",
    "TooShortError",
    "TooLongError",
]
