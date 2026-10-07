"""Custom domain exceptions for STT pipeline with user-friendly UI messages."""

from __future__ import annotations

from typing import Iterable, Optional


class STTError(Exception):
    """Base class for all STT exceptions safe to display in UI."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        return self.message


class UnsupportedFormatError(STTError):
    """Raised when file extension or container is not supported."""

    def __init__(
        self,
        ext: str,
        allowed: Optional[Iterable[str]] = None,
        details: Optional[str] = None,
    ) -> None:
        allowed_str = ", ".join(allowed) if allowed else "mp3, wav, m4a, mp4, ogg, flac, webm, aac"
        msg = f"Unsupported audio format '{ext}'. Supported formats: {allowed_str}."
        super().__init__(msg, details=details)
        self.ext = ext


class EmptyFileError(STTError):
    """Raised when audio file size is 0 bytes."""

    def __init__(self, message: str = "The audio file is empty (0 bytes).", details: Optional[str] = None) -> None:
        super().__init__(message, details=details)


class FileNotFoundAudioError(STTError):
    """Raised when specified audio path does not exist."""

    def __init__(self, path: str, details: Optional[str] = None) -> None:
        super().__init__(f"Audio file was not found: '{path}'.", details=details)
        self.path = path


class CorruptAudioError(STTError):
    """Raised when file cannot be parsed or decoded by ffprobe/ffmpeg."""

    def __init__(
        self,
        message: str = "The audio file is corrupt or could not be decoded. Please upload a valid media file.",
        details: Optional[str] = None,
    ) -> None:
        super().__init__(message, details=details)


class NoAudioStreamError(STTError):
    """Raised when media file contains video/data but no valid audio track."""

    def __init__(
        self,
        message: str = "No audio stream found in the uploaded file.",
        details: Optional[str] = None,
    ) -> None:
        super().__init__(message, details=details)


class NoSpeechError(STTError):
    """Raised when audio file contains pure silence or no decodable speech."""

    def __init__(
        self,
        message: str = "No spoken content detected in the audio recording.",
        details: Optional[str] = None,
    ) -> None:
        super().__init__(message, details=details)


class TooShortError(STTError):
    """Raised when audio duration is less than the allowed minimum."""

    def __init__(
        self,
        duration: float,
        min_duration: float,
        details: Optional[str] = None,
    ) -> None:
        msg = (
            f"The audio recording is too short ({duration:.1f}s). "
            f"Minimum required duration is {min_duration:.1f}s."
        )
        super().__init__(msg, details=details)
        self.duration = duration
        self.min_duration = min_duration


class TooLongError(STTError):
    """Raised when audio duration exceeds the allowed maximum."""

    def __init__(
        self,
        duration: float,
        max_duration: float,
        details: Optional[str] = None,
    ) -> None:
        msg = (
            f"The audio recording ({duration:.1f}s) exceeds the maximum "
            f"allowed duration of {max_duration:.1f}s."
        )
        super().__init__(msg, details=details)
        self.duration = duration
        self.max_duration = max_duration


class NonEnglishAudioError(STTError):
    """Raised when input audio is identified as non-English."""

    def __init__(
        self,
        detected_lang: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        lang_info = f" (detected '{detected_lang}')" if detected_lang else ""
        msg = (
            f"Non-English audio detected{lang_info}. "
            "This assistant only processes English-language meetings."
        )
        super().__init__(msg, details=details)
        self.detected_lang = detected_lang
