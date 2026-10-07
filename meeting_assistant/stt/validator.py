"""Audio validation module using ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from config import settings
from .errors import (
    CorruptAudioError,
    EmptyFileError,
    FileNotFoundAudioError,
    NoAudioStreamError,
    TooLongError,
    TooShortError,
    UnsupportedFormatError,
)
from .schema import AudioInfo


def validate_audio(
    path: Union[str, Path],
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    allowed_extensions: Optional[Sequence[str]] = None,
    ffprobe_path: Optional[str] = None,
) -> AudioInfo:
    """Validate that audio file exists, is supported, readable, and within duration limits.

    Trusts ffprobe's true content inspection over file extensions.

    Raises:
        FileNotFoundAudioError: If file does not exist.
        EmptyFileError: If file size is 0 bytes.
        UnsupportedFormatError: If file extension is not in allowed list.
        CorruptAudioError: If ffprobe cannot parse the file container/stream.
        NoAudioStreamError: If file has no valid audio streams.
        TooShortError: If audio duration is below min_duration_s.
        TooLongError: If audio duration exceeds max_duration_s.
    """
    file_path = Path(path)

    # 1. Check file existence
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundAudioError(str(file_path))

    # 2. Check empty file
    try:
        if file_path.stat().st_size == 0:
            raise EmptyFileError()
    except OSError as e:
        raise CorruptAudioError(details=f"Could not stat file: {e}") from e

    # 3. Check extension against allowlist
    allowed = allowed_extensions or settings.supported_formats
    ext = file_path.suffix.lower()
    if ext not in allowed:
        raise UnsupportedFormatError(ext=ext, allowed=allowed)

    # 4. Probe content using ffprobe
    probe_bin = ffprobe_path or settings.ffprobe_path
    cmd = [
        probe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(file_path.resolve()),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise CorruptAudioError(
            message="ffprobe binary was not found on the system. Please ensure ffmpeg/ffprobe is installed.",
            details=str(e),
        ) from e
    except Exception as e:
        raise CorruptAudioError(
            message="Failed to inspect audio file with ffprobe.",
            details=str(e),
        ) from e

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        raise CorruptAudioError(
            message="The audio file is corrupt or could not be decoded. Please upload a valid media file.",
            details=f"ffprobe exit code {proc.returncode}: {err_msg}",
        )

    try:
        probe_data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise CorruptAudioError(
            message="Unable to parse audio stream metadata.",
            details=str(e),
        ) from e

    # 5. Check for valid audio streams
    streams = probe_data.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise NoAudioStreamError()

    primary_stream = audio_streams[0]

    # 6. Extract duration (check stream duration first, then format duration)
    duration_str = primary_stream.get("duration")
    if not duration_str or duration_str == "N/A":
        duration_str = probe_data.get("format", {}).get("duration")

    if not duration_str or duration_str == "N/A":
        raise CorruptAudioError(
            message="Could not determine audio duration from file.",
            details="Missing duration tag in both stream and format headers.",
        )

    try:
        duration = float(duration_str)
    except (ValueError, TypeError) as e:
        raise CorruptAudioError(
            message="Audio file has an invalid duration header.",
            details=str(e),
        ) from e

    if duration <= 0:
        raise CorruptAudioError(
            message="Audio duration is invalid or zero seconds.",
            details=f"Parsed duration: {duration}",
        )

    # 7. Check duration bounds
    min_dur = min_duration_s if min_duration_s is not None else settings.min_duration_s
    max_dur = max_duration_s if max_duration_s is not None else settings.max_duration_s

    if duration < min_dur:
        raise TooShortError(duration=duration, min_duration=min_dur)
    if duration > max_dur:
        raise TooLongError(duration=duration, max_duration=max_dur)

    # 8. Extract stream properties
    codec = primary_stream.get("codec_name", "unknown")
    sample_rate = int(primary_stream.get("sample_rate", 0))
    channels = int(primary_stream.get("channels", 1))

    bit_rate_val = primary_stream.get("bit_rate") or probe_data.get("format", {}).get("bit_rate")
    bit_rate = int(bit_rate_val) if bit_rate_val and str(bit_rate_val).isdigit() else None

    return AudioInfo(
        duration=duration,
        codec=codec,
        sample_rate=sample_rate,
        channels=channels,
        bit_rate=bit_rate,
    )
