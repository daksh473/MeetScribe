"""Tests for audio file validator."""

import subprocess
import pytest
from pathlib import Path

from config import settings
from stt.errors import (
    CorruptAudioError,
    EmptyFileError,
    FileNotFoundAudioError,
    NoAudioStreamError,
    TooLongError,
    TooShortError,
    UnsupportedFormatError,
)
from stt.validator import validate_audio


@pytest.fixture
def temp_audio_factory(tmp_path: Path):
    """Factory fixture to generate synthetic test audio files using ffmpeg."""

    def _generate(filename: str, duration: float = 3.0, channels: int = 1, sample_rate: int = 16000) -> Path:
        out_file = tmp_path / filename
        cmd = [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1000:duration={duration}:sample_rate={sample_rate}",
            "-ac",
            str(channels),
            str(out_file),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"ffmpeg failed: {res.stderr}"
        return out_file

    return _generate


def test_validate_valid_file(temp_audio_factory):
    """Valid WAV file passes validation and returns correct AudioInfo."""
    audio_path = temp_audio_factory("valid.wav", duration=3.0, channels=1, sample_rate=16000)
    info = validate_audio(audio_path, min_duration_s=1.0, max_duration_s=10.0)

    assert info.duration == pytest.approx(3.0, rel=0.05)
    assert info.channels == 1
    assert info.sample_rate == 16000
    assert "pcm" in info.codec.lower()


def test_validate_empty_file(tmp_path: Path):
    """Empty 0-byte file raises EmptyFileError."""
    empty_file = tmp_path / "empty.mp3"
    empty_file.touch()

    with pytest.raises(EmptyFileError) as exc_info:
        validate_audio(empty_file)

    assert "empty" in str(exc_info.value).lower()


def test_validate_fake_mp3_corrupt(tmp_path: Path):
    """Text file renamed to .mp3 raises user-friendly CorruptAudioError."""
    fake_mp3 = tmp_path / "not_audio.mp3"
    fake_mp3.write_text("This is plain text and definitely not an MP3 audio file.", encoding="utf-8")

    with pytest.raises(CorruptAudioError) as exc_info:
        validate_audio(fake_mp3)

    assert "corrupt or could not be decoded" in str(exc_info.value)


def test_validate_unsupported_extension(tmp_path: Path):
    """File with unsupported extension raises UnsupportedFormatError."""
    bad_ext_file = tmp_path / "recording.txt"
    bad_ext_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError) as exc_info:
        validate_audio(bad_ext_file)

    assert ".txt" in str(exc_info.value)
    assert "Supported formats" in str(exc_info.value)


def test_validate_file_not_found():
    """Non-existent file raises FileNotFoundAudioError."""
    missing = Path("non_existent_audio_file.wav")
    with pytest.raises(FileNotFoundAudioError) as exc_info:
        validate_audio(missing)

    assert "not found" in str(exc_info.value).lower()


def test_validate_no_audio_stream(tmp_path: Path):
    """Video file without audio stream raises NoAudioStreamError."""
    video_only = tmp_path / "video_only.mp4"
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=160x120:rate=1",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(video_only),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"ffmpeg failed: {res.stderr}"

    with pytest.raises(NoAudioStreamError) as exc_info:
        validate_audio(video_only)

    assert "No audio stream found" in str(exc_info.value)


def test_validate_too_short(temp_audio_factory):
    """Audio shorter than minimum duration raises TooShortError."""
    short_audio = temp_audio_factory("short.wav", duration=1.0)

    with pytest.raises(TooShortError) as exc_info:
        validate_audio(short_audio, min_duration_s=2.0)

    assert "too short" in str(exc_info.value).lower()
    assert exc_info.value.min_duration == 2.0


def test_validate_too_long(temp_audio_factory):
    """Audio exceeding maximum duration raises TooLongError."""
    long_audio = temp_audio_factory("long.wav", duration=5.0)

    with pytest.raises(TooLongError) as exc_info:
        validate_audio(long_audio, max_duration_s=3.0)

    assert "exceeds the maximum" in str(exc_info.value).lower()
    assert exc_info.value.max_duration == 3.0
