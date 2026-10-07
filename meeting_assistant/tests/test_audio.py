"""Tests for audio normalization and silence-based chunking."""

import subprocess
import pytest
from pathlib import Path

from config import settings
from stt.audio import normalize_audio, split_on_silence
from stt.validator import validate_audio


@pytest.fixture
def test_audio_factory(tmp_path: Path):
    """Fixture to build test audio files."""

    def _build_sine(filename: str, duration: float = 3.0, sample_rate: int = 44100, channels: int = 2) -> Path:
        out_path = tmp_path / filename
        cmd = [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}:sample_rate={sample_rate}",
            "-ac",
            str(channels),
            str(out_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"ffmpeg error: {res.stderr}"
        return out_path

    return _build_sine


def test_normalize_audio_to_16khz_mono(test_audio_factory, tmp_path: Path):
    """normalize_audio converts 44.1 kHz stereo audio to 16 kHz mono WAV."""
    raw_audio = test_audio_factory("raw_stereo.wav", duration=3.0, sample_rate=44100, channels=2)
    normalized_path = tmp_path / "normalized.wav"

    res_path = normalize_audio(raw_audio, normalized_path)
    assert res_path.exists()

    info = validate_audio(normalized_path, min_duration_s=1.0)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert "pcm" in info.codec.lower()


def test_normalize_extracts_audio_from_video(tmp_path: Path):
    """normalize_audio extracts audio track from video files and discards video."""
    video_with_audio = tmp_path / "meeting_video.mp4"
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=3:size=160x120:rate=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=500:duration=3:sample_rate=48000",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(video_with_audio),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"ffmpeg video create failed: {res.stderr}"

    out_wav = tmp_path / "video_extracted.wav"
    normalize_audio(video_with_audio, out_wav)

    info = validate_audio(out_wav, min_duration_s=1.0)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert "pcm" in info.codec.lower()


def test_split_on_silence_short_file(test_audio_factory, tmp_path: Path):
    """Audio shorter than max_chunk_s is returned as a single chunk without cutting."""
    short_audio = test_audio_factory("short_meeting.wav", duration=4.0)
    chunks = split_on_silence(short_audio, max_chunk_s=10.0, overlap_s=1.0)

    assert len(chunks) == 1
    chunk_path, offset_s = chunks[0]
    assert offset_s == 0.0
    assert chunk_path.resolve() == short_audio.resolve()


def test_split_on_silence_multiminute_synthetic_file(tmp_path: Path):
    """Multi-minute synthetic file splits at silence points with correct global offsets."""
    # Synthesize ~154s file: 50s sound + 2s silence + 50s sound + 2s silence + 50s sound
    multi_file = tmp_path / "multiminute_synthetic.wav"
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=50:sample_rate=16000",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=50:sample_rate=16000",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=660:duration=50:sample_rate=16000",
        "-filter_complex",
        "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[out]",
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(multi_file),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"ffmpeg multimin synthesis failed: {res.stderr}"

    info = validate_audio(multi_file)
    assert info.duration >= 150.0

    # Split with max_chunk_s=60.0 and overlap_s=2.0
    chunks_dir = tmp_path / "custom_chunks"
    chunks = split_on_silence(multi_file, max_chunk_s=60.0, overlap_s=2.0, out_dir=chunks_dir)

    # Should produce 3 chunks split at silence boundaries (~51s and ~104s)
    assert len(chunks) == 3

    # Check Chunk 0
    path_0, offset_0 = chunks[0]
    assert path_0.exists()
    assert offset_0 == 0.0
    info_0 = validate_audio(path_0, min_duration_s=1.0)
    # Cut at ~51s (middle of first silence)
    assert 50.0 <= info_0.duration <= 52.0

    # Check Chunk 1
    path_1, offset_1 = chunks[1]
    assert path_1.exists()
    # Offset should be cut point (~51s) minus 2.0s overlap = ~49s
    assert 48.0 <= offset_1 <= 50.5
    info_1 = validate_audio(path_1, min_duration_s=1.0)
    assert 50.0 <= info_1.duration <= 60.0

    # Check Chunk 2
    path_2, offset_2 = chunks[2]
    assert path_2.exists()
    assert offset_2 > offset_1
    info_2 = validate_audio(path_2, min_duration_s=1.0)
    assert info_2.duration > 0
