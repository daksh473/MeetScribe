"""Audio normalization and silence-based chunking using ffmpeg."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

from config import settings
from .errors import CorruptAudioError, FileNotFoundAudioError
from .validator import validate_audio


class ChunkInfo(NamedTuple):
    """Metadata for an individual audio chunk."""

    path: Path
    offset_s: float
    duration_s: float


def normalize_audio(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    enable_denoising: Optional[bool] = None,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    """Normalize input audio to 16 kHz mono WAV with loudness normalization.

    - Video containers: audio is extracted (-vn).
    - Audio format: 16 kHz, single channel (mono), 16-bit PCM WAV.
    - Normalization: loudnorm filter (I=-16, TP=-1.5, LRA=11).
    - Denoising: optional (default disabled via config).
    """
    input_file = Path(in_path)
    output_file = Path(out_path)

    if not input_file.exists():
        raise FileNotFoundAudioError(str(input_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    denoise = enable_denoising if enable_denoising is not None else settings.enable_denoising
    ff_bin = ffmpeg_path or settings.ffmpeg_path

    # Construct audio filter chain
    audio_filters: List[str] = []
    if denoise:
        # Subtle highpass/lowpass to filter room rumble and high hiss
        audio_filters.append("highpass=f=80,lowpass=f=7500")

    audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    filter_arg = ",".join(audio_filters)

    cmd = [
        ff_bin,
        "-y",
        "-i",
        str(input_file.resolve()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filter_arg,
        str(output_file.resolve()),
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
            message="ffmpeg binary not found. Please install ffmpeg.",
            details=str(e),
        ) from e

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise CorruptAudioError(
            message="Failed to normalize audio file.",
            details=f"ffmpeg exit code {proc.returncode}: {err}",
        )

    return output_file.resolve()


def detect_silences(
    wav_path: Union[str, Path],
    noise_thresh_db: float = -30.0,
    min_silence_duration_s: float = 0.3,
    ffmpeg_path: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """Detect silence intervals in audio using ffmpeg silencedetect filter.

    Returns a list of (silence_start_s, silence_end_s) intervals.
    """
    file_path = Path(wav_path)
    ff_bin = ffmpeg_path or settings.ffmpeg_path

    cmd = [
        ff_bin,
        "-i",
        str(file_path.resolve()),
        "-af",
        f"silencedetect=noise={noise_thresh_db}dB:d={min_silence_duration_s}",
        "-f",
        "null",
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        raise CorruptAudioError(
            message="Failed to execute silence detection on audio.",
            details=str(e),
        ) from e

    # Parse silence intervals from stderr
    silences: List[Tuple[float, float]] = []
    current_start: Optional[float] = None

    for line in proc.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            try:
                current_start = float(start_match.group(1))
            except ValueError:
                current_start = None

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match:
            try:
                silence_end = float(end_match.group(1))
                silence_start = current_start if current_start is not None else max(0.0, silence_end - min_silence_duration_s)
                silences.append((silence_start, silence_end))
            except ValueError:
                pass
            current_start = None

    return silences


def _extract_chunk(
    in_path: Path,
    start_s: float,
    end_s: float,
    out_path: Path,
    ffmpeg_path: Optional[str] = None,
) -> None:
    """Extract an audio segment between start_s and end_s into out_path."""
    ff_bin = ffmpeg_path or settings.ffmpeg_path
    duration_s = max(0.01, end_s - start_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ff_bin,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(in_path.resolve()),
        "-c",
        "copy",
        str(out_path.resolve()),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise CorruptAudioError(
            message="Failed to extract audio chunk.",
            details=f"ffmpeg exit code {proc.returncode}: {err}",
        )


def split_on_silence(
    wav_path: Union[str, Path],
    max_chunk_s: Optional[float] = None,
    overlap_s: Optional[float] = None,
    out_dir: Optional[Union[str, Path]] = None,
    ffmpeg_path: Optional[str] = None,
) -> List[Tuple[Path, float]]:
    """Split audio at silence points, returning list of (chunk_path, global_offset_s).

    If audio is shorter than max_chunk_s, returns it as a single chunk without cutting.
    Never cuts mid-word if silence points are available near the chunk boundary.
    """
    file_path = Path(wav_path)
    if not file_path.exists():
        raise FileNotFoundAudioError(str(file_path))

    max_chunk = max_chunk_s if max_chunk_s is not None else settings.chunk_max_s
    overlap = overlap_s if overlap_s is not None else settings.chunk_overlap_s

    # Inspect file duration
    info = validate_audio(file_path, min_duration_s=0.01, max_duration_s=float("inf"))
    total_duration = info.duration

    # If shorter than max chunk duration, return as single chunk
    if total_duration <= max_chunk:
        return [(file_path.resolve(), 0.0)]

    # Prepare chunks output folder
    if out_dir is None:
        chunks_folder = file_path.parent / f"{file_path.stem}_chunks"
    else:
        chunks_folder = Path(out_dir)
    chunks_folder.mkdir(parents=True, exist_ok=True)

    # Detect all silences in the file
    silences = detect_silences(file_path, ffmpeg_path=ffmpeg_path)

    chunks: List[Tuple[Path, float]] = []
    current_start = 0.0
    chunk_idx = 0

    while current_start < total_duration:
        remaining = total_duration - current_start
        if remaining <= max_chunk:
            chunk_end = total_duration
            chunk_file = chunks_folder / f"chunk_{chunk_idx:04d}.wav"
            _extract_chunk(file_path, current_start, chunk_end, chunk_file, ffmpeg_path)
            chunks.append((chunk_file.resolve(), current_start))
            break

        target_cut = current_start + max_chunk
        search_min = current_start + (max_chunk * 0.70)
        search_max = target_cut

        # Look for silence midpoints within [search_min, search_max]
        candidates = [
            (s_start, s_end)
            for s_start, s_end in silences
            if search_min <= ((s_start + s_end) / 2.0) <= search_max
        ]

        if candidates:
            # Pick silence with largest duration to ensure safest split
            best_sil = max(candidates, key=lambda s: (s[1] - s[0]))
            split_point = (best_sil[0] + best_sil[1]) / 2.0
        else:
            # Wider search in [current_start + max_chunk * 0.5, target_cut]
            wider = [
                (s_start, s_end)
                for s_start, s_end in silences
                if (current_start + (max_chunk * 0.50)) <= ((s_start + s_end) / 2.0) <= target_cut
            ]
            if wider:
                best_sil = max(wider, key=lambda s: (s[1] - s[0]))
                split_point = (best_sil[0] + best_sil[1]) / 2.0
            else:
                # No silence found, fallback to hard cut at max_chunk
                split_point = target_cut

        chunk_file = chunks_folder / f"chunk_{chunk_idx:04d}.wav"
        _extract_chunk(file_path, current_start, split_point, chunk_file, ffmpeg_path)
        chunks.append((chunk_file.resolve(), current_start))

        # Determine next start position with overlap
        next_start = max(current_start + 1.0, split_point - overlap)
        if next_start >= total_duration - 0.5:
            break

        current_start = next_start
        chunk_idx += 1

    return chunks
