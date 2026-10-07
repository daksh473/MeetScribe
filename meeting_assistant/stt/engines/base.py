"""Abstract base class for all STT engines.

Every concrete engine must subclass STTEngine and implement ``_do_transcribe``.
The public ``transcribe`` method wraps it with offset-shifting and exception
safety: callers are guaranteed to never receive an exception for expected
failures (network, quota, bad audio, missing key).  Instead they get an
``EngineResult`` with ``success=False`` and a human-readable ``error`` string.
"""

from __future__ import annotations

import abc
import time
from typing import List

from ..schema import EngineResult, Segment, Word


class STTEngine(abc.ABC):
    """Base class for speech-to-text engines.

    Subclasses must set ``name`` and implement ``_do_transcribe``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, unique engine identifier (e.g. ``"elevenlabs"``, ``"whisper"``)."""
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, wav_path: str, offset_s: float = 0.0) -> EngineResult:
        """Transcribe *wav_path* and return an :class:`EngineResult`.

        Parameters
        ----------
        wav_path:
            Path to a WAV (or other ffmpeg-compatible) audio file.
        offset_s:
            Global timeline offset to add to every timestamp so that
            chunk results can be merged without re-computation.

        Returns
        -------
        EngineResult
            Always returned – never raises for expected failures.
        """
        t0 = time.perf_counter()
        try:
            result = self._do_transcribe(wav_path)
        except Exception as exc:  # noqa: BLE001 – intentional blanket catch
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                processing_time_s=time.perf_counter() - t0,
            )

        elapsed = time.perf_counter() - t0

        # Apply the global timeline offset to all timestamps.
        if offset_s:
            result = self._apply_offset(result, offset_s)

        result.processing_time_s = elapsed
        return result

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _do_transcribe(self, wav_path: str) -> EngineResult:
        """Engine-specific transcription logic.

        May raise – the caller (``transcribe``) will catch and wrap it.
        Timestamps should be relative to the *start of the chunk* (0-based).
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_offset(result: EngineResult, offset_s: float) -> EngineResult:
        """Shift every timestamp in *result* forward by *offset_s*."""

        def _shift_words(words: List[Word]) -> List[Word]:
            return [
                word.model_copy(
                    update={
                        "start": word.start + offset_s,
                        "end": word.end + offset_s,
                    }
                )
                for word in words
            ]

        def _shift_segments(segments: List[Segment]) -> List[Segment]:
            return [
                seg.model_copy(
                    update={
                        "start": seg.start + offset_s,
                        "end": seg.end + offset_s,
                        "words": _shift_words(seg.words),
                    }
                )
                for seg in segments
            ]

        return result.model_copy(
            update={
                "words": _shift_words(result.words),
                "segments": _shift_segments(result.segments),
            }
        )
