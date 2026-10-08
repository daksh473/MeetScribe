"""Local faster-whisper STT engine.

Uses the ``faster-whisper`` package (``pip install faster-whisper``),
which is a CTranslate2-based re-implementation of OpenAI Whisper.

Package references
------------------
* Package: ``faster-whisper`` (PyPI, SYSTRAN/faster-whisper on GitHub)
* Class:   ``faster_whisper.WhisperModel``
* Method:  ``model.transcribe(audio, ...)``
  Returns ``(segments_generator, TranscriptionInfo)``.
* Key parameters used in ``model.transcribe()``:
  - ``language="en"``
  - ``beam_size=5``
  - ``vad_filter=True``
  - ``condition_on_previous_text=False``
  - ``word_timestamps=True``
  - ``initial_prompt=<str>``
* ``WhisperModel(model_size, device=..., compute_type=...)``
* Each segment yielded has:
  - ``.text``, ``.start``, ``.end``
  - ``.words`` list, each with ``.word``, ``.start``, ``.end``,
    ``.probability``
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..schema import EngineResult, Segment, Word
from .base import STTEngine

logger = logging.getLogger(__name__)


def _resolve_device_and_compute(
    device: str, compute_type: str, model_size: str
) -> tuple:
    """Pick the best device / compute-type / model-size combination.

    Rules
    -----
    * ``device="auto"`` → try CUDA first, fall back to CPU.
    * On CUDA the default is ``float16`` with the requested model size.
    * On CPU the default is ``int8``, and if the user asked for a large
      model we silently downgrade to ``medium`` to avoid OOM / slowness.
    """
    import torch  # type: ignore[import-untyped]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "int8"

    # Downgrade large models on CPU to keep memory usage sane.
    if device == "cpu" and model_size.startswith("large"):
        logger.warning(
            "Downgrading Whisper model from %s to 'medium' for CPU inference.",
            model_size,
        )
        model_size = "medium"

    return device, compute_type, model_size


class WhisperLocalEngine(STTEngine):
    """Local faster-whisper engine.

    The underlying ``WhisperModel`` is loaded **lazily** on the first
    ``transcribe`` call and cached for subsequent calls.

    Parameters
    ----------
    model_size:
        Whisper model name (default from config or ``"large-v3"``).
    device:
        ``"auto"`` (default), ``"cuda"``, or ``"cpu"``.
    compute_type:
        ``"default"`` (auto-select), ``"float16"``, ``"int8"``, etc.
    initial_prompt:
        Optional prompt fed to the decoder as conditioning context.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self._cfg_model_size = model_size or os.environ.get(
            "WHISPER_MODEL_SIZE", "large-v3"
        )
        self._cfg_device = device or os.environ.get("WHISPER_DEVICE", "auto")
        self._cfg_compute_type = compute_type or os.environ.get(
            "WHISPER_COMPUTE_TYPE", "default"
        )
        self._initial_prompt = initial_prompt or ""
        self._model = None  # lazily loaded

    # ------------------------------------------------------------------
    # STTEngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "whisper"

    def _do_transcribe(self, wav_path: str) -> EngineResult:
        try:
            model = self._get_model()
        except (TypeError, ImportError) as e:
            # Library-level incompatibilities (e.g. PyAV metadata_errors mismatch)
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=f"{e.__class__.__name__}: {e}",
                error_class=e.__class__.__name__,
            )
        except Exception as e:
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=str(e),
                error_class=e.__class__.__name__,
            )

        try:
            import wave
            import numpy as np

            with wave.open(wav_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

                # Our pipeline ensures 16-bit PCM at 16 kHz
                if sample_width == 2:
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:
                    audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

            segments_gen, info = model.transcribe(
                audio_array,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=True,
                initial_prompt=self._initial_prompt or None,
            )

            # Consume the generator eagerly so we can build our result.
            all_words: List[Word] = []
            all_segments: List[Segment] = []

            for seg in segments_gen:
                seg_words: List[Word] = []
                for w in seg.words or []:
                    word = Word(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    )
                    seg_words.append(word)
                    all_words.append(word)

                all_segments.append(
                    Segment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        words=seg_words,
                    )
                )

            full_text = " ".join(s.text for s in all_segments)

            return EngineResult(
                engine_name=self.name,
                success=True,
                text=full_text,
                segments=all_segments,
                words=all_words,
            )
        except (TypeError, ImportError) as e:
            # Catch library-level incompatibilities during transcription
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=f"{e.__class__.__name__}: {e}",
                error_class=e.__class__.__name__,
            )
        except Exception as e:
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=str(e),
                error_class=e.__class__.__name__,
            )

    # ------------------------------------------------------------------
    # Model loading (lazy + cached)
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the cached ``WhisperModel``, loading it on first call."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        device = self._cfg_device
        if device == "auto":
            try:
                if torch.cuda.is_available():
                    _ = torch.zeros(1).cuda()
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        device, compute_type, model_size = _resolve_device_and_compute(
            device,
            self._cfg_compute_type,
            self._cfg_model_size,
        )

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )

        import time
        t0 = time.perf_counter()
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.environ.get("WHISPER_MODEL_PATH")
            )
            self.resolved_device = device
            self.resolved_model_size = model_size
        except Exception as e:
            logger.warning(f"Whisper load failed on {device}/{compute_type}/{model_size}: {e}. Retrying on cpu/int8/small.")
            try:
                self._model = WhisperModel(
                    "small", 
                    device="cpu", 
                    compute_type="int8",
                    download_root=os.environ.get("WHISPER_MODEL_PATH")
                )
                self.resolved_device = "cpu"
                self.resolved_model_size = "small"
            except Exception as e2:
                err_str = str(e2).lower()
                if "metadata_errors" in err_str:
                    raise RuntimeError("faster-whisper incompatible with Python 3.13 (tarfile metadata_errors).") from e2
                elif "network" in err_str or "connection" in err_str or "timeout" in err_str or "401" in err_str or "403" in err_str or "huggingface" in err_str or "download" in err_str or "max retries" in err_str:
                    raise RuntimeError("model download failed: check internet/proxy; or place the model in <path> and set WHISPER_MODEL_PATH") from e2
                raise e2
            
        load_time = time.perf_counter() - t0
        logger.info(f"Loaded faster-whisper model={self.resolved_model_size} device={self.resolved_device} compute_type={compute_type} in {load_time:.2f}s")
        return self._model
