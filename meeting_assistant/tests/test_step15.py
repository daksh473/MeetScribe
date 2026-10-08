"""Step 15 tests: engine error handling, WAV loading, error summary, single-engine fallback.

No network, no GPU, no API keys required.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stt.engines.whisper_local import WhisperLocalEngine
from stt.errors import STTError
from stt.pipeline import run_stt
from stt.schema import EngineResult, Segment, Word


# ======================================================================
# 1. TypeError from PyAV metadata_errors → failed EngineResult
# ======================================================================


class TestWhisperTypeErrorHandling:
    """Whisper engine must catch TypeError/ImportError and return failed EngineResult."""

    def test_model_load_typeerror_returns_failed_result(self):
        """TypeError during model loading (e.g. av metadata_errors) → failed EngineResult."""
        engine = WhisperLocalEngine(model_size="tiny", device="cpu", compute_type="int8")

        with patch.object(
            engine,
            "_get_model",
            side_effect=TypeError(
                "open() got an unexpected keyword argument 'metadata_errors'"
            ),
        ):
            result = engine.transcribe("test.wav")

        assert result.success is False
        assert result.error_class == "TypeError"
        assert "metadata_errors" in result.error
        assert result.engine_name == "whisper"

    def test_import_error_returns_failed_result(self):
        """ImportError during model loading → failed EngineResult."""
        engine = WhisperLocalEngine(model_size="tiny", device="cpu", compute_type="int8")

        with patch.object(
            engine,
            "_get_model",
            side_effect=ImportError("No module named 'ctranslate2'"),
        ):
            result = engine.transcribe("test.wav")

        assert result.success is False
        assert result.error_class == "ImportError"
        assert "ctranslate2" in result.error

    def test_typeerror_during_transcription_returns_failed_result(self, tmp_path):
        """TypeError raised during transcription (not model loading) → failed EngineResult."""
        engine = WhisperLocalEngine(model_size="tiny", device="cpu", compute_type="int8")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = TypeError(
            "open() got an unexpected keyword argument 'metadata_errors'"
        )
        engine._model = mock_model

        # Create a valid WAV so the wav loading succeeds but model.transcribe fails
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, duration_s=0.5)

        result = engine.transcribe(str(wav_path))

        assert result.success is False
        assert result.error_class == "TypeError"
        assert "metadata_errors" in result.error


# ======================================================================
# 2. Loading 16 kHz mono WAV into numpy array
# ======================================================================


def _write_test_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000):
    """Generate a silent 16-bit mono WAV file."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


class TestWavNumpyLoading:
    """Verify the WAV → numpy loading path used by whisper_local.py."""

    def test_wav_loads_correct_length_and_dtype(self, tmp_path):
        """A 1s mono 16 kHz WAV → 16000-sample float32 array."""
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, duration_s=1.0, sample_rate=16000)

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        assert n_channels == 1
        assert sample_width == 2

        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        assert audio_array.dtype == np.float32
        assert len(audio_array) == 16000

    def test_wav_loads_correct_range(self, tmp_path):
        """Values should be in [-1.0, 1.0] range."""
        wav_path = tmp_path / "test_range.wav"
        n_samples = 100
        # Write max-amplitude samples
        samples = [32767] * (n_samples // 2) + [-32768] * (n_samples // 2)
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack(f"<{n_samples}h", *samples))

        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        assert audio_array.max() <= 1.0
        assert audio_array.min() >= -1.0

    def test_stereo_wav_downmixed_to_mono(self, tmp_path):
        """Stereo WAV should be downmixed to mono."""
        wav_path = tmp_path / "stereo.wav"
        n_samples = 160  # per channel
        # Left channel: 100, Right channel: -100 → mean = 0
        samples = []
        for _ in range(n_samples):
            samples.extend([100, -100])  # interleaved L, R

        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack(f"<{n_samples * 2}h", *samples))

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

        assert len(audio_array) == n_samples
        # Mean of 100 and -100 should be ~0
        assert abs(audio_array.mean()) < 0.01


# ======================================================================
# 3. Combined error message lists both engines
# ======================================================================


class TestCombinedErrorMessage:
    """When ALL engines fail, the error must list EVERY engine with its reason."""

    @patch("stt.pipeline.validate_audio")
    @patch("stt.pipeline.normalize_audio")
    @patch("stt.pipeline.split_on_silence")
    @patch("stt.pipeline._get_engines")
    def test_all_fail_error_lists_both_engines(
        self, mock_get_engines, mock_split, mock_norm, mock_val
    ):
        """Error message must contain both engine names and their distinct reasons."""
        mock_val.return_value = MagicMock(duration=60.0)
        mock_split.return_value = [("chunk1.wav", 0.0)]

        whisper_engine = MagicMock()
        whisper_engine.name = "whisper"
        whisper_engine.transcribe.return_value = EngineResult(
            engine_name="whisper",
            success=False,
            error="audio decoding library mismatch",
            error_class="TypeError",
        )

        elevenlabs_engine = MagicMock()
        elevenlabs_engine.name = "elevenlabs"
        elevenlabs_engine.transcribe.return_value = EngineResult(
            engine_name="elevenlabs",
            success=False,
            error="API key rejected (401)",
            error_class="ApiError",
        )

        mock_get_engines.return_value = [whisper_engine, elevenlabs_engine]

        with pytest.raises(STTError) as exc_info:
            run_stt("dummy.wav")

        error_msg = str(exc_info.value)
        # Both engine names must appear
        assert "whisper" in error_msg
        assert "elevenlabs" in error_msg
        # Both reasons must appear
        assert "audio decoding library mismatch" in error_msg
        assert "API key rejected (401)" in error_msg

    @patch("stt.pipeline.validate_audio")
    @patch("stt.pipeline.normalize_audio")
    @patch("stt.pipeline.split_on_silence")
    @patch("stt.pipeline._get_engines")
    def test_all_fail_error_includes_error_class(
        self, mock_get_engines, mock_split, mock_norm, mock_val
    ):
        """Error should include the error class when available."""
        mock_val.return_value = MagicMock(duration=60.0)
        mock_split.return_value = [("chunk1.wav", 0.0)]

        whisper_engine = MagicMock()
        whisper_engine.name = "whisper"
        whisper_engine.transcribe.return_value = EngineResult(
            engine_name="whisper",
            success=False,
            error="open() got an unexpected keyword argument 'metadata_errors'",
            error_class="TypeError",
        )

        mock_get_engines.return_value = [whisper_engine]

        with pytest.raises(STTError) as exc_info:
            run_stt("dummy.wav")

        error_msg = str(exc_info.value)
        assert "TypeError" in error_msg
        assert "metadata_errors" in error_msg


# ======================================================================
# 4. One engine failing → pipeline continues with single-engine warning
# ======================================================================


class TestSingleEngineFallback:
    """When one engine fails, pipeline must continue with the other + emit a warning."""

    @patch("stt.pipeline.validate_audio")
    @patch("stt.pipeline.normalize_audio")
    @patch("stt.pipeline.split_on_silence")
    @patch("stt.pipeline._get_engines")
    def test_one_engine_fails_pipeline_continues(
        self, mock_get_engines, mock_split, mock_norm, mock_val
    ):
        """If elevenlabs fails but whisper succeeds, pipeline produces a result."""
        mock_val.return_value = MagicMock(duration=60.0)
        mock_split.return_value = [("chunk1.wav", 0.0)]

        good_engine = MagicMock()
        good_engine.name = "whisper"
        good_engine.transcribe.return_value = EngineResult(
            engine_name="whisper",
            success=True,
            text="hello world",
            words=[
                Word(text="hello", start=0.0, end=0.5),
                Word(text="world", start=0.5, end=1.0),
            ],
        )

        bad_engine = MagicMock()
        bad_engine.name = "elevenlabs"
        bad_engine.transcribe.return_value = EngineResult(
            engine_name="elevenlabs",
            success=False,
            error="API key rejected (401)",
            error_class="ApiError",
        )

        mock_get_engines.return_value = [bad_engine, good_engine]

        result = run_stt("dummy.wav")

        # Pipeline should succeed with single engine
        assert result.raw_text == "hello world"
        assert "whisper" in result.metadata.engines_used
        assert "elevenlabs" not in result.metadata.engines_used
        assert result.metadata.fallback_used is True
        assert "single_engine" in result.flags

    @patch("stt.pipeline.validate_audio")
    @patch("stt.pipeline.normalize_audio")
    @patch("stt.pipeline.split_on_silence")
    @patch("stt.pipeline._get_engines")
    def test_single_engine_warning_names_failed_engine(
        self, mock_get_engines, mock_split, mock_norm, mock_val
    ):
        """The warning must name the failed engine and its specific reason."""
        mock_val.return_value = MagicMock(duration=60.0)
        mock_split.return_value = [("chunk1.wav", 0.0)]

        good_engine = MagicMock()
        good_engine.name = "whisper"
        good_engine.transcribe.return_value = EngineResult(
            engine_name="whisper",
            success=True,
            text="test",
            words=[Word(text="test", start=0.0, end=0.5)],
        )

        bad_engine = MagicMock()
        bad_engine.name = "elevenlabs"
        bad_engine.transcribe.return_value = EngineResult(
            engine_name="elevenlabs",
            success=False,
            error="API key rejected (401)",
            error_class="ApiError",
        )

        mock_get_engines.return_value = [bad_engine, good_engine]

        result = run_stt("dummy.wav")

        # Check warning contains both engine name and reason
        assert len(result.metadata.warnings) >= 1
        warning_text = " ".join(result.metadata.warnings)
        assert "elevenlabs" in warning_text
        assert "API key rejected (401)" in warning_text
        assert "Single-engine mode" in warning_text

    @patch("stt.pipeline.validate_audio")
    @patch("stt.pipeline.normalize_audio")
    @patch("stt.pipeline.split_on_silence")
    @patch("stt.pipeline._get_engines")
    def test_whisper_fails_elevenlabs_succeeds(
        self, mock_get_engines, mock_split, mock_norm, mock_val
    ):
        """If whisper fails but elevenlabs succeeds, pipeline still works."""
        mock_val.return_value = MagicMock(duration=60.0)
        mock_split.return_value = [("chunk1.wav", 0.0)]

        bad_engine = MagicMock()
        bad_engine.name = "whisper"
        bad_engine.transcribe.return_value = EngineResult(
            engine_name="whisper",
            success=False,
            error="TypeError: open() got an unexpected keyword argument 'metadata_errors'",
            error_class="TypeError",
        )

        good_engine = MagicMock()
        good_engine.name = "elevenlabs"
        good_engine.transcribe.return_value = EngineResult(
            engine_name="elevenlabs",
            success=True,
            text="hello",
            words=[Word(text="hello", start=0.0, end=0.5)],
        )

        mock_get_engines.return_value = [bad_engine, good_engine]

        result = run_stt("dummy.wav")

        assert result.raw_text == "hello"
        assert "elevenlabs" in result.metadata.engines_used
        assert result.metadata.fallback_used is True
        # Warning should name whisper and its reason
        warning_text = " ".join(result.metadata.warnings)
        assert "whisper" in warning_text
        assert "TypeError" in warning_text
