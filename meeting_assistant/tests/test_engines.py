"""Unit tests for STT engines (mocks only – no network, no GPU).

Tests verify:
1. Failures become ``EngineResult(success=False, ...)``  (never an exception).
2. The ``offset_s`` parameter shifts every timestamp correctly.
3. Non-speech audio events are excluded from the spoken text.
4. Missing ElevenLabs API key returns a clear error message.
"""

from __future__ import annotations

import struct
import types
import wave as wave_mod
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stt.engines.base import STTEngine
from stt.engines.scribe import ScribeEngine
from stt.engines.whisper_local import WhisperLocalEngine
from stt.schema import EngineResult, Segment, Word


# ======================================================================
# Helpers
# ======================================================================


def _make_test_wav(path, duration_s=1.0, sample_rate=16000):
    """Generate a silent 16-bit mono WAV file for testing."""
    n_samples = int(sample_rate * duration_s)
    with wave_mod.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def _make_scribe_word(text, start, end, w_type="word", speaker_id=None):
    """Build a mock object mimicking ``SpeechToTextWordResponseModel``."""
    w = MagicMock()
    w.text = text
    w.start = start
    w.end = end
    w.type = w_type
    w.speaker_id = speaker_id
    return w


def _make_whisper_word(word, start, end, probability=0.95):
    """Build a mock faster-whisper word."""
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.probability = probability
    return w


def _make_whisper_segment(text, start, end, words):
    """Build a mock faster-whisper segment."""
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.words = words
    return seg


# ======================================================================
# ScribeEngine
# ======================================================================


class TestScribeEngine:
    """Tests for the ElevenLabs Scribe v2 engine."""

    def test_missing_api_key_returns_failed_result(self, tmp_path):
        """When no API key is set, the engine must return a clear error."""
        engine = ScribeEngine(api_key="")
        dummy = tmp_path / "dummy.wav"
        dummy.write_text("fake")
        result = engine.transcribe(str(dummy))

        assert result.success is False
        assert "API key" in result.error or "api key" in result.error.lower()
        assert result.engine_name == "elevenlabs"

    def test_missing_api_key_env_fallback(self, monkeypatch, tmp_path):
        """Even when env var is empty, the error is helpful."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        engine = ScribeEngine()
        dummy = tmp_path / "dummy.wav"
        dummy.write_text("fake")
        result = engine.transcribe(str(dummy))

        assert result.success is False
        assert "ELEVENLABS_API_KEY" in result.error

    def test_api_exception_becomes_failed_result(self):
        """Network / HTTP errors must not propagate to the caller."""
        engine = ScribeEngine(api_key="test-key-123")

        with patch.object(engine, "_call_api", side_effect=ConnectionError("timeout")):
            result = engine.transcribe("dummy.wav")

        assert result.success is False
        assert result.error_class == "ConnectionError"

    def test_non_speech_events_excluded_from_text(self):
        """Audio events like (laughter) must not appear in spoken text."""
        engine = ScribeEngine(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.text = "Hello (laughter) world"
        mock_response.words = [
            _make_scribe_word("Hello", 0.0, 0.5, "word"),
            _make_scribe_word(" ", 0.5, 0.5, "spacing"),
            _make_scribe_word("(laughter)", 0.5, 1.0, "audio_event"),
            _make_scribe_word(" ", 1.0, 1.0, "spacing"),
            _make_scribe_word("world", 1.0, 1.5, "word"),
        ]

        with patch.object(engine, "_call_api", return_value=mock_response):
            result = engine.transcribe("dummy.wav")

        assert result.success is True
        assert "(laughter)" not in result.text
        assert "Hello" in result.text
        assert "world" in result.text
        # Only actual words – no audio-event Word objects.
        assert len(result.words) == 2

    def test_diarization_speaker_id_propagated(self):
        """Speaker IDs from diarization should appear on Word objects."""
        engine = ScribeEngine(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.text = "Hi there"
        mock_response.words = [
            _make_scribe_word("Hi", 0.0, 0.3, "word", speaker_id="speaker_0"),
            _make_scribe_word(" ", 0.3, 0.3, "spacing"),
            _make_scribe_word("there", 0.3, 0.7, "word", speaker_id="speaker_1"),
        ]

        with patch.object(engine, "_call_api", return_value=mock_response):
            result = engine.transcribe("dummy.wav")

        assert result.words[0].speaker == "speaker_0"
        assert result.words[1].speaker == "speaker_1"

    def test_offset_applied_to_timestamps(self):
        """When offset_s is given, all timestamps must be shifted."""
        engine = ScribeEngine(api_key="test-key-123")
        offset = 60.0

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = [
            _make_scribe_word("test", 1.0, 2.0, "word"),
        ]

        with patch.object(engine, "_call_api", return_value=mock_response):
            result = engine.transcribe("dummy.wav", offset_s=offset)

        assert result.success is True
        assert result.words[0].start == pytest.approx(61.0)
        assert result.words[0].end == pytest.approx(62.0)
        assert result.segments[0].start == pytest.approx(61.0)
        assert result.segments[0].end == pytest.approx(62.0)

    def test_keyterms_trimmed_to_max(self):
        """Engine must silently trim keyterms to the API maximum (1000)."""
        big_list = [f"term_{i}" for i in range(1200)]
        engine = ScribeEngine(api_key="k", keyterms=big_list)
        assert len(engine._keyterms) == 1000


# ======================================================================
# WhisperLocalEngine
# ======================================================================


class TestWhisperLocalEngine:
    """Tests for the local faster-whisper engine."""

    def _make_engine_with_mock_model(self, segments, info=None):
        """Create a ``WhisperLocalEngine`` whose model is already mocked."""
        engine = WhisperLocalEngine(model_size="tiny", device="cpu", compute_type="int8")

        mock_model = MagicMock()
        mock_info = info or MagicMock()
        mock_model.transcribe.return_value = (iter(segments), mock_info)
        engine._model = mock_model
        return engine

    def test_basic_transcription(self, tmp_path):
        """Happy path: segments and words are mapped correctly."""
        wav_path = tmp_path / "test.wav"
        _make_test_wav(wav_path)

        words = [
            _make_whisper_word(" Hello", 0.0, 0.5, 0.99),
            _make_whisper_word(" world", 0.5, 1.0, 0.95),
        ]
        segments = [_make_whisper_segment(" Hello world", 0.0, 1.0, words)]

        engine = self._make_engine_with_mock_model(segments)
        result = engine.transcribe(str(wav_path))

        assert result.success is True
        assert result.engine_name == "whisper"
        assert "Hello" in result.text
        assert "world" in result.text
        assert len(result.words) == 2
        assert result.words[0].confidence == pytest.approx(0.99)

    def test_offset_applied(self, tmp_path):
        """Global offset must shift all whisper timestamps."""
        wav_path = tmp_path / "test.wav"
        _make_test_wav(wav_path)

        words = [_make_whisper_word(" hi", 0.0, 0.3, 0.9)]
        segments = [_make_whisper_segment(" hi", 0.0, 0.3, words)]

        engine = self._make_engine_with_mock_model(segments)
        result = engine.transcribe(str(wav_path), offset_s=100.0)

        assert result.words[0].start == pytest.approx(100.0)
        assert result.words[0].end == pytest.approx(100.3)

    def test_model_load_failure_becomes_failed_result(self):
        """If the model can't load, the caller gets a failed EngineResult."""
        engine = WhisperLocalEngine(model_size="tiny", device="cpu", compute_type="int8")

        with patch.object(
            engine,
            "_get_model",
            side_effect=RuntimeError("CUDA OOM"),
        ):
            result = engine.transcribe("test.wav")

        assert result.success is False
        assert "CUDA OOM" in result.error

    def test_empty_segments(self, tmp_path):
        """If whisper returns no segments, result should still be valid."""
        wav_path = tmp_path / "silence.wav"
        _make_test_wav(wav_path)

        engine = self._make_engine_with_mock_model([])
        result = engine.transcribe(str(wav_path))

        assert result.success is True
        assert result.text == ""
        assert result.words == []

    def test_model_cached_across_calls(self, tmp_path):
        """The WhisperModel should be loaded once and reused."""
        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        _make_test_wav(wav_a)
        _make_test_wav(wav_b)

        words = [_make_whisper_word(" ok", 0.0, 0.2, 0.8)]
        segments = [_make_whisper_segment(" ok", 0.0, 0.2, words)]

        engine = self._make_engine_with_mock_model(segments)
        mock_model = engine._model

        engine.transcribe(str(wav_a))
        # Re-set the iterator for the second call.
        mock_model.transcribe.return_value = (iter(segments), MagicMock())
        engine.transcribe(str(wav_b))

        # _get_model should return the same object, not re-instantiate.
        assert engine._model is mock_model


# ======================================================================
# Base STTEngine (offset logic)
# ======================================================================


class TestBaseOffset:
    """Verify the base class offset-shifting helper in isolation."""

    def test_apply_offset_shifts_words_and_segments(self):
        word = Word(text="hi", start=1.0, end=2.0)
        seg = Segment(text="hi", start=1.0, end=2.0, words=[word])
        result = EngineResult(
            engine_name="test",
            success=True,
            text="hi",
            words=[word],
            segments=[seg],
        )

        shifted = STTEngine._apply_offset(result, 10.0)

        assert shifted.words[0].start == pytest.approx(11.0)
        assert shifted.words[0].end == pytest.approx(12.0)
        assert shifted.segments[0].start == pytest.approx(11.0)
        assert shifted.segments[0].end == pytest.approx(12.0)
        assert shifted.segments[0].words[0].start == pytest.approx(11.0)

    def test_zero_offset_is_noop(self):
        word = Word(text="x", start=5.0, end=6.0)
        result = EngineResult(
            engine_name="test", success=True, text="x", words=[word]
        )
        # offset_s=0 should skip the shift entirely (optimisation path
        # in transcribe()), but _apply_offset itself still works.
        shifted = STTEngine._apply_offset(result, 0.0)
        assert shifted.words[0].start == pytest.approx(5.0)
