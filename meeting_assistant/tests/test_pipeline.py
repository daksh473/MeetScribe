"""Tests for STT pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from stt.errors import NoSpeechError, STTError
from stt.pipeline import _merge_chunk_results, run_stt
from stt.schema import EngineResult, Segment, Word


def _make_word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_merge_chunk_results_overlap_deduplication():
    # Chunk 1: "hello" (0-1), "world" (1-2)
    chunk1 = EngineResult(
        engine_name="test",
        success=True,
        words=[_make_word("hello", 0.0, 1.0), _make_word("world", 1.0, 2.0)],
    )
    # Chunk 2: "world" (1-2) - overlap duplicate, "test" (2-3)
    chunk2 = EngineResult(
        engine_name="test",
        success=True,
        words=[_make_word("world", 1.0, 2.0), _make_word("test", 2.0, 3.0)],
    )
    
    merged = _merge_chunk_results([chunk1, chunk2])
    
    assert merged.success is True
    # "world" from chunk 2 should be dropped because its start (1.0) is not >= last_end (2.0) - 0.05
    assert [w.text for w in merged.words] == ["hello", "world", "test"]
    assert merged.text == "hello world test"


@patch("stt.pipeline.validate_audio")
@patch("stt.pipeline.normalize_audio")
@patch("stt.pipeline.split_on_silence")
@patch("stt.pipeline._get_engines")
def test_pipeline_end_to_end_one_engine_fails(mock_get_engines, mock_split, mock_norm, mock_val):
    # Setup mocks
    mock_val.return_value = MagicMock(duration=60.0)
    
    # Mock chunks
    mock_split.return_value = [("chunk1.wav", 0.0)]
    
    # Mock engines
    good_engine = MagicMock()
    good_engine.name = "whisper"
    good_engine.transcribe.return_value = EngineResult(
        engine_name="whisper",
        success=True,
        text="hello",
        words=[_make_word("hello", 0.0, 1.0)]
    )
    
    bad_engine = MagicMock()
    bad_engine.name = "elevenlabs"
    bad_engine.transcribe.side_effect = Exception("API Down")
    
    mock_get_engines.return_value = [bad_engine, good_engine]
    
    # Run
    res = run_stt("dummy.wav")
    
    assert res.raw_text == "hello"
    assert "single_engine" in res.flags
    assert res.metadata.fallback_used is True
    assert "whisper" in res.metadata.engines_used
    assert "elevenlabs" not in res.metadata.engines_used


@patch("stt.pipeline.validate_audio")
@patch("stt.pipeline.normalize_audio")
@patch("stt.pipeline.split_on_silence")
@patch("stt.pipeline._get_engines")
def test_pipeline_all_engines_fail(mock_get_engines, mock_split, mock_norm, mock_val):
    mock_val.return_value = MagicMock(duration=60.0)
    
    mock_split.return_value = [("chunk1.wav", 0.0)]
    
    bad_engine = MagicMock()
    bad_engine.name = "whisper"
    bad_engine.transcribe.side_effect = Exception("Crash")
    
    mock_get_engines.return_value = [bad_engine]
    
    with pytest.raises(STTError, match="All STT engines failed"):
        run_stt("dummy.wav")


@patch("stt.pipeline.validate_audio")
@patch("stt.pipeline.normalize_audio")
@patch("stt.pipeline.split_on_silence")
@patch("stt.pipeline._get_engines")
def test_pipeline_no_speech_error(mock_get_engines, mock_split, mock_norm, mock_val):
    mock_val.return_value = MagicMock(duration=60.0)
    
    mock_split.return_value = [("chunk1.wav", 0.0)]
    
    # Engine succeeds but returns empty text
    empty_engine = MagicMock()
    empty_engine.name = "whisper"
    empty_engine.transcribe.return_value = EngineResult(
        engine_name="whisper",
        success=True,
        text="",
        words=[]
    )
    
    mock_get_engines.return_value = [empty_engine]
    
    with pytest.raises(NoSpeechError):
        run_stt("dummy.wav")
