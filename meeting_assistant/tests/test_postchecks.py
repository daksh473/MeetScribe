"""Tests for STT postchecks."""

from stt.postchecks import run_postchecks
from stt.schema import STTMetadata, STTResult


def _make_result(text: str, duration: float = 60.0) -> STTResult:
    return STTResult(
        raw_text=text,
        metadata=STTMetadata(audio_duration_s=duration)
    )


def test_no_flags_for_good_text():
    res = _make_result("This is a completely normal sentence with good word rate.", duration=10.0)
    run_postchecks(res)
    assert not res.flags


def test_hallucination_flag():
    res = _make_result("Thanks for watching, please subscribe to my channel.")
    run_postchecks(res)
    assert "possible_hallucination" in res.flags


def test_repeated_phrases_flag():
    res = _make_result("I think that I think that I think that we should go.")
    run_postchecks(res)
    assert "repeated_phrases" in res.flags


def test_low_wpm_flag():
    # 5 words in 60 seconds = 5 WPM
    res = _make_result("This is a short transcript.", duration=60.0)
    run_postchecks(res)
    assert "low_wpm" in res.flags


def test_non_english_flag():
    res = _make_result("こんにちは 世界 이것은 테스트입니다")
    run_postchecks(res)
    assert "non_english_suspected" in res.flags
