"""Tests for STT consensus alignment and uncertainty mapping."""

import pytest

from stt.consensus import build_consensus
from stt.schema import EngineResult, Segment, Word
from stt.uncertainty import generate_spans


def _make_words(text: str, start_offset: float = 0.0) -> list[Word]:
    """Helper to quickly make a list of Word objects from a string."""
    words = []
    current_time = start_offset
    for w in text.split():
        words.append(Word(text=w, start=current_time, end=current_time + 0.5, confidence=0.9, speaker="spk1"))
        current_time += 0.5
    return words


def _make_engine_result(name: str, text: str, success: bool = True, start_offset: float = 0.0) -> EngineResult:
    """Helper to make an EngineResult with sequential word timestamps."""
    if not success:
        return EngineResult(engine_name=name, success=False, error="Failed")
    
    words = _make_words(text, start_offset)
    seg = Segment(text=text, start=words[0].start if words else 0.0, end=words[-1].end if words else 0.0, words=words)
    return EngineResult(engine_name=name, success=True, text=text, words=words, segments=[seg])


def test_identical_outputs_zero_spans():
    res1 = _make_engine_result("elevenlabs", "hello world this is a test")
    res2 = _make_engine_result("whisper", "hello world this is a test")
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 0
    assert " ".join(w.text for w in consensus.words) == "hello world this is a test"


def test_negation_critical_span():
    # "we will ship friday" vs "we will not ship friday"
    res1 = _make_engine_result("elevenlabs", "we will ship friday")
    res2 = _make_engine_result("whisper", "we will not ship friday")
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 1
    span = spans[0]
    assert span.category == "CRITICAL"
    assert "not" in span.alternatives["whisper"].lower()


def test_percentage_critical_span():
    # "fifteen percent" vs "fifty percent"
    res1 = _make_engine_result("elevenlabs", "fifteen percent")
    res2 = _make_engine_result("whisper", "fifty percent")
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 1
    assert spans[0].category == "CRITICAL"


def test_number_normalization_not_disputed():
    # "fifteen" vs "15" -> NOT disputed
    res1 = _make_engine_result("elevenlabs", "I have fifteen apples")
    res2 = _make_engine_result("whisper", "I have 15 apples")
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 0


def test_proper_noun_high_span():
    # "kubernetes" vs "cooper nettys"
    res1 = _make_engine_result("elevenlabs", "we use kubernetes here")
    res2 = _make_engine_result("whisper", "we use cooper nettys here")
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 1
    assert spans[0].category == "HIGH"


def test_one_engine_failed():
    res1 = _make_engine_result("elevenlabs", "hello", success=True)
    res2 = _make_engine_result("whisper", "", success=False)
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 0
def test_two_whisper_engines_consensus():
    """Test consensus between two Whisper-like engines (e.g. local vs Groq) with realistic differences."""
    # Engine 1 (local whisper): "We have 15 users, but John didn't like it."
    # Engine 2 (groq whisper): "We have fifteen users, but Jon did not like it."
    
    words1 = [
        Word(text="We", start=0.0, end=0.5, confidence=0.9),
        Word(text="have", start=0.5, end=1.0, confidence=0.9),
        Word(text="15", start=1.0, end=1.5, confidence=0.9),
        Word(text="users,", start=1.5, end=2.0, confidence=0.9),
        Word(text="but", start=2.0, end=2.5, confidence=0.9),
        Word(text="John", start=2.5, end=3.0, confidence=0.9),
        Word(text="didn't", start=3.0, end=3.5, confidence=0.9),
        Word(text="like", start=3.5, end=4.0, confidence=0.9),
        Word(text="it.", start=4.0, end=4.5, confidence=0.9),
    ]
    
    words2 = [
        Word(text="We", start=0.0, end=0.5, confidence=0.95),
        Word(text="have", start=0.5, end=1.0, confidence=0.95),
        Word(text="fifteen", start=1.0, end=1.5, confidence=0.95),
        Word(text="users,", start=1.5, end=2.0, confidence=0.95),
        Word(text="but", start=2.0, end=2.5, confidence=0.95),
        Word(text="Jon", start=2.5, end=3.0, confidence=0.95),
        Word(text="did", start=3.0, end=3.25, confidence=0.95),
        Word(text="not", start=3.25, end=3.5, confidence=0.95),
        Word(text="like", start=3.5, end=4.0, confidence=0.95),
        Word(text="it.", start=4.0, end=4.5, confidence=0.95),
    ]
    
    res1 = EngineResult(engine_name="whisper", success=True, words=words1)
    res2 = EngineResult(engine_name="groq_whisper", success=True, words=words2)
    
    consensus = build_consensus([res1, res2])
    
    # 15 vs fifteen should be normalized to the same token, not disputed
    # John vs Jon -> disputed
    # didn't vs did not -> normalization expands didn't to did not, so not disputed
    
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) >= 1
    
    # Verify John/Jon span is present
    john_span = next(s for s in spans if any("John" in t for t in list(s.alternatives.values()) + [s.chosen_text]))
    texts = list(john_span.alternatives.values()) + [john_span.chosen_text]
    assert "John" in texts or "John didn't" in texts
    assert "Jon" in texts or "Jon did not" in texts
    assert len(consensus.words) > 0


def test_timestamps_offset():
    # timestamps offset between engines by a few hundred ms still align correctly
    # res1: words at 0.0, 0.5, 1.0
    # res2: words at 0.2, 0.7, 1.2
    # Even if texts differ, time overlap should align them.
    res1 = _make_engine_result("elevenlabs", "a b c", start_offset=0.0)
    res2 = _make_engine_result("whisper", "a x c", start_offset=0.2)
    
    consensus = build_consensus([res1, res2])
    spans = generate_spans(consensus.disputed_slots, [res1, res2])
    
    assert len(spans) == 1  # "b" vs "x"
    assert " ".join(w.text for w in consensus.words) == "a b c"
