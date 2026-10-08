import pytest
from stt.schema import Word, EngineResult, Segment
from stt.segmenter import build_segments
from stt.pipeline import _merge_chunk_results
from stt.consensus import build_consensus

def test_segmenter_basic():
    # 40 words over 60 seconds with pauses
    words = []
    t = 0.0
    for i in range(40):
        # pause at i=20
        if i == 20:
            t += 2.0
        words.append(Word(text=f"w{i}", start=t, end=t+0.5, speaker="spk1"))
        t += 0.5
        
    segments = build_segments(words)
    assert len(segments) > 1 # split by pause
    
    # Monotonic times check
    for i in range(1, len(segments)):
        assert segments[i].start >= segments[i-1].end
        
def test_segmenter_max_words():
    # 120 words run, no punctuation, no pauses
    words = []
    t = 0.0
    for i in range(120):
        words.append(Word(text=f"w{i}", start=t, end=t+0.1, speaker="spk1"))
        t += 0.1
        
    segments = build_segments(words)
    assert len(segments) > 1 # must be split by max words
    
def test_joined_text_equals_raw():
    words = [
        Word(text="Hello", start=0.0, end=0.5, speaker="A"),
        Word(text="world.", start=0.5, end=1.0, speaker="A"),
        Word(text="Next", start=2.0, end=2.5, speaker="A"),
        Word(text="segment.", start=2.5, end=3.0, speaker="B"),
    ]
    segments = build_segments(words)
    raw_text = " ".join([w.text for w in words])
    joined = " ".join([s.text for s in segments])
    assert joined == raw_text
    
def test_single_engine_path():
    words = [
        Word(text="Hello", start=0.0, end=0.5, speaker="A"),
        Word(text="world.", start=0.5, end=1.0, speaker="A"),
    ]
    res = EngineResult(engine_name="test", success=True, words=words)
    c_res = build_consensus([res])
    
    assert len(c_res.words) == 2
    segments = build_segments(c_res.words)
    assert len(segments) == 1
    assert segments[0].text == "Hello world."
