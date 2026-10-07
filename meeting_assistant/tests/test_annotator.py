"""Tests for annotator and segmenter (Pass A)."""

from unittest.mock import MagicMock

from llm.schemas import Utterance, DialogueAct
from minutes.annotator import annotate_segment, DialogueActList
from minutes.segmenter import segment_utterances


def test_segmenter():
    utts = [Utterance(id=f"u{i}", speaker="s1", start=0, end=1, text="test") for i in range(10)]
    
    segs = segment_utterances(utts, max_size=5, overlap=2)
    assert len(segs) > 1
    
    # First seg is size 5
    assert len(segs[0]) == 5
    # Next seg starts at end_idx (5) - overlap (2) = index 3
    assert segs[1][0].id == "u3"


def test_annotator_fake_client():
    utts = [Utterance(id="u1", speaker="s1", start=0, end=1, text="We should do X")]
    
    mock_client = MagicMock()
    mock_resp = DialogueActList(acts=[
        DialogueAct(utterance_id="u1", label="PROPOSAL", quote="We should do X")
    ])
    mock_client.complete_json.return_value = mock_resp
    
    acts = annotate_segment(utts, mock_client)
    
    assert len(acts) == 1
    assert acts[0].label == "PROPOSAL"
    assert acts[0].quote == "We should do X"
