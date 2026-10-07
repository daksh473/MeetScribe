"""Tests for the Refiner module."""

from unittest.mock import MagicMock

import pytest

from llm.client import LLMClientError
from refine.refiner import refine, LLMRefinementResponse, LLMSpanResolution
from stt.schema import STTMetadata, STTResult, Segment, UncertainSpan, Word


def _make_stt_result(text: str, spans: list[UncertainSpan]) -> STTResult:
    # Fake words just so timestamps vaguely match
    words = []
    current_t = 0.0
    for w in text.split():
        words.append(Word(text=w, start=current_t, end=current_t+0.5, speaker="spk1"))
        current_t += 0.5
        
    seg = Segment(text=text, start=0.0, end=current_t, words=words)
    return STTResult(
        raw_text=text,
        segments=[seg],
        uncertain_spans=spans,
        metadata=STTMetadata()
    )


def test_refine_llm_unavailable_succeeds_with_warning():
    span = UncertainSpan(
        id="s1", chosen_text="there",
        start=0.0, end=1.0, 
        context_before="hello", context_after="world",
        alternatives={"test": "there"}, category="HIGH"
    )
    stt_res = _make_stt_result("hello there world", [span])
    
    mock_client = MagicMock()
    mock_client.complete_json.side_effect = Exception("API Down")
    
    res = refine(stt_res, mock_client)
    
    assert res.refined_text == "hello there world"
    assert len(res.edits) == 0
    assert any("llm_refinement_failed" in f for f in res.flags)


def test_refine_edit_budget_exceeded():
    span = UncertainSpan(
        id="s2", chosen_text="hi",
        start=0.0, end=0.5, 
        context_before="", context_after="world",
        alternatives={"a": "hi", "b": "hello"}, category="HIGH"
    )
    stt_res = _make_stt_result("hi world", [span])
    
    mock_client = MagicMock()
    mock_resp = LLMRefinementResponse(resolutions=[
        LLMSpanResolution(span_id="span_0", choice="ENGINE_B", reason="test")
    ])
    mock_client.complete_json.return_value = mock_resp
    
    # 1 word changed out of 2 total words -> 50% edit ratio. Max is 3%.
    res = refine(stt_res, mock_client, max_edit_ratio=0.03)
    
    assert "edit_budget_exceeded" in res.flags
    assert len(res.edits) == 0
    assert len(res.rejected_edits) == 1
    assert "budget exceeded" in res.rejected_edits[0].reason.lower()
    assert res.refined_text == "hi world"


def test_refine_unresolved_returns_unchanged_with_flag():
    span = UncertainSpan(
        id="s3", chosen_text="hi",
        start=0.0, end=0.5, 
        context_before="", context_after="world",
        alternatives={"a": "hi", "b": "hello"}, category="HIGH"
    )
    stt_res = _make_stt_result("hi world", [span])
    
    mock_client = MagicMock()
    mock_resp = LLMRefinementResponse(resolutions=[
        LLMSpanResolution(span_id="span_0", choice="UNRESOLVED", reason="Too garbled")
    ])
    mock_client.complete_json.return_value = mock_resp
    
    res = refine(stt_res, mock_client)
    
    assert "unresolved_span_span_0" in res.flags
    assert len(res.edits) == 0
    assert res.refined_text == "hi world"


def test_refine_valid_edit_applied():
    # Make a longer text so budget isn't exceeded (need > 34 words for 1 word edit to be < 3%)
    text = "this is a very long sentence just to make sure we do not exceed the edit budget threshold hi world. " * 4
    span = UncertainSpan(
        id="s4", chosen_text="hi",
        start=9.0, end=9.5, 
        context_before="threshold", context_after="world",
        alternatives={"ENGINE_A": "hi", "ENGINE_B": "hello"}, category="HIGH"
    )
    stt_res = _make_stt_result(text, [span])
    
    mock_client = MagicMock()
    mock_resp = LLMRefinementResponse(resolutions=[
        LLMSpanResolution(span_id="span_0", choice="ENGINE_B", reason="test")
    ])
    mock_client.complete_json.return_value = mock_resp
    
    res = refine(stt_res, mock_client)
    
    assert len(res.edits) == 1
    assert res.edits[0].choice == "ENGINE_B"
    assert res.edits[0].final_text == "hello"
    
    # Verify text was replaced
    expected_text = text.replace("hi world", "hello world", 1)
    assert res.refined_text == expected_text
