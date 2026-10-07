"""Tests for writer (Pass D)."""

from unittest.mock import MagicMock
from llm.schemas import Decision, MeetingMetadata
from minutes.writer import write_minutes, WriterResponse, MinutesSection


def test_writer_citing_unknown_id_triggers_regeneration_then_template_fallback():
    d = Decision(id="D1", text="Test", status="PROPOSED", evidence=[], low_confidence=False)
    meta = MeetingMetadata(models_used=[], call_log_summary={}, warnings=[])
    
    mock_client = MagicMock()
    # Always return a response citing a fake ID 'D99'
    mock_resp = WriterResponse(
        summary="Test", 
        minutes=[MinutesSection(title="T", content="C", cited_ids=["D99"])]
    )
    mock_client.complete_json.return_value = mock_resp
    
    rec = write_minutes([d], [], meta, mock_client)
    
    # Called twice (1 retry)
    assert mock_client.complete_json.call_count == 2
    # Falls back to template (summary should indicate fallback)
    assert "fallback" in rec.summary.lower()
    # Warning added
    assert any("LLM failed or post-checks rejected" in w for w in meta.warnings)
