"""End to End test for the whole pipeline logic."""

import json
from unittest.mock import MagicMock, patch
from stt.schema import STTResult, Segment, Word, STTMetadata

# The run_all script doesn't have an easily callable Python entrypoint that returns 
# objects (it writes to disk and exits). We will test by calling the components just like run_all does.
from refine.refiner import refine
from minutes.segmenter import segment_utterances
from minutes.annotator import annotate_segment
from minutes.compiler import compile_ledger
from minutes.verifier import verify_ledger
from minutes.writer import write_minutes
from minutes.render import render_record
from refine.refiner import LLMRefinementResponse
from minutes.annotator import DialogueActList
from minutes.verifier import VerificationResponse
from minutes.writer import WriterResponse, MinutesSection
from llm.client import LLMClient
from pathlib import Path

def test_e2e_fake_stt_result(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    words = [Word(text="We", start=0, end=1, speaker="s1"), Word(text="should", start=1, end=2, speaker="s1"), Word(text="ship", start=2, end=3, speaker="s1")]
    stt_res = STTResult(
        raw_text="We should ship",
        segments=[Segment(text="We should ship", start=0, end=3, words=words)],
        uncertain_spans=[],
        flags=[],
        metadata=STTMetadata()
    )
    
    mock_client = MagicMock(spec=LLMClient)
    mock_client.call_log = []
    
    # 1. Refiner
    mock_client.complete_json.return_value = LLMRefinementResponse(resolutions=[])
    ref_res = refine(stt_res, mock_client)
    
    # 2. Annotator
    from llm.schemas import DialogueAct
    mock_client.complete_json.return_value = DialogueActList(acts=[
        DialogueAct(utterance_id=ref_res.utterances[0].id, label="PROPOSAL", quote="We should ship")
    ])
    segs = segment_utterances(ref_res.utterances)
    acts = []
    for s in segs:
        acts.extend(annotate_segment(s, mock_client))
        
    # 3. Compiler
    decs, tasks = compile_ledger(acts, ref_res.utterances)
    
    # 4. Verifier
    mock_client.complete_json.return_value = VerificationResponse(verifications=[])
    v_decs, v_tasks, v_flags = verify_ledger(decs, tasks, ref_res.utterances, stt_res.uncertain_spans, client=mock_client)
    
    # 5. Writer
    from llm.schemas import MeetingMetadata
    mock_client.complete_json.return_value = WriterResponse(
        summary="Good meeting",
        minutes=[MinutesSection(title="Decisions", content="We will ship", cited_ids=[v_decs[0].id])]
    )
    meta = MeetingMetadata(models_used=[], call_log_summary={}, warnings=[])
    rec = write_minutes(v_decs, v_tasks, meta, mock_client)
    
    # 6. Render
    render_record(rec, out_dir)
    
    assert (out_dir / "meeting_record.md").exists()
    assert (out_dir / "meeting_record.json").exists()
    
    j = json.loads((out_dir / "meeting_record.json").read_text())
    assert j["summary"] == "Good meeting"
    assert len(j["decisions"]) == 1
    assert j["decisions"][0]["status"] == "PROPOSED"
