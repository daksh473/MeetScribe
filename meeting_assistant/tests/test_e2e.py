"""End to End test for the whole pipeline logic."""

import json
from unittest.mock import MagicMock, patch
from stt.schema import STTResult, Segment, Word, STTMetadata
from llm.schemas import DialogueAct, Utterance
from minutes.compiler import compile_ledger

def test_deterministic_mock_meeting():
    """
    Part 9 - Mock Meeting Test:
    Verify the compiler deterministic logic against the exact requested transcript.
    """
    # 1. Mock transcript
    mock_transcript = [
        "Maybe we should deploy Friday.",
        "Should John handle the deployment?",
        "John might look into it.",
        "I'll handle the testing.",
        "Yes, let's deploy on Friday.",
        "Actually, no, let's postpone it.",
        "We will not ship this version."
    ]
    
    utterances = [
        Utterance(id=f"u{i}", speaker="spk1", start=i, end=i+1, text=t)
        for i, t in enumerate(mock_transcript)
    ]
    
    # 2. Mock Annotations (Simulate what the LLM-2 should extract)
    acts = [
        # "Maybe we should deploy Friday."
        DialogueAct(utterance_id="u0", label="PROPOSAL", quote="Maybe we should deploy Friday."),
        DialogueAct(utterance_id="u0", label="TIMEFRAME", quote="Friday"),
        
        # "Should John handle the deployment?"
        DialogueAct(utterance_id="u1", label="TASK", quote="John handle the deployment"),
        DialogueAct(utterance_id="u1", label="OWNER", quote="John"),
        
        # "John might look into it."
        DialogueAct(utterance_id="u2", label="OWNER", quote="John", in_reply_to="u1"),
        
        # "I'll handle the testing."
        DialogueAct(utterance_id="u3", label="TASK", quote="I'll handle the testing."),
        DialogueAct(utterance_id="u3", label="OWNER", quote="I'll handle the testing."),
        
        # "Yes, let's deploy on Friday."
        DialogueAct(utterance_id="u4", label="AGREEMENT", quote="Yes, let's deploy", in_reply_to="u0"),
        
        # "Actually, no, let's postpone it."
        DialogueAct(utterance_id="u5", label="REJECTION", quote="Actually, no, let's postpone it.", in_reply_to="u0"),
        
        # "We will not ship this version."
        DialogueAct(utterance_id="u6", label="REJECTION", quote="We will not ship this version.")
    ]
    
    # 3. Deterministic Compilation
    decs, tasks = compile_ledger(acts, utterances)
    
    # Validation
    # Decision 1: Deploy Friday -> Was agreed (u4) but later rejected/deferred (u5)
    # The compiler assigns REJECTED (or DEFERRED based on clustering)
    deploy_dec = next((d for d in decs if "deploy" in d.text.lower()), None)
    assert deploy_dec is not None
    assert deploy_dec.status in ("REJECTED", "DEFERRED")
    assert deploy_dec.status != "AGREED"
    
    # Task 1: John handle deployment -> tentative, not confirmed
    john_task = next((t for t in tasks if "john" in t.task.lower()), None)
    assert john_task is not None
    assert john_task.status == "TENTATIVE"
    assert john_task.status != "CONFIRMED"
    
    # Task 2: I'll handle testing -> confirmed, owner = self-assigned
    test_task = next((t for t in tasks if "testing" in t.task.lower()), None)
    assert test_task is not None
    assert test_task.status == "CONFIRMED"
    assert "self-assigned" in test_task.owner.lower()

