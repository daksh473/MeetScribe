"""Tests for the Verifier."""

from unittest.mock import MagicMock
from llm.schemas import Decision, Utterance
from stt.schema import UncertainSpan
from minutes.verifier import verify_ledger, VerificationResponse, ItemVerification


def test_ungrounded_quote_dropped():
    # Quote isn't in utterance
    d = Decision(id="D1", text="Test", status="PROPOSED", evidence=["Not in text [u1]"], low_confidence=False)
    u = Utterance(id="u1", speaker="s1", start=0, end=1, text="Different text")
    
    decs, tasks, flags = verify_ledger([d], [], [u], [])
    
    assert len(decs) == 0
    assert any("Dropped ungrounded" in f for f in flags)


def test_low_confidence_propagated_from_disputed_span():
    d = Decision(id="D1", text="Test", status="PROPOSED", evidence=["Overlap text [u1]"], low_confidence=False)
    u = Utterance(id="u1", speaker="s1", start=0, end=2, text="Overlap text is here")
    span = UncertainSpan(id="s1", start=0.5, end=1.5, chosen_text="text", alternatives={}, category="HIGH", context_before="", context_after="")
    
    decs, tasks, flags = verify_ledger([d], [], [u], [span])
    
    assert len(decs) == 1
    assert decs[0].low_confidence is True


def test_verifier_can_only_downgrade():
    d1 = Decision(id="D1", text="Test", status="AGREED", evidence=["Test [u1]"], low_confidence=False)
    d2 = Decision(id="D2", text="Test2", status="PROPOSED", evidence=["Test2 [u1]"], low_confidence=False)
    u = Utterance(id="u1", speaker="s1", start=0, end=1, text="Test and Test2")
    
    mock_client = MagicMock()
    # LLM marks both unsupported
    mock_resp = VerificationResponse(verifications=[
        ItemVerification(item_id="D1", unsupported_owner=False, unsupported_deadline=False, unsupported_status=True, reason=""),
        ItemVerification(item_id="D2", unsupported_owner=False, unsupported_deadline=False, unsupported_status=True, reason="")
    ])
    mock_client.complete_json.return_value = mock_resp
    
    decs, tasks, flags = verify_ledger([d1, d2], [], [u], [], client=mock_client)
    
    assert len(decs) == 2
    # D1 downgraded
    assert decs[0].status == "PROPOSED"
    # D2 stays PROPOSED (cannot be downgraded further or dropped)
    assert decs[1].status == "PROPOSED"
