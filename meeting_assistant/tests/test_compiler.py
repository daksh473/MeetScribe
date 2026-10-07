"""Tests for compiler (Pass B)."""

from llm.schemas import DialogueAct, Utterance
from minutes.compiler import compile_ledger


def _u(id: str, text: str, speaker="spk1"):
    return Utterance(id=id, speaker=speaker, start=0, end=1, text=text)

def _a(uid: str, label: str, quote: str, reply_to=None):
    return DialogueAct(utterance_id=uid, label=label, quote=quote, in_reply_to=reply_to)


def test_proposal_only_proposed():
    utts = [_u("u1", "We should use AWS.")]
    acts = [_a("u1", "PROPOSAL", "We should use AWS.")]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(decs) == 1
    assert decs[0].status == "PROPOSED"


def test_proposal_plus_agreement_agreed():
    utts = [_u("u1", "We should use AWS."), _u("u2", "Sounds good.")]
    acts = [
        _a("u1", "PROPOSAL", "We should use AWS."),
        _a("u2", "AGREEMENT", "Sounds good.", reply_to="u1")
    ]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(decs) == 1
    assert decs[0].status == "AGREED"


def test_proposal_plus_agreement_plus_rejection_rejected():
    utts = [_u("u1", "Use AWS."), _u("u2", "Yes."), _u("u3", "Wait, no, it's too expensive.")]
    acts = [
        _a("u1", "PROPOSAL", "Use AWS."),
        _a("u2", "AGREEMENT", "Yes.", reply_to="u1"),
        _a("u3", "REJECTION", "Wait, no, it's too expensive.", reply_to="u1")
    ]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(decs) == 1
    assert decs[0].status == "REJECTED"


def test_task_no_owner_unspecified():
    utts = [_u("u1", "Can John look at it?")]
    acts = [_a("u1", "TASK", "Can John look at it?")]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(tasks) == 1
    assert tasks[0].owner == "Unspecified"
    assert tasks[0].status == "TENTATIVE"


def test_task_first_person_owner():
    utts = [_u("u1", "I'll do it.", speaker="SPEAKER_01")]
    acts = [
        _a("u1", "TASK", "I'll do it."),
        _a("u1", "OWNER", "I'll do it.")
    ]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(tasks) == 1
    assert tasks[0].owner == "SPEAKER_01 (self-assigned)"
    assert tasks[0].status == "CONFIRMED"


def test_task_deadline_verbatim():
    utts = [_u("u1", "Fix it by next Friday.")]
    acts = [
        _a("u1", "TASK", "Fix it"),
        _a("u1", "TIMEFRAME", "by next Friday.")
    ]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(tasks) == 1
    assert tasks[0].deadline == "by next Friday."


def test_act_with_no_evidence_dropped():
    utts = [_u("u1", "We should do X.")]
    # Intentionally bad act missing quote
    acts = [DialogueAct(utterance_id="u1", label="PROPOSAL", quote="")]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(decs) == 0


def test_low_confidence_inexact_quote():
    utts = [_u("u1", "We should use AWS.")]
    # Quote differs from utterance
    acts = [_a("u1", "PROPOSAL", "Use AWS")]
    
    decs, tasks = compile_ledger(acts, utts)
    assert len(decs) == 1
    assert decs[0].low_confidence is True
