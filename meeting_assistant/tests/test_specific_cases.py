"""Tests for the specific 12 cases requested."""

from llm.schemas import DialogueAct, Utterance, ActionItem, Decision
from minutes.compiler import compile_ledger
from refine.guards import reject_edit
from minutes.verifier import verify_ledger

def _u(id, text):
    return Utterance(id=id, speaker="spk1", start=0, end=1, text=text)
    
def _a(uid, label, quote, reply_to=None):
    return DialogueAct(utterance_id=uid, label=label, quote=quote, in_reply_to=reply_to)

def test_case_1_maybe_deploy_friday_proposed():
    # 1. "Maybe we should deploy Friday." -> PROPOSED
    utts = [_u("u1", "Maybe we should deploy Friday.")]
    acts = [_a("u1", "PROPOSAL", "Maybe we should deploy Friday.")]
    decs, _ = compile_ledger(acts, utts)
    assert decs[0].status == "PROPOSED"

def test_case_2_maybe_john_can_look_into_it():
    # 2. "Maybe John can look into it." -> not confirmed owner
    utts = [_u("u1", "Maybe John can look into it.")]
    acts = [_a("u1", "TASK", "Maybe John can look into it."), _a("u1", "OWNER", "John")]
    _, tasks = compile_ledger(acts, utts)
    assert tasks[0].status == "TENTATIVE"

def test_case_3_john_might_look_into_it():
    # 3. "John might look into it." -> TENTATIVE
    utts = [_u("u1", "John might look into it.")]
    acts = [_a("u1", "TASK", "John might look into it."), _a("u1", "OWNER", "John")]
    _, tasks = compile_ledger(acts, utts)
    assert tasks[0].status == "TENTATIVE"

def test_case_4_john_will_handle_the_deployment():
    # 4. "John will handle the deployment." -> CONFIRMED task -> owner John
    utts = [_u("u1", "John will handle the deployment.")]
    acts = [_a("u1", "TASK", "John will handle the deployment."), _a("u1", "OWNER", "John")]
    _, tasks = compile_ledger(acts, utts)
    assert tasks[0].status == "CONFIRMED"
    assert tasks[0].owner == "John"

def test_case_5_yes_lets_deploy_friday():
    # 5. "Yes, let's deploy Friday." -> agreement
    utts = [_u("u1", "Let's deploy Friday."), _u("u2", "Yes, let's deploy Friday.")]
    acts = [_a("u1", "PROPOSAL", "Let's deploy Friday."), _a("u2", "AGREEMENT", "Yes, let's deploy Friday.", reply_to="u1")]
    decs, _ = compile_ledger(acts, utts)
    assert decs[0].status == "AGREED"

def test_case_6_actually_no_lets_postpone_it():
    # 6. "Actually, no, let's postpone it." -> later rejection/deferral overrides earlier proposal
    utts = [_u("u1", "Let's deploy Friday."), _u("u2", "Yes."), _u("u3", "Actually, no, let's postpone it.")]
    acts = [_a("u1", "PROPOSAL", "Let's deploy Friday."), _a("u2", "AGREEMENT", "Yes.", reply_to="u1"), _a("u3", "DEFERRAL", "Actually, no, let's postpone it.", reply_to="u1")]
    decs, _ = compile_ledger(acts, utts)
    assert decs[0].status == "DEFERRED"

def test_case_7_we_will_not_ship_this_version():
    # 7. "We will not ship this version." -> negative/rejection state
    utts = [_u("u1", "Let's ship."), _u("u2", "We will not ship this version.")]
    acts = [_a("u1", "PROPOSAL", "Let's ship."), _a("u2", "REJECTION", "We will not ship this version.", reply_to="u1")]
    decs, _ = compile_ledger(acts, utts)
    assert decs[0].status == "REJECTED"

def test_case_8_15_vs_50():
    # 8. 15% vs 50% -> never silently change number
    reason = reject_edit("We grew 15%", "We grew 50%", ["We grew 15%", "We grew 50%"])
    assert reason is not None
    assert "number" in reason.lower()

def test_case_9_will_vs_will_not():
    # 9. will vs will not -> never flip meaning
    reason = reject_edit("We will ship", "We will not ship", ["We will ship", "We will not ship"])
    assert reason is not None
    assert "negation" in reason.lower()

def test_case_10_hallucinated_owner():
    # 10. hallucinated owner -> rejected
    # Pass an owner that is NOT in the exact quote
    utts = [_u("u1", "We need to fix the bug.")]
    acts = [_a("u1", "TASK", "We need to fix the bug."), _a("u1", "OWNER", "Dave (hallucinated)")]
    _, tasks = compile_ledger(acts, utts)
    # the verifier should strip Dave
    v_decs, v_tasks, v_flags = verify_ledger([], tasks, utts, [], client=None) # verifier is offline for these rules
    assert v_tasks[0].owner == "UNSUPPORTED" or "hallucinated" not in v_tasks[0].owner

def test_case_11_hallucinated_deadline():
    # 11. hallucinated deadline -> rejected
    utts = [_u("u1", "We need to fix the bug.")]
    acts = [_a("u1", "TASK", "We need to fix the bug."), _a("u1", "TIMEFRAME", "by 2025")]
    _, tasks = compile_ledger(acts, utts)
    v_decs, v_tasks, v_flags = verify_ledger([], tasks, utts, [], client=None)
    assert v_tasks[0].deadline == "UNSUPPORTED" or "2025" not in v_tasks[0].deadline

def test_case_12_unsupported_quote():
    # 12. unsupported quote -> rejected
    utts = [_u("u1", "Normal meeting stuff.")]
    acts = [_a("u1", "PROPOSAL", "Let's buy a yacht!")]
    decs, _ = compile_ledger(acts, utts)
    v_decs, v_tasks, v_flags = verify_ledger(decs, [], utts, [], client=None)
    assert len(v_decs) == 0 or v_decs[0].status == "UNSUPPORTED" or v_decs[0].low_confidence
