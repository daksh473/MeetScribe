"""Pass C: Verifier. Validates and down-grades compiled ledger items."""

import re
from typing import List, Optional
from pydantic import BaseModel
from llm.client import LLMClient
from llm.schemas import ActionItem, Decision, Utterance
from stt.schema import UncertainSpan

def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


class ItemVerification(BaseModel):
    item_id: str
    unsupported_owner: bool
    unsupported_deadline: bool
    unsupported_status: bool
    reason: str


class VerificationResponse(BaseModel):
    verifications: List[ItemVerification]


VERIFIER_SYSTEM = """You are a strict Ledger Verifier. Your job is to check if the extracted fields (status, owner, deadline) of decisions and tasks are STRICTLY supported by the provided evidence quotes.
- First identify which parts (owner, deadline, status) are NOT directly supported by the quoted evidence.
- Score each field with a boolean `unsupported_...`. True means the field is an hallucination or overly confident interpretation.
- You CANNOT upgrade or add anything. You only check support.
"""


def _parse_evidence(ev: str) -> tuple[str, str]:
    match = re.search(r"^(.*) \[(u\d+)\]$", ev)
    if match:
        return match.group(1), match.group(2)
    return ev, ""


def _is_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
    return not (e1 < s2 or s1 > e2)


def verify_ledger(
    decisions: List[Decision],
    tasks: List[ActionItem],
    utterances: List[Utterance],
    spans: List[UncertainSpan],
    client: Optional[LLMClient] = None
) -> tuple[List[Decision], List[ActionItem], List[str]]:
    
    flags = []
    
    # 1. Code checks: Grounding & Uncertainty Overlap
    valid_decisions = []
    for d in decisions:
        keep = False
        new_ev = []
        for ev in d.evidence:
            quote, uid = _parse_evidence(ev)
            utt = next((u for u in utterances if u.id == uid), None)
            if not utt:
                continue
                
            norm_q = normalize_text(quote)
            norm_t = normalize_text(utt.text)
            if norm_q in norm_t:
                keep = True
                new_ev.append(ev)
                # Check overlap with unresolved/disputed spans
                for span in spans:
                    if span.category in {"HIGH", "CRITICAL"} or "unresolved" in span.chosen_text.lower():
                        if _is_overlap(utt.start, utt.end, span.start, span.end):
                            d.low_confidence = True
                            d.text += f" (Warning: Evidence overlaps disputed span {span.id})"
                            break
        if keep:
            d.evidence = new_ev
            valid_decisions.append(d)
        else:
            flags.append(f"Dropped ungrounded decision {d.id}")
            
    valid_tasks = []
    for t in tasks:
        keep = False
        new_ev = []
        for ev in t.evidence:
            quote, uid = _parse_evidence(ev)
            utt = next((u for u in utterances if u.id == uid), None)
            if not utt:
                continue
                
            norm_q = normalize_text(quote)
            norm_t = normalize_text(utt.text)
            if norm_q in norm_t:
                keep = True
                new_ev.append(ev)
                for span in spans:
                    if span.category in {"HIGH", "CRITICAL"} or "unresolved" in span.chosen_text.lower():
                        if _is_overlap(utt.start, utt.end, span.start, span.end):
                            t.low_confidence = True
                            t.task += f" (Warning: Evidence overlaps disputed span {span.id})"
                            break
        if keep:
            t.evidence = new_ev
            valid_tasks.append(t)
        else:
            flags.append(f"Dropped ungrounded task {t.id}")
            
    # 2. LLM Checks
    if not client:
        flags.append("LLM Client unavailable or omitted for Verification (Pass C). Code checks applied.")
        return valid_decisions, valid_tasks, flags
        
    if not valid_decisions and not valid_tasks:
        return valid_decisions, valid_tasks, flags
        
    user_prompt = "Verify these ledger items against their evidence:\n"
    for d in valid_decisions:
        user_prompt += f"DECISION {d.id}: {d.text} | Status: {d.status} | Evidence: {d.evidence}\n"
    for t in valid_tasks:
        user_prompt += f"TASK {t.id}: {t.task} | Status: {t.status} | Owner: {t.owner} | Deadline: {t.deadline} | Evidence: {t.evidence}\n"
        
    try:
        resp = client.complete_json(
            system=VERIFIER_SYSTEM,
            user=user_prompt,
            schema=VerificationResponse
        )
        
        v_map = {v.item_id: v for v in resp.verifications}
        
        for d in valid_decisions:
            v = v_map.get(d.id)
            if v and v.unsupported_status and d.status == "AGREED":
                d.status = "PROPOSED"
                
        for t in valid_tasks:
            v = v_map.get(t.id)
            if not v:
                continue
            if v.unsupported_status and t.status == "CONFIRMED":
                t.status = "TENTATIVE"
            if v.unsupported_owner and t.owner != "Unspecified":
                t.owner = "Unspecified"
            if v.unsupported_deadline and t.deadline != "Unspecified":
                t.deadline = "Unspecified"
                
    except Exception as e:
        flags.append(f"LLM verification failed: {e}. Kept code-verified items.")
        
    return valid_decisions, valid_tasks, flags
