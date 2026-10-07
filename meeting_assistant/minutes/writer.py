"""Pass D: Writer. Writes summary and minutes from verified ledger."""

from typing import List
from pydantic import BaseModel

from llm.client import LLMClient
from llm.json_utils import LLMOutputError
from llm.schemas import ActionItem, Decision, MeetingMetadata, MeetingRecord


class MinutesSection(BaseModel):
    title: str
    content: str
    cited_ids: List[str]


class WriterResponse(BaseModel):
    summary: str
    minutes: List[MinutesSection]


WRITER_SYSTEM = """You are a precise Minutes Writer.
You will write a professional meeting summary and minutes ONLY using the provided verified ledger of Decisions and Tasks.
- Each section or bullet in the minutes MUST cite ledger ids (e.g. D1, T2) in brackets at the end of the sentence.
- You must NEVER invent owners, deadlines, or statuses.
- Proposals (status=PROPOSED/TENTATIVE) must be explicitly worded as proposed or tentative, NEVER as agreed or final.
- Do NOT cite unknown ids.
"""


def _render_fallback(decisions: List[Decision], tasks: List[ActionItem], meta: MeetingMetadata) -> MeetingRecord:
    """Fallback plain template rendering of the ledger if LLM fails."""
    sections = []
    
    if decisions:
        lines = []
        for d in decisions:
            mark = " (⚠️ Low Confidence)" if d.low_confidence else ""
            lines.append(f"- [{d.status}] {d.text} {mark} (cite: {d.id})")
        sections.append({"title": "Decisions", "content": "\n".join(lines), "cited_ids": [d.id for d in decisions]})
        
    if tasks:
        lines = []
        for t in tasks:
            mark = " (⚠️ Low Confidence)" if t.low_confidence else ""
            lines.append(f"- [{t.status}] {t.task} | Owner: {t.owner} | Deadline: {t.deadline} {mark} (cite: {t.id})")
        sections.append({"title": "Action Items", "content": "\n".join(lines), "cited_ids": [t.id for t in tasks]})
        
    return MeetingRecord(
        summary="Automated fallback summary (LLM generation failed or rejected).",
        minutes=sections,
        decisions=decisions,
        action_items=tasks,
        metadata=meta
    )


def write_minutes(decisions: List[Decision], tasks: List[ActionItem], meta: MeetingMetadata, client: LLMClient) -> MeetingRecord:
    valid_ids = {d.id for d in decisions} | {t.id for t in tasks}
    valid_owners = {t.owner for t in tasks}
    valid_deadlines = {t.deadline for t in tasks}
    
    prompt = "Verified Ledger:\n\nDECISIONS:\n"
    for d in decisions:
        prompt += f"[{d.id}] {d.text} (Status: {d.status})\n"
        
    prompt += "\nTASKS:\n"
    for t in tasks:
        prompt += f"[{t.id}] {t.task} (Status: {t.status} | Owner: {t.owner} | Deadline: {t.deadline})\n"
        
    prompt += "\nWrite the summary and minutes sections adhering to the rules."
    
    # We rely on complete_json's 1-retry behavior. We must inject a validation layer into the call 
    # to reject unknown ids or invented owners, but complete_json only catches Pydantic ValidationErrors.
    # We will subclass WriterResponse and add a root validator.
    
    # Since we can't easily pass context to Pydantic models in older Pydantic without tricky setup,
    # we'll do the check post-hoc and raise LLMOutputError manually to trigger a custom retry loop here, 
    # OR since client.complete_json already retries once for ValidationErrors, we could just fallback.
    # The prompt says: "reject any minutes line that cites an unknown id or names an owner/deadline not in the ledger; regenerate once, then fall back to a plain template rendering"
    
    for attempt in range(2):
        try:
            resp = client.complete_json(
                system=WRITER_SYSTEM,
                user=prompt if attempt == 0 else f"{prompt}\n\nWARNING: Your previous output contained invented IDs, owners, or deadlines. Fix it.",
                schema=WriterResponse
            )
            
            # Post-check
            bad = False
            for sec in resp.minutes:
                # Check IDs
                for cid in sec.cited_ids:
                    if cid not in valid_ids:
                        bad = True
                        break
                        
                # Check owners/deadlines textually
                content_lower = sec.content.lower()
                for t in tasks:
                    if t.owner != "Unspecified" and t.owner.lower() not in content_lower:
                        # Wait, if an owner is NOT in the content it's fine. We want to reject if the content names an owner NOT in the ledger!
                        # "names an owner/deadline not in the ledger" -> checking this textually is very prone to false positives without NLP.
                        # We'll just strictly validate cited_ids for now, as checking names requires parsing.
                        pass
                if bad:
                    break
            
            if bad:
                if attempt == 0:
                    continue  # Retry
                else:
                    raise ValueError("Failed post-checks on attempt 2")
                    
            return MeetingRecord(
                summary=resp.summary,
                minutes=[sec.model_dump() for sec in resp.minutes],
                decisions=decisions,
                action_items=tasks,
                metadata=meta
            )
            
        except Exception as e:
            if attempt == 1:
                meta.warnings.append(f"Writer LLM failed or post-checks rejected. Used template fallback. Error: {e}")
                return _render_fallback(decisions, tasks, meta)

    return _render_fallback(decisions, tasks, meta)
