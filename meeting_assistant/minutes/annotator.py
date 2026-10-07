"""Pass A: Annotator. Uses LLM-2 to extract dialogue acts."""

from typing import List
from pydantic import BaseModel

from llm.client import LLMClient
from llm.schemas import DialogueAct, Utterance


class DialogueActList(BaseModel):
    acts: List[DialogueAct]


ANNOTATOR_SYSTEM = """You are a meticulous meeting annotator. Your job is to classify dialogue acts based strictly on what is explicitly said.
- Label only what is explicitly said. A question is not an assignment.
- "Maybe we should..." is a PROPOSAL, not a decision. 
- "Sounds good", "yeah let's do that", "okay" in reply to a proposal is AGREEMENT.
- Do not infer task owners from who was asked, who was mentioned, or who is best placed.
- If unsure, use the INFO label.
- You MUST quote exactly from the utterance text. If you cannot quote exactly, omit the act entirely.
- Link acts to earlier utterances using `in_reply_to` when clearly responding to them.
"""

def annotate_segment(segment: List[Utterance], client: LLMClient) -> List[DialogueAct]:
    """Process a segment of utterances to extract Dialogue Acts."""
    if not segment:
        return []
        
    lines = []
    for u in segment:
        lines.append(f"[{u.id}] {u.speaker}: {u.text}")
        
    user_prompt = "Analyze the following conversation and extract all Dialogue Acts:\n\n" + "\n".join(lines)
    
    # Use fallback models if necessary, handled transparently by LLMClient
    response = client.complete_json(
        system=ANNOTATOR_SYSTEM,
        user=user_prompt,
        schema=DialogueActList,
        temperature=0.0
    )
    
    # We don't verify exact quotes here to remain pure, compiler handles low_confidence.
    return response.acts
