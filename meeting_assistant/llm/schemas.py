"""Pydantic schemas for the LLM stages of the pipeline."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    id: str = Field(..., description="Unique ID for this utterance, e.g., 'u001'.")
    speaker: str = Field(..., description="Speaker identifier.")
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")
    text: str = Field(..., description="The spoken text.")


class Edit(BaseModel):
    span_id: str = Field(..., description="ID of the disputed span.")
    choice: Literal["ENGINE_A", "ENGINE_B", "UNRESOLVED"] = Field(
        ..., description="Which engine's alternative was chosen, or UNRESOLVED."
    )
    final_text: str = Field(..., description="The chosen text to insert.")
    reason: str = Field(..., description="Brief reason for the choice.")


class RefinementStats(BaseModel):
    edit_ratio: float
    counts: dict[str, int]


class RefinementResult(BaseModel):
    refined_text: str
    utterances: List[Utterance]
    edits: List[Edit]
    rejected_edits: List[Edit]
    flags: List[str]
    stats: RefinementStats


class DialogueAct(BaseModel):
    utterance_id: str
    label: Literal[
        "ISSUE", "PROPOSAL", "AGREEMENT", "REJECTION", "DEFERRAL", "TASK", "OWNER", "TIMEFRAME", "INFO"
    ]
    quote: str
    in_reply_to: Optional[str] = None


class Decision(BaseModel):
    id: str
    text: str
    status: Literal["AGREED", "PROPOSED", "REJECTED", "DEFERRED"]
    evidence: List[str] = Field(..., description="List of 'quote + utterance_id'.")
    low_confidence: bool


class ActionItem(BaseModel):
    id: str
    task: str
    status: Literal["CONFIRMED", "TENTATIVE"]
    owner: str = Field(..., description="Person name | 'Speaker N (self-assigned)' | 'Unspecified'")
    deadline: str = Field(..., description="Verbatim phrase | 'Unspecified'")
    evidence: List[str]
    low_confidence: bool


class MeetingMetadata(BaseModel):
    models_used: List[str]
    call_log_summary: dict
    warnings: List[str]


class MeetingRecord(BaseModel):
    summary: str
    minutes: List[dict] = Field(..., description="List of sections, each line cites ledger ids")
    decisions: List[Decision]
    action_items: List[ActionItem]
    metadata: MeetingMetadata
