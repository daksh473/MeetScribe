"""Pydantic data models for Speech-To-Text pipeline."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Word(BaseModel):
    """Word-level transcription token with timestamps."""

    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None


class Segment(BaseModel):
    """Segment / utterance block containing transcribed words."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    words: List[Word] = Field(default_factory=list)


class EngineResult(BaseModel):
    """Direct result returned by an individual STT engine."""

    engine_name: str
    success: bool
    error: Optional[str] = None
    error_class: Optional[str] = None
    text: str = ""
    segments: List[Segment] = Field(default_factory=list)
    words: List[Word] = Field(default_factory=list)
    processing_time_s: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class UncertainSpan(BaseModel):
    """Disputed span where STT engines disagree during consensus alignment."""

    id: str
    start: float
    end: float
    category: Literal["CRITICAL", "HIGH", "LOW"]
    chosen_text: str
    alternatives: Dict[str, str] = Field(default_factory=dict)
    context_before: str = ""
    context_after: str = ""


class STTMetadata(BaseModel):
    """Operational metadata regarding the STT pipeline run."""

    engines_used: List[str] = Field(default_factory=list)
    engine_statuses: Dict[str, str] = Field(default_factory=dict)
    fallback_used: bool = False
    audio_duration_s: float = 0.0
    chunk_count: int = 1
    processing_time_s: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class STTResult(BaseModel):
    """Consolidated STT output consumed by downstream LLM-1 refinement."""

    raw_text: str
    segments: List[Segment] = Field(default_factory=list)
    uncertain_spans: List[UncertainSpan] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    metadata: STTMetadata = Field(default_factory=STTMetadata)


class AudioInfo(BaseModel):
    """Audio metadata extracted from file inspection via ffprobe."""

    duration: float
    codec: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
