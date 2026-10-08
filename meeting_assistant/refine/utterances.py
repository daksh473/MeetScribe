"""Convert STT segments to Utterance objects."""

from typing import List

from llm.schemas import Utterance
from stt.schema import STTResult


def build_utterances(stt_result: STTResult) -> List[Utterance]:
    """Map STT segments 1-to-1 to Utterance objects for the ledger."""
    utterances = []
    
    for i, seg in enumerate(stt_result.segments):
        speaker = seg.speaker
        if not speaker:
            # Fallback to Unspecified for the ledger
            speaker = "Unspecified"
            
        utterances.append(Utterance(
            id=f"u{i+1:03d}",
            speaker=speaker,
            start=seg.start,
            end=seg.end,
            text=seg.text
        ))
        
    return utterances
