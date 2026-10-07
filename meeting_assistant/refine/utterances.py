"""Convert STT segments to Utterance objects."""

from typing import List

from llm.schemas import Utterance
from stt.schema import STTResult


def build_utterances(stt_result: STTResult) -> List[Utterance]:
    """Group words by speaker into deterministic Utterance segments."""
    utterances = []
    if not stt_result.segments or not stt_result.segments[0].words:
        return []
    
    words = stt_result.segments[0].words
    current_speaker = words[0].speaker or "SPEAKER_00"
    current_words = [words[0]]
    u_idx = 1
    
    for w in words[1:]:
        spk = w.speaker or "SPEAKER_00"
        gap = w.start - current_words[-1].end
        
        if spk != current_speaker or gap > 2.0:
            utterances.append(Utterance(
                id=f"u{u_idx:03d}",
                speaker=current_speaker,
                start=current_words[0].start,
                end=current_words[-1].end,
                text=" ".join(cw.text for cw in current_words)
            ))
            u_idx += 1
            current_speaker = spk
            current_words = [w]
        else:
            current_words.append(w)
            
    if current_words:
        utterances.append(Utterance(
            id=f"u{u_idx:03d}",
            speaker=current_speaker,
            start=current_words[0].start,
            end=current_words[-1].end,
            text=" ".join(cw.text for cw in current_words)
        ))
        
    return utterances
