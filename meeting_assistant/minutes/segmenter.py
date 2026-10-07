"""Segment list of Utterances for LLM batched processing."""

from typing import List
from llm.schemas import Utterance


def segment_utterances(utterances: List[Utterance], max_size: int = 60, overlap: int = 5) -> List[List[Utterance]]:
    """
    Split the refined utterance list into topic segments of bounded size.
    Adds a small overlap to preserve context for in_reply_to links.
    """
    if not utterances:
        return []
        
    segments = []
    start_idx = 0
    total = len(utterances)
    
    while start_idx < total:
        end_idx = min(start_idx + max_size, total)
        segments.append(utterances[start_idx:end_idx])
        
        if end_idx == total:
            break
            
        prev_start = end_idx - max_size if end_idx - start_idx == max_size else start_idx
        start_idx = end_idx - overlap
        
        # Prevent infinite loop if we didn't advance
        if start_idx <= prev_start:
            start_idx = end_idx
            
    return segments
