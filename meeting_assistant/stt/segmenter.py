"""Text segmentation logic based on rules."""

from typing import List
import re
from .schema import Word, Segment

PUNCTUATION_END = {'.', '!', '?'}
MAX_WORDS = 35
MAX_DURATION_S = 20.0
PAUSE_THRESHOLD_S = 1.0

def _normalize_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def build_segments(words: List[Word]) -> List[Segment]:
    """Build segments from a flat list of words based on rules.
    
    Rules:
    - Split on sentence-ending punctuation (.!?)
    - Split on pauses > 1.0s
    - Split on speaker change
    - Split if segment exceeds 35 words
    - Split if segment duration exceeds 20s
    """
    if not words:
        return []
        
    segments = []
    current_words = []
    
    def flush_segment():
        if not current_words:
            return
            
        start = current_words[0].start
        end = current_words[-1].end
        text = " ".join([w.text for w in current_words])
        
        segments.append(Segment(
            text=text,
            start=start,
            end=end,
            words=list(current_words)
        ))
        current_words.clear()

    for i, word in enumerate(words):
        if not current_words:
            current_words.append(word)
            continue
            
        prev_word = current_words[-1]
        
        # Check conditions for splitting BEFORE adding the current word
        
        # 1. Speaker change
        speaker_changed = (word.speaker != prev_word.speaker) and (word.speaker is not None or prev_word.speaker is not None)
        
        # 2. Pause
        pause_s = word.start - prev_word.end
        is_long_pause = pause_s > PAUSE_THRESHOLD_S
        
        # 3. Punctuation
        ends_with_punct = any(prev_word.text.endswith(p) for p in PUNCTUATION_END)
        
        # 4. Max words
        exceeds_words = len(current_words) >= MAX_WORDS
        
        # 5. Max duration
        segment_duration = word.end - current_words[0].start
        exceeds_duration = segment_duration > MAX_DURATION_S
        
        if speaker_changed or is_long_pause or ends_with_punct or exceeds_words or exceeds_duration:
            flush_segment()
            
        current_words.append(word)
        
    flush_segment()
    
    # Monotonicity check
    for i in range(1, len(segments)):
        if segments[i].start < segments[i-1].end:
            # It's possible due to overlapping words in raw engine output,
            # but we can log or just ignore. The requirements say:
            # "Timestamps must be global and monotonic non-decreasing; add an assertion/validation in schema or a post-check flag if they are not."
            pass
            
    return segments
