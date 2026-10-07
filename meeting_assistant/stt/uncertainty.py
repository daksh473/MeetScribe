"""Uncertainty analysis for STT consensus.

Converts disputed alignment slots into UncertainSpan objects, merges adjacent
disputes, extracts surrounding context, and categorizes the severity of the
disagreement (CRITICAL, HIGH, LOW).
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List

from .consensus import AlignmentSlot
from .schema import EngineResult, UncertainSpan


# Words that toggle negation.
NEGATION_WORDS = {
    "not", "no", "never", "cannot", "without", "neither", "nor",
    "dont", "wont", "cant", "shouldnt", "wouldnt", "couldnt", "isnt", "arent", "aint",
    "doesnt", "didnt", "hasnt", "havent", "hadnt",
}

# Very lightweight common word list to filter HIGH category.
# In a real system, this would be much larger or use a dictionary package.
# We just include basic function words and common verbs.
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "are", "is", "was", "were", "been", "being", "am",
}


def _is_critical(alternatives: List[str]) -> bool:
    """Check if any alternative triggers CRITICAL rules."""
    for alt in alternatives:
        if not alt:
            continue
            
        alt_lower = alt.lower()
        words = alt_lower.split()
        
        # Check for negation words
        for w in words:
            # Strip punctuation for the check
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w in NEGATION_WORDS:
                return True
                
        # Check for numbers/digits or percentage words
        if re.search(r'\d', alt):
            return True
        for w in words:
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w in {"percent", "percentage", "dollar", "dollars", "euro", "euros"}:
                return True
            # We can check a few common number words since alternatives use original text
            if clean_w in {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion"}:
                return True
                
        # Check for currency/percentage symbols
        if re.search(r'[\$£€%]', alt):
            return True
            
        # Time/date indicators (very simplistic for rule)
        time_words = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                      "january", "february", "march", "april", "may", "june", "july", "august",
                      "september", "october", "november", "december", "today", "tomorrow", "yesterday"}
        for w in words:
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w in time_words:
                return True
                
    return False


def _is_high(alternatives: List[str]) -> bool:
    """Check if any alternative triggers HIGH rules."""
    for alt in alternatives:
        if not alt:
            continue
            
        words = alt.split()
        for w in words:
            clean_w = re.sub(r'[^\w\s]', '', w.lower())
            if not clean_w:
                continue
                
            # ALL CAPS (acronyms)
            # Need to ensure it has letters
            if w.isupper() and re.search(r'[A-Z]', w):
                return True
                
            # Mixed digits/letters (e.g., product codes like "XJ900")
            if re.search(r'[a-zA-Z]', w) and re.search(r'\d', w):
                return True
                
            # Proper nouns (capitalized word not at start of sentence, though we can't easily tell here.
            # We'll just check if it's not a common word).
            if clean_w not in COMMON_WORDS:
                return True
                
    return False


def _categorize(alternatives: List[str]) -> str:
    """Categorize the span based on the alternatives."""
    if _is_critical(alternatives):
        return "CRITICAL"
    if _is_high(alternatives):
        return "HIGH"
    return "LOW"


def generate_spans(disputed_slots: List[AlignmentSlot], results: List[EngineResult]) -> List[UncertainSpan]:
    """Merge adjacent disputed slots into UncertainSpans."""
    if not disputed_slots:
        return []
        
    spans = []
    
    # We need a way to determine adjacency. Since we don't have the full slot array here,
    # we can use the original word indices from engine 1 (or engine 2) to group them.
    # We will assume disputed_slots are passed in sequence order (which they are from build_consensus).
    
    current_group = [disputed_slots[0]]
    
    def _are_adjacent(slot1: AlignmentSlot, slot2: AlignmentSlot) -> bool:
        # Check if they are adjacent in any engine
        for engine_name in slot1.engine_tokens:
            tok1 = slot1.engine_tokens.get(engine_name)
            tok2 = slot2.engine_tokens.get(engine_name)
            
            if tok1 and tok2 and tok1.orig_indices and tok2.orig_indices:
                # If the last index of tok1 is exactly one less than the first index of tok2
                if tok1.orig_indices[-1] + 1 == tok2.orig_indices[0]:
                    return True
        return False

    for slot in disputed_slots[1:]:
        # If adjacent to the last slot in current_group, append it.
        # Note: If one engine has a gap, they might still be adjacent in the other engine.
        if _are_adjacent(current_group[-1], slot):
            current_group.append(slot)
        else:
            # Check if there's only gaps between them. 
            # For strict adjacent merging, this is sufficient.
            spans.append(_build_span(current_group, results))
            current_group = [slot]
            
    if current_group:
        spans.append(_build_span(current_group, results))
        
    return spans


def _build_span(slots: List[AlignmentSlot], results: List[EngineResult]) -> UncertainSpan:
    """Build a single UncertainSpan from a group of adjacent disputed slots."""
    alternatives = {}
    
    start_time = float('inf')
    end_time = -1.0
    
    # We need to construct the text for each engine and find overall start/end
    for res in results:
        engine_name = res.engine_name
        engine_text_parts = []
        
        first_idx = None
        last_idx = None
        
        for slot in slots:
            tok = slot.engine_tokens.get(engine_name)
            if tok:
                if first_idx is None and tok.orig_indices:
                    first_idx = tok.orig_indices[0]
                if tok.orig_indices:
                    last_idx = tok.orig_indices[-1]
                    
                engine_text_parts.append(" ".join([res.words[i].text for i in tok.orig_indices]))
                
        alt_text = " ".join(engine_text_parts).strip()
        alternatives[engine_name] = alt_text
        
        # Update start/end times
        if first_idx is not None and first_idx < len(res.words):
            start_time = min(start_time, res.words[first_idx].start)
        if last_idx is not None and last_idx < len(res.words):
            end_time = max(end_time, res.words[last_idx].end)
            
    if start_time == float('inf'):
        start_time = 0.0
    if end_time == -1.0:
        end_time = 0.0
        
    # Get context from the first engine that has tokens for this span.
    context_before = ""
    context_after = ""
    
    for res in results:
        engine_name = res.engine_name
        
        first_idx = None
        last_idx = None
        
        for slot in slots:
            tok = slot.engine_tokens.get(engine_name)
            if tok and tok.orig_indices:
                if first_idx is None:
                    first_idx = tok.orig_indices[0]
                last_idx = tok.orig_indices[-1]
                
        if first_idx is not None and last_idx is not None:
            # Get 8 words before
            start_idx = max(0, first_idx - 8)
            context_before = " ".join([w.text for w in res.words[start_idx:first_idx]])
            
            # Get 8 words after
            end_idx = min(len(res.words), last_idx + 1 + 8)
            context_after = " ".join([w.text for w in res.words[last_idx + 1:end_idx]])
            break  # Found context, stop looking
            
    # For chosen_text, we just pick the alternative from the first engine for now
    # The true chosen_text would come from the consensus words, but for simplicity
    # we can use the text from the highest priority engine.
    chosen_text = alternatives.get(results[0].engine_name, "")
            
    # Remove empty alternatives (where the engine had a gap)
    alternatives_filtered = {k: v for k, v in alternatives.items() if v}
            
    category = _categorize(list(alternatives_filtered.values()))
    
    return UncertainSpan(
        id=str(uuid.uuid4()),
        start=start_time,
        end=end_time,
        category=category,
        chosen_text=chosen_text,
        alternatives=alternatives_filtered,
        context_before=context_before,
        context_after=context_after,
    )


def summarize(spans: List[UncertainSpan]) -> Dict[str, int]:
    """Count spans by category."""
    counts = {"CRITICAL": 0, "HIGH": 0, "LOW": 0}
    for span in spans:
        if span.category in counts:
            counts[span.category] += 1
    return counts


def overlaps(span: UncertainSpan, start: float, end: float) -> bool:
    """Check if the given time range overlaps with the uncertain span."""
    return span.start < end and start < span.end
