"""Pure code guards to reject unsafe edits."""

import re
from typing import List, Optional

NEGATIONS = {
    "not", "no", "never", "cannot", "without", "neither", "nor", 
    "dont", "wont", "cant", "shouldnt", "wouldnt", "couldnt", 
    "isnt", "arent", "aint", "doesnt", "didnt", "hasnt", "havent", "hadnt"
}

MODALS = {
    "will", "won't", "would", "should", "shouldn't", "might", "may", 
    "must", "can", "can't", "cannot", "could", "shall", "let's"
}

PHRASAL_MODALS = ["going to", "need to", "have to"]

def _get_tokens(text: str) -> set[str]:
    # Strip punctuation except apostrophes
    text = re.sub(r'[^\w\s\']', '', text.lower())
    return set(text.split())


def reject_edit(original_text: str, final_text: str, engine_alternatives: List[str]) -> Optional[str]:
    """Check if an edit violates any hard safety guards. Returns rejection reason, or None if safe."""
    
    orig_tokens = _get_tokens(original_text)
    final_tokens = _get_tokens(final_text)
    alt_tokens = set()
    for alt in engine_alternatives:
        alt_tokens.update(_get_tokens(alt))
        
    # 1. Introduces hallucinated token
    for ft in final_tokens:
        if ft not in alt_tokens and ft not in orig_tokens:
            # We allow basic case/punctuation differences due to _get_tokens
            return f"Introduces token not present in any engine alternative: '{ft}'"
            
    # 2. Changes a number/date/time
    orig_nums = set(re.findall(r'\d+', original_text))
    final_nums = set(re.findall(r'\d+', final_text))
    if orig_nums != final_nums:
        return "Changes a number."
        
    # 3. Negation toggles
    orig_negs = orig_tokens & NEGATIONS
    final_negs = final_tokens & NEGATIONS
    if orig_negs != final_negs:
        return "Adds, removes, or toggles a negation word."
        
    # 4. Modals
    orig_modals = orig_tokens & MODALS
    final_modals = final_tokens & MODALS
    if orig_modals != final_modals:
        return "Changes a modal word."
        
    orig_lower = original_text.lower()
    final_lower = final_text.lower()
    for pm in PHRASAL_MODALS:
        if (pm in orig_lower) != (pm in final_lower):
            return "Changes a modal phrase."
        
    return None
