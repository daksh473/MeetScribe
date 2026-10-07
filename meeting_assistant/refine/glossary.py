"""Glossary checking and unification."""

import difflib
import re
from collections import Counter
from typing import List, Tuple

from stt.schema import UncertainSpan


def find_inconsistent_spellings(text: str, disputed_spans: List[UncertainSpan]) -> Tuple[List[str], List[dict]]:
    """Find inconsistent spellings of capitalized terms/acronyms.
    
    Returns:
        flags: List of warning strings.
        unifications: List of dicts with {'old': '...', 'new': '...'} to apply.
    """
    words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    # Also find some non-capitalized terms that might match capitalized ones
    all_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    word_counts = Counter(all_words)
    candidates = set(words)
    
    # Gather disputed texts
    disputed_texts = set()
    for span in disputed_spans:
        for alt in span.alternatives.values():
            disputed_texts.update(re.findall(r'\b[a-zA-Z]+\b', alt))
            
    flags = []
    unifications = []
    
    checked = set()
    
    for cand in list(candidates):
        if cand in checked or len(cand) < 4:
            continue
            
        # Find similar words
        similar = difflib.get_close_matches(cand, list(word_counts.keys()), n=5, cutoff=0.8)
        if len(similar) > 1:
            # We found variants!
            checked.update(similar)
            
            # Sort by frequency
            similar.sort(key=lambda x: word_counts[x], reverse=True)
            dominant = similar[0]
            
            for minority in similar[1:]:
                # Is dominant clearly dominant? e.g. > 2x
                if word_counts[dominant] > 2 * word_counts[minority]:
                    if minority in disputed_texts:
                        unifications.append({'old': minority, 'new': dominant})
                    else:
                        flags.append(f"Inconsistent spelling detected: '{dominant}' vs '{minority}'")
                else:
                    flags.append(f"Inconsistent spelling detected: '{dominant}' vs '{minority}'")
                    
    return flags, unifications
