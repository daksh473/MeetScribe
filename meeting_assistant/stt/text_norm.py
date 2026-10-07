"""Text normalization for STT consensus alignment.

This module normalizes text strictly for comparison purposes (e.g., aligning
"fifteen" with "15", or "don't" with "dont"). The normalized text is never
shown to the user.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Tuple

# Simple mapping for number words to values.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_ORDINALS = {
    "first": (1, "st"), "second": (2, "nd"), "third": (3, "rd"),
    "fourth": (4, "th"), "fifth": (5, "th"), "sixth": (6, "th"),
    "seventh": (7, "th"), "eighth": (8, "th"), "ninth": (9, "th"),
    "tenth": (10, "th"), "eleventh": (11, "th"), "twelfth": (12, "th"),
    "thirteenth": (13, "th"), "fourteenth": (14, "th"), "fifteenth": (15, "th"),
    "sixteenth": (16, "th"), "seventeenth": (17, "th"), "eighteenth": (18, "th"),
    "nineteenth": (19, "th"), "twentieth": (20, "th"), "thirtieth": (30, "th"),
    "fortieth": (40, "th"), "fiftieth": (50, "th"), "sixtieth": (60, "th"),
    "seventieth": (70, "th"), "eightieth": (80, "th"), "ninetieth": (90, "th"),
}

# Combine all recognizable number words for quick lookup.
_ALL_NUM_WORDS = set(_UNITS.keys()) | set(_TENS.keys()) | set(_SCALES.keys()) | set(_ORDINALS.keys())


class NormalizedToken(NamedTuple):
    """A normalized token and its mapping to the original words."""
    norm_text: str
    orig_indices: List[int]
    orig_words: List[str]


def _clean_word(word: str) -> str:
    """Lowercase, strip punctuation, normalize apostrophes."""
    w = word.lower()
    # Remove commas from numbers (e.g., "1,000" -> "1000")
    if re.match(r'^\d{1,3}(,\d{3})+$', w):
        w = w.replace(',', '')
    
    # Strip all punctuation except internal apostrophes.
    # We'll just remove all punctuation for maximum robustness.
    # E.g., "don't" -> "dont", "hello," -> "hello"
    w = re.sub(r'[^\w\s]', '', w)
    return w.strip()


def _parse_number_sequence(words: List[str]) -> Tuple[str, int]:
    """Parse a sequence of number words starting at index 0.
    
    Returns:
        (string representation of the number, number of words consumed)
    """
    current_val = 0
    total_val = 0
    consumed = 0
    suffix = ""
    
    for i, w in enumerate(words):
        if w in _UNITS:
            current_val += _UNITS[w]
            consumed += 1
        elif w in _TENS:
            current_val += _TENS[w]
            consumed += 1
        elif w in _SCALES:
            if current_val == 0:
                current_val = 1
            current_val *= _SCALES[w]
            if w in ["thousand", "million", "billion"]:
                total_val += current_val
                current_val = 0
            consumed += 1
        elif w in _ORDINALS:
            val, suf = _ORDINALS[w]
            current_val += val
            suffix = suf
            consumed += 1
            break
        elif w == "and" and (current_val > 0 or total_val > 0):
            # "one hundred and five" -> valid.
            # But don't consume 'and' if it's the last word or not followed by a number.
            if i + 1 < len(words) and words[i+1] in _ALL_NUM_WORDS:
                consumed += 1
            else:
                break
        else:
            break
            
    if consumed == 0:
        return "", 0
        
    final_val = total_val + current_val
    return f"{final_val}{suffix}", consumed


def normalize_sequence(words: List[str]) -> List[NormalizedToken]:
    """Normalize a sequence of words, combining multi-word numbers.
    
    Args:
        words: List of original display words (e.g., from EngineResult.words)
        
    Returns:
        List of NormalizedToken objects tracking original indices.
    """
    cleaned_words = []
    for w in words:
        cleaned = _clean_word(w)
        # If the word is entirely whitespace or empty after cleaning, it's skipped
        # However, for alignment we might want to keep placeholders if necessary,
        # but STT engines usually output distinct tokens. We'll skip completely empty.
        cleaned_words.append(cleaned)
        
    result = []
    i = 0
    while i < len(cleaned_words):
        cw = cleaned_words[i]
        if not cw:
            i += 1
            continue
            
        # Try to parse a sequence of number words.
        if cw in _ALL_NUM_WORDS:
            num_str, consumed = _parse_number_sequence(cleaned_words[i:])
            if consumed > 0:
                orig_indices = list(range(i, i + consumed))
                orig_words_sub = words[i:i + consumed]
                result.append(NormalizedToken(num_str, orig_indices, orig_words_sub))
                i += consumed
                continue
                
        # Digits with ordinal suffixes (e.g., "15th" -> "15th")
        # Digits without suffixes (e.g., "15" -> "15") are already handled by keeping cw.
        
        result.append(NormalizedToken(cw, [i], [words[i]]))
        i += 1
        
    return result
