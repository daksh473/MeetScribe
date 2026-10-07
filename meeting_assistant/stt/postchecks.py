"""Post-checks for STT pipeline.

Inspects the final consensus transcript and adds warning flags for:
- Suspiciously short transcripts (low WPM).
- Known Whisper hallucinations.
- Repeated phrases.
- Non-English character predominance.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from .schema import STTResult

# Common hallucination phrases from Whisper when encountering silence or noise
HALLUCINATIONS = {
    "thanks for watching",
    "subscribe",
    "thank you for watching",
    "please subscribe",
    "don't forget to subscribe",
    "amara.org",
    "by the way",
}


def _check_hallucinations(text: str) -> bool:
    """True if known hallucination phrases are detected."""
    lower_text = text.lower()
    for phrase in HALLUCINATIONS:
        if phrase in lower_text:
            return True
    return False


def _check_repeated_phrases(text: str) -> bool:
    """True if a phrase of 3+ words is repeated 3+ times consecutively."""
    words = text.split()
    if len(words) < 9:
        return False
        
    # Check for consecutive unigrams, bigrams, or trigrams repeated 3+ times
    for n in range(1, 10):
        for i in range(len(words) - 3 * n):
            phrase1 = " ".join(words[i:i + n])
            phrase2 = " ".join(words[i + n:i + 2 * n])
            phrase3 = " ".join(words[i + 2 * n:i + 3 * n])
            if phrase1 == phrase2 == phrase3:
                return True
    return False


def _check_wpm(text: str, duration_s: float, min_wpm: float = 30.0) -> bool:
    """True if words per minute is suspiciously low (below min_wpm)."""
    if duration_s < 10.0:
        return False  # Too short to reliably judge WPM
        
    word_count = len(text.split())
    minutes = duration_s / 60.0
    wpm = word_count / minutes
    return wpm < min_wpm


def _check_non_english(text: str) -> bool:
    """True if a large portion of characters are non-ASCII/non-Latin."""
    if not text:
        return False
        
    # Count basic Latin letters vs other letters
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    non_latin_chars = len(re.findall(r'[^\x00-\x7F\s\d\.,!\?\'\"-]', text))
    
    total_letters = latin_chars + non_latin_chars
    if total_letters == 0:
        return False
        
    return (non_latin_chars / total_letters) > 0.2  # >20% non-Latin letters


def run_postchecks(result: STTResult) -> None:
    """Run all post-checks and append to result.flags in-place."""
    if not result.raw_text:
        return
        
    if _check_hallucinations(result.raw_text):
        result.flags.append("possible_hallucination")
        
    if _check_repeated_phrases(result.raw_text):
        result.flags.append("repeated_phrases")
        
    if _check_wpm(result.raw_text, result.metadata.audio_duration_s):
        result.flags.append("low_wpm")
        
    if _check_non_english(result.raw_text):
        result.flags.append("non_english_suspected")
