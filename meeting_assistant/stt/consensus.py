"""ROVER-style consensus alignment for STT outputs.

Algorithm Overview:
This module combines multiple STT engine results into a single consensus transcript.
1. If only one engine succeeds, its output is used directly.
2. For multiple engines, it normalizes the text (lowercasing, number expansion,
   punctuation stripping) and aligns the sequences progressively.
3. Alignment primarily relies on word timestamp overlaps. If words from different
   engines overlap in time, they are aligned. If they don't overlap, edit distance
   on the normalized text acts as a fallback to keep the sequences aligned.
4. Once aligned into "slots", engines vote on the best word for each slot.
   Voting is weighted by a configurable engine prior (e.g., Scribe > Whisper)
   plus local word confidence.
5. Slots where the normalized text differs across engines are marked as DISPUTED.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

from .schema import EngineResult, Word
from .text_norm import NormalizedToken, normalize_sequence


# Configurable prior weights for voting.
# We give elevenlabs (Scribe) a slight edge over whisper if confidence is missing/tied.
ENGINE_PRIORS = {
    "elevenlabs": 1.1,
    "whisper": 1.0,
}


class AlignmentSlot(NamedTuple):
    """A single position in the aligned sequence containing each engine's token."""
    engine_tokens: Dict[str, Optional[NormalizedToken]]
    
    @property
    def is_disputed(self) -> bool:
        """True if engines disagree on the normalized text."""
        texts = set()
        for tok in self.engine_tokens.values():
            if tok is not None:
                texts.add(tok.norm_text)
            else:
                texts.add(None)
        # If there's more than one unique normalized text (including None/gap), it's disputed.
        # But wait, if one engine failed, it shouldn't be here. All engines here succeeded.
        return len(texts) > 1


class ConsensusResult(NamedTuple):
    """The final consensus output."""
    words: List[Word]
    disputed_slots: List[AlignmentSlot]
    flags: List[str]


def _time_overlap(tok1: NormalizedToken, tok2: NormalizedToken, words1: List[Word], words2: List[Word]) -> bool:
    """Check if the time ranges of two normalized tokens overlap."""
    if not tok1.orig_indices or not tok2.orig_indices:
        return False
        
    start1 = words1[tok1.orig_indices[0]].start
    end1 = words1[tok1.orig_indices[-1]].end
    
    start2 = words2[tok2.orig_indices[0]].start
    end2 = words2[tok2.orig_indices[-1]].end
    
    # Overlap condition: start of A < end of B AND start of B < end of A
    return start1 < end2 and start2 < end1


def _align_pairwise(
    seq1: List[NormalizedToken],
    seq2: List[NormalizedToken],
    words1: List[Word],
    words2: List[Word]
) -> List[Tuple[Optional[NormalizedToken], Optional[NormalizedToken]]]:
    """Needleman-Wunsch pairwise alignment prioritizing time overlap."""
    n = len(seq1)
    m = len(seq2)
    
    # dp[i][j] = (cost, operations path)
    # operations: 0 = match/sub, 1 = insert (gap in seq1), 2 = delete (gap in seq2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialization
    for i in range(1, n + 1):
        dp[i][0] = i * 1.0  # cost of delete
        ptr[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = j * 1.0  # cost of insert
        ptr[0][j] = 1
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            tok1 = seq1[i - 1]
            tok2 = seq2[j - 1]
            
            # Match / Substitution cost
            if tok1.norm_text == tok2.norm_text:
                match_cost = 0.0
            else:
                if _time_overlap(tok1, tok2, words1, words2):
                    match_cost = 0.5  # Time overlaps, likely a substitution
                else:
                    match_cost = 2.0  # No time overlap, discourage substitution
                    
            cost_sub = dp[i - 1][j - 1] + match_cost
            cost_ins = dp[i][j - 1] + 1.0
            cost_del = dp[i - 1][j] + 1.0
            
            # Find minimum cost
            min_cost = min(cost_sub, cost_ins, cost_del)
            dp[i][j] = min_cost
            
            if min_cost == cost_sub:
                ptr[i][j] = 0
            elif min_cost == cost_del:
                ptr[i][j] = 2
            else:
                ptr[i][j] = 1

    # Backtrack
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if ptr[i][j] == 0:
            alignment.append((seq1[i - 1], seq2[j - 1]))
            i -= 1
            j -= 1
        elif ptr[i][j] == 2:
            alignment.append((seq1[i - 1], None))
            i -= 1
        else:
            alignment.append((None, seq2[j - 1]))
            j -= 1
            
    alignment.reverse()
    return alignment


def build_consensus(results: List[EngineResult]) -> ConsensusResult:
    """Build consensus from N engine results."""
    # Filter to successful results
    success_results = [r for r in results if r.success]
    
    if not success_results:
        # All engines failed, return empty consensus
        return ConsensusResult([], [], ["all_engines_failed"])
        
    if len(success_results) == 1:
        # Single engine success
        res = success_results[0]
        return ConsensusResult(res.words, [], ["single_engine"])
        
    # Multi-engine alignment (progressive alignment)
    # We'll align engine 2 to engine 1, then engine 3 to the slot array, etc.
    # For now, implemented for 2 engines as it's the primary use case.
    res1 = success_results[0]
    res2 = success_results[1]
    
    seq1 = normalize_sequence([w.text for w in res1.words])
    seq2 = normalize_sequence([w.text for w in res2.words])
    
    alignment = _align_pairwise(seq1, seq2, res1.words, res2.words)
    
    slots = []
    for tok1, tok2 in alignment:
        slots.append(AlignmentSlot(
            engine_tokens={
                res1.engine_name: tok1,
                res2.engine_name: tok2,
            }
        ))
        
    consensus_words = []
    disputed_slots = []
    
    for slot in slots:
        if slot.is_disputed:
            disputed_slots.append(slot)
            
        # Voting
        best_word = None
        best_score = -1.0
        
        for engine_name, res in [(res1.engine_name, res1), (res2.engine_name, res2)]:
            tok = slot.engine_tokens.get(engine_name)
            if not tok:
                continue
                
            prior = ENGINE_PRIORS.get(engine_name, 1.0)
            
            # Compute average confidence for the normalized token
            conf_sum = 0.0
            conf_count = 0
            for idx in tok.orig_indices:
                c = res.words[idx].confidence
                if c is not None:
                    conf_sum += c
                    conf_count += 1
            
            local_conf = (conf_sum / conf_count) if conf_count > 0 else 1.0
            
            score = prior + local_conf
            
            if score > best_score:
                best_score = score
                
                # Construct the display word(s)
                # We merge the original words into a single Word object for the consensus
                # Or we can keep them separate? "Output: consensus word list (original display tokens, with start/end)"
                # If a token spans multiple words, we just take the first word's start and last word's end, and join text.
                start = res.words[tok.orig_indices[0]].start
                end = res.words[tok.orig_indices[-1]].end
                text = " ".join([res.words[i].text for i in tok.orig_indices])
                speaker = res.words[tok.orig_indices[0]].speaker
                
                best_word = Word(
                    text=text,
                    start=start,
                    end=end,
                    confidence=local_conf,
                    speaker=speaker
                )
                
        if best_word:
            consensus_words.append(best_word)
            
    return ConsensusResult(consensus_words, disputed_slots, [])
