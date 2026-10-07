"""Core LLM-1 Refinement logic."""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from llm.client import LLMClient
from llm.schemas import Edit, RefinementResult, RefinementStats
from stt.schema import STTResult, UncertainSpan
from .glossary import find_inconsistent_spellings
from .guards import reject_edit
from .prompts import REFINE_SYSTEM_PROMPT, build_refine_user_prompt
from .utterances import build_utterances


class LLMSpanResolution(BaseModel):
    span_id: str
    choice: Literal["ENGINE_A", "ENGINE_B", "UNRESOLVED"]
    reason: str


class LLMRefinementResponse(BaseModel):
    resolutions: List[LLMSpanResolution]


def _get_span_original_text(stt_result: STTResult, span: UncertainSpan) -> str:
    """Extract the original consensus text for the span based on timestamps."""
    if not stt_result.segments or not stt_result.segments[0].words:
        return ""
        
    span_words = [
        w.text for w in stt_result.segments[0].words 
        if w.start >= span.start - 0.01 and w.end <= span.end + 0.01
    ]
    return " ".join(span_words)


def refine(stt_result: STTResult, client: LLMClient, max_edit_ratio: float = 0.03) -> RefinementResult:
    """Refine transcript using LLM-1 for CRITICAL/HIGH spans."""
    
    utterances = build_utterances(stt_result)
    
    # 1. Filter spans
    target_spans = [s for s in stt_result.uncertain_spans if s.category in {"CRITICAL", "HIGH"}]
    
    accepted_edits: List[Edit] = []
    rejected_edits: List[Edit] = []
    flags = list(stt_result.flags)
    
    # We will track which span_id maps to which span
    span_map = {}
    spans_data = []
    
    for i, span in enumerate(target_spans):
        span_id = f"span_{i}"
        span_map[span_id] = span
        
        # We need a stable assignment of ENGINE_A and ENGINE_B
        engines = list(span.alternatives.keys())
        eng_a = engines[0] if len(engines) > 0 else "unknown"
        eng_b = engines[1] if len(engines) > 1 else "unknown"
        
        spans_data.append({
            "span_id": span_id,
            "context_before": span.context_before,
            "context_after": span.context_after,
            "alternatives": {
                "ENGINE_A": span.alternatives.get(eng_a, ""),
                "ENGINE_B": span.alternatives.get(eng_b, ""),
            },
            "engines": {"ENGINE_A": eng_a, "ENGINE_B": eng_b}
        })
        
    total_words = len(stt_result.raw_text.split()) if stt_result.raw_text else 1
    total_changed_words = 0
    
    if spans_data:
        # 2. Call LLM (Batched if large, but for now single call for simplicity in tests)
        user_prompt = build_refine_user_prompt(spans_data)
        
        try:
            response = client.complete_json(
                system=REFINE_SYSTEM_PROMPT,
                user=user_prompt,
                schema=LLMRefinementResponse,
                temperature=0.0
            )
            
            # 3. Process resolutions and Guards
            for res in response.resolutions:
                if res.span_id not in span_map:
                    continue
                    
                span = span_map[res.span_id]
                s_data = next(s for s in spans_data if s["span_id"] == res.span_id)
                
                original_text = _get_span_original_text(stt_result, span)
                
                if res.choice == "UNRESOLVED":
                    flags.append(f"unresolved_span_{res.span_id}")
                    continue
                    
                final_text = s_data["alternatives"].get(res.choice, "")
                
                # Check guards
                engine_alts = list(span.alternatives.values())
                rejection_reason = reject_edit(original_text, final_text, engine_alts)
                
                edit = Edit(
                    span_id=res.span_id,
                    choice=res.choice,
                    final_text=final_text,
                    reason=res.reason
                )
                
                if rejection_reason:
                    edit.reason = f"Rejected: {rejection_reason} | LLM Reason: {edit.reason}"
                    rejected_edits.append(edit)
                else:
                    accepted_edits.append(edit)
                    total_changed_words += len(final_text.split())
                    
        except Exception as e:
            flags.append(f"llm_refinement_failed: {e}")
            
    # 4. Edit Budget Check
    if (total_changed_words / total_words) > max_edit_ratio:
        flags.append("edit_budget_exceeded")
        rejected_edits.extend(accepted_edits)
        for e in accepted_edits:
            e.reason = f"Rejected: Edit budget exceeded | {e.reason}"
        accepted_edits = []
        
    # 5. Apply Edits to recreate text
    # We will reconstruct raw_text by substituting the original texts of the spans with the accepted edits
    refined_text = stt_result.raw_text
    if accepted_edits:
        # Reverse order to avoid index shifting if we used string replacement
        # But we'll just use a simple replace based on context for robustness
        for edit in accepted_edits:
            span = span_map[edit.span_id]
            original_text = _get_span_original_text(stt_result, span)
            
            if not original_text.strip():
                continue
                
            # Find the exact occurrence using context
            pattern = re.escape(span.context_before.strip()) + r"\s*" + re.escape(original_text) + r"\s*" + re.escape(span.context_after.strip())
            
            def replacer(match):
                return match.group(0).replace(original_text, edit.final_text, 1)
                
            refined_text = re.sub(pattern, replacer, refined_text, count=1)

    # 6. Glossary Unification
    glossary_flags, unifications = find_inconsistent_spellings(refined_text, stt_result.uncertain_spans)
    flags.extend(glossary_flags)
    
    for unif in unifications:
        # Simple whole word replace for minority spelling
        refined_text = re.sub(r'\b' + re.escape(unif['old']) + r'\b', unif['new'], refined_text)
        
    stats = RefinementStats(
        edit_ratio=(total_changed_words / total_words),
        counts={
            "accepted": len(accepted_edits),
            "rejected": len(rejected_edits),
            "unresolved": len([f for f in flags if f.startswith("unresolved_span")]),
            "unifications": len(unifications)
        }
    )
    
    return RefinementResult(
        refined_text=refined_text,
        utterances=utterances,
        edits=accepted_edits,
        rejected_edits=rejected_edits,
        flags=flags,
        stats=stats
    )
