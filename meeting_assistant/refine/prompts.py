"""Prompts for LLM-1 Refinement."""

REFINE_SYSTEM_PROMPT = """You are an expert transcript editor. Your task is to resolve disputed audio transcription spans.
You are NOT a rewriter. You must ONLY resolve the DISPUTED spans provided to you.
Choose exactly one of: ENGINE_A, ENGINE_B, or UNRESOLVED.
For CRITICAL spans (numbers, dates, negation) you may only choose between the engine alternatives, never write new text.
Choose based on the surrounding context only; never use outside knowledge to invent names or numbers.
If neither alternative makes sense in context, choose UNRESOLVED. Do not guess.
"""

def build_refine_user_prompt(spans_data: list[dict]) -> str:
    """Build the user prompt string from a list of span data dicts."""
    lines = ["Please resolve the following transcription disputes:\n"]
    
    for span in spans_data:
        lines.append(f"Span ID: {span['span_id']}")
        lines.append(f"Context Before: ... {span['context_before']}")
        
        for engine, alt in span['alternatives'].items():
            lines.append(f"Alternative {engine.upper()}: {alt}")
            
        lines.append(f"Context After: {span['context_after']} ...\n")
        
    return "\n".join(lines)
