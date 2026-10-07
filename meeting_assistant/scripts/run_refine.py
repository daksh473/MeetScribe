#!/usr/bin/env python
"""CLI to run LLM-1 Refinement on STTResults.

Usage:
    python scripts/run_refine.py outputs/stt_result.json --out outputs/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from stt.schema import STTResult
from llm.client import LLMClient
from refine import refine


def main():
    parser = argparse.ArgumentParser(description="Refine STT outputs using LLM.")
    parser.add_argument("stt_result", help="Path to stt_result.json")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    stt_path = Path(args.stt_result)
    out_dir = Path(args.out)

    if not stt_path.exists():
        print(f"Error: {stt_path} not found.", file=sys.stderr)
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    console = Console()

    console.print(f"[cyan]Loading {stt_path}...[/cyan]")
    stt_json = stt_path.read_text(encoding="utf-8")
    stt_result = STTResult.model_validate_json(stt_json)
    
    console.print("[cyan]Initializing LLM Client...[/cyan]")
    client = LLMClient()
    
    console.print("[cyan]Running Refinement (this may take a moment)...[/cyan]")
    
    # refine catches LLM exceptions and appends to flags to prevent crash when completely unavailable
    ref_result = refine(stt_result, client)
    
    if any("llm_refinement_failed" in f for f in ref_result.flags):
        console.print("[yellow]WARNING: LLM refinement failed. The original consensus transcript is preserved.[/yellow]")
        
    if "edit_budget_exceeded" in ref_result.flags:
        console.print("[yellow]WARNING: Edit budget exceeded. All edits were rejected.[/yellow]")

    console.print("\n[green]Refinement complete! Writing outputs...[/green]")
    
    # 1. refined_transcript.txt
    txt_path = out_dir / "refined_transcript.txt"
    txt_path.write_text(ref_result.refined_text, encoding="utf-8")
    
    # 2. refinement_audit.json
    audit_path = out_dir / "refinement_audit.json"
    audit_path.write_text(ref_result.model_dump_json(indent=2), encoding="utf-8")
    
    # 3. refinement_report.md
    md_path = out_dir / "refinement_report.md"
    
    md_lines = [
        "# STT Refinement Report",
        "",
        "## Summary",
        f"- **Accepted Edits**: {ref_result.stats.counts['accepted']}",
        f"- **Rejected Edits**: {ref_result.stats.counts['rejected']}",
        f"- **Unresolved Spans**: {ref_result.stats.counts['unresolved']}",
        f"- **Glossary Unifications**: {ref_result.stats.counts['unifications']}",
        f"- **Edit Ratio**: {ref_result.stats.edit_ratio:.2%}",
        "",
    ]
    
    if ref_result.flags:
        md_lines.append("## Flags & Warnings")
        for f in ref_result.flags:
            md_lines.append(f"- {f}")
        md_lines.append("")
        
    if ref_result.edits:
        md_lines.append("## Accepted Edits")
        for e in ref_result.edits:
            md_lines.append(f"- **{e.span_id}**: Chose `{e.choice}` -> \"{e.final_text}\"")
            md_lines.append(f"  *Reason*: {e.reason}")
        md_lines.append("")
        
    if ref_result.rejected_edits:
        md_lines.append("## Rejected Edits")
        for e in ref_result.rejected_edits:
            md_lines.append(f"- **{e.span_id}**: Attempted `{e.choice}` -> \"{e.final_text}\"")
            md_lines.append(f"  *Reason*: {e.reason}")
            
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    console.print(f"Results saved to [cyan]{out_dir}[/cyan]")


if __name__ == "__main__":
    main()
