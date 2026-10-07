#!/usr/bin/env python
"""End-to-End Orchestrator for MeetScribe.

Usage:
    python scripts/run_all.py path/to/audio.wav --out outputs/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from stt.pipeline import run_stt
from refine.refiner import refine
from llm.client import LLMClient
from minutes.segmenter import segment_utterances
from minutes.annotator import annotate_segment
from minutes.compiler import compile_ledger
from minutes.verifier import verify_ledger
from minutes.writer import write_minutes
from minutes.render import render_record
from llm.schemas import MeetingMetadata


def main():
    parser = argparse.ArgumentParser(description="Run complete MeetScribe E2E Pipeline.")
    parser.add_argument("audio", help="Path to input audio file")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out)
    
    if not audio_path.exists():
        print(f"Error: Audio file {audio_path} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    
    def cb(stage: str, frac: float, msg: str):
        console.print(f"[dim]{stage}[/dim] {msg} ({frac:.0%})")
        
    # STT
    console.print("\n[bold cyan]=== Phase 1: STT & Consensus ===[/bold cyan]")
    stt_res = run_stt(str(audio_path), progress_cb=cb)
    
    (out_dir / "raw_transcript.txt").write_text(stt_res.raw_text, encoding="utf-8")
    
    with open(out_dir / "uncertainty_report.md", "w", encoding="utf-8") as f:
        f.write("# Uncertainty Report\n")
        for s in stt_res.uncertain_spans:
            f.write(f"- {s.category} ({s.start:.1f}s-{s.end:.1f}s): {s.alternatives}\n")
            
    # Init LLM
    try:
        client = LLMClient()
    except Exception as e:
        console.print(f"[bold red]LLM Client Init Failed: {e}[/bold red]")
        console.print("[yellow]Continuing with STT results only.[/yellow]")
        sys.exit(0)
        
    # Refine
    console.print("\n[bold cyan]=== Phase 2: LLM-1 Refinement ===[/bold cyan]")
    ref_res = refine(stt_res, client)
    
    (out_dir / "refined_transcript.txt").write_text(ref_res.refined_text, encoding="utf-8")
    (out_dir / "refinement_audit.json").write_text(ref_res.model_dump_json(indent=2), encoding="utf-8")
    
    if any("llm_refinement_failed" in f for f in ref_res.flags):
        console.print("[yellow]LLM refinement failed. Proceeding with unrefined STT consensus text.[/yellow]")
        
    # Minutes Pipeline
    console.print("\n[bold cyan]=== Phase 3: LLM-2 Ledger & Minutes ===[/bold cyan]")
    
    # 1. Segment & Annotate
    console.print("[cyan]Annotating segments...[/cyan]")
    segments = segment_utterances(ref_res.utterances, max_size=60)
    all_acts = []
    annotator_failed = False
    
    for i, seg in enumerate(segments):
        console.print(f"  Segment {i+1}/{len(segments)}...")
        try:
            acts = annotate_segment(seg, client)
            all_acts.extend(acts)
        except Exception as e:
            console.print(f"[yellow]Annotator failed on segment {i+1}: {e}[/yellow]")
            annotator_failed = True
            
    if annotator_failed and not all_acts:
        console.print("[bold red]Annotation failed completely. Stopping.[/bold red]")
        sys.exit(0)
        
    # 2. Compile
    console.print("[cyan]Compiling ledger...[/cyan]")
    decisions, tasks = compile_ledger(all_acts, ref_res.utterances)
    
    # 3. Verify
    console.print("[cyan]Verifying ledger items...[/cyan]")
    v_decs, v_tasks, v_flags = verify_ledger(decisions, tasks, ref_res.utterances, stt_res.uncertain_spans, client)
    
    # 4. Write
    console.print("[cyan]Drafting minutes...[/cyan]")
    meta = MeetingMetadata(
        models_used=[c["model"] for c in client.call_log],
        call_log_summary={"total_calls": len(client.call_log)},
        warnings=ref_res.flags + v_flags
    )
    
    record = write_minutes(v_decs, v_tasks, meta, client)
    
    # 5. Render
    render_record(record, out_dir)
    console.print(f"\n[bold green]Success! All files written to {out_dir}[/bold green]")


if __name__ == "__main__":
    main()
