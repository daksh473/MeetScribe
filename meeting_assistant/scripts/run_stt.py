#!/usr/bin/env python
"""CLI to run the full STT pipeline on an audio file.

Usage:
    python scripts/run_stt.py path/to/audio --out outputs/
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from stt import run_stt
from stt.errors import STTError
from stt.uncertainty import summarize


def main():
    parser = argparse.ArgumentParser(description="Run the MeetScribe STT pipeline.")
    parser.add_argument("audio", help="Path to the audio file.")
    parser.add_argument("--out", required=True, help="Output directory to save results.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out)

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=100)

        def progress_cb(stage: str, fraction: float, message: str):
            progress.update(task, completed=fraction * 100, description=f"[cyan]{message}")

        try:
            result = run_stt(str(audio_path), progress_cb=progress_cb)
        except STTError as e:
            progress.stop()
            console.print(f"\n[red]STT Error:[/red] {e}", style="bold")
            sys.exit(1)
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Unexpected Error:[/red] {e}", style="bold")
            sys.exit(1)

    # Write outputs
    console.print("\n[green]Transcription complete![/green] Writing outputs...")
    
    # 1. raw_transcript.txt
    raw_path = out_dir / "raw_transcript.txt"
    raw_path.write_text(result.raw_text, encoding="utf-8")
    
    # 2. stt_result.json
    json_path = out_dir / "stt_result.json"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    
    # 3. uncertainty_report.md
    md_path = out_dir / "uncertainty_report.md"
    summary_counts = summarize(result.uncertain_spans)
    
    md_lines = [
        "# STT Uncertainty Report",
        "",
        "## Summary",
        f"- **CRITICAL**: {summary_counts.get('CRITICAL', 0)}",
        f"- **HIGH**: {summary_counts.get('HIGH', 0)}",
        f"- **LOW**: {summary_counts.get('LOW', 0)}",
        "",
        "## Spans",
    ]
    
    for span in result.uncertain_spans:
        if span.category == "LOW":
            continue
            
        md_lines.append(f"### [{span.category}] {span.start:.2f}s - {span.end:.2f}s")
        md_lines.append(f"**Context**: ... {span.context_before} **[DISPUTED]** {span.context_after} ...")
        md_lines.append("")
        md_lines.append("**Alternatives**:")
        for eng, alt in span.alternatives.items():
            md_lines.append(f"- **{eng}**: {alt or '(gap)'}")
        md_lines.append("")
        
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    
    # Terminal summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Duration: {result.metadata.audio_duration_s:.1f}s")
    console.print(f"  Processing Time: {result.metadata.processing_time_s:.1f}s")
    console.print(f"  Engines Used: {', '.join(result.metadata.engines_used)}")
    console.print(f"  Flags: {', '.join(result.flags) if result.flags else 'None'}")
    console.print(f"  Critical Uncertainties: {summary_counts.get('CRITICAL', 0)}")
    
    console.print(f"\nResults saved to [cyan]{out_dir}[/cyan]")


if __name__ == "__main__":
    main()
