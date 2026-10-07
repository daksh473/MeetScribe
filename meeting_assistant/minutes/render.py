"""Render meeting records to Markdown and JSON."""

import json
from pathlib import Path

from llm.schemas import MeetingRecord


def render_record(record: MeetingRecord, out_dir: Path):
    """Render MeetingRecord to meeting_record.md and meeting_record.json."""
    
    # 1. JSON
    json_path = out_dir / "meeting_record.json"
    json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    
    # 2. Markdown
    md_path = out_dir / "meeting_record.md"
    
    lines = [
        "# Meeting Record",
        "",
        "## Summary",
        record.summary,
        "",
        "## Minutes",
    ]
    
    for sec in record.minutes:
        lines.append(f"### {sec.get('title', 'Section')}")
        lines.append(sec.get('content', ''))
        lines.append("")
        
    lines.append("## Ledger")
    
    if record.decisions:
        lines.append("### Decisions")
        for d in record.decisions:
            mark = " ⚠️" if d.low_confidence else ""
            lines.append(f"- **[{d.id}]** ({d.status}) {d.text}{mark}")
        lines.append("")
        
    if record.action_items:
        lines.append("### Action Items")
        for t in record.action_items:
            mark = " ⚠️" if t.low_confidence else ""
            lines.append(f"- **[{t.id}]** ({t.status}) {t.task} | **Owner**: {t.owner} | **Deadline**: {t.deadline}{mark}")
        lines.append("")
        
    md_path.write_text("\n".join(lines), encoding="utf-8")
