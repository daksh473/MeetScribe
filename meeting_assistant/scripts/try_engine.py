#!/usr/bin/env python
"""Quick manual test: run a single STT engine on one audio file.

Usage
-----
    python scripts/try_engine.py --engine whisper --file samples/x.wav
    python scripts/try_engine.py --engine elevenlabs --file samples/x.wav

The transcript and per-word timings are printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``stt`` can be imported
# regardless of from where the script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stt.engines.scribe import ScribeEngine
from stt.engines.whisper_local import WhisperLocalEngine


_ENGINES = {
    "whisper": WhisperLocalEngine,
    "elevenlabs": ScribeEngine,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single STT engine on an audio file and print the result."
    )
    parser.add_argument(
        "--engine",
        choices=list(_ENGINES.keys()),
        default="whisper",
        help="Which engine to use (default: whisper).",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the audio file (WAV or any ffmpeg-compatible format).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Global timeline offset in seconds (default: 0).",
    )
    args = parser.parse_args()

    audio_path = Path(args.file).resolve()
    if not audio_path.is_file():
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    engine = _ENGINES[args.engine]()
    print(f"Engine : {engine.name}")
    print(f"File   : {audio_path}")
    print(f"Offset : {args.offset}s")
    print("-" * 60)

    result = engine.transcribe(str(audio_path), offset_s=args.offset)

    if not result.success:
        print(f"FAILED: {result.error}", file=sys.stderr)
        sys.exit(2)

    print(f"\n{'='*60}")
    print("TRANSCRIPT")
    print(f"{'='*60}")
    print(result.text)
    print(f"\n{'='*60}")
    print(f"WORDS  ({len(result.words)} total)")
    print(f"{'='*60}")
    for w in result.words:
        spk = f"  [{w.speaker}]" if w.speaker else ""
        conf = f"  conf={w.confidence:.3f}" if w.confidence is not None else ""
        print(f"  [{w.start:8.2f}s → {w.end:8.2f}s]  {w.text}{spk}{conf}")

    print(f"\nProcessing time: {result.processing_time_s:.2f}s")
    print(f"Segments: {len(result.segments)}")


if __name__ == "__main__":
    main()
