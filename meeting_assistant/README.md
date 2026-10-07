# Meeting Assistant - Speech-To-Text (STT) Stage

Part of the Multimodal AI Meeting Assistant for Inter IIT Tech Meet Bootcamp.

## Architecture Overview

The assistant transforms recorded meeting audio into structured, reliable meeting minutes:
1. **Audio Validation & Preprocessing (This Stage)**: Validates audio files, normalizes loudness and channels to 16 kHz mono WAV, and detects silence points for seamless chunking.
2. **Consensus Lattice STT**: Multi-engine transcription combining ElevenLabs Scribe v2 and faster-whisper with deterministic ROVER alignment and uncertainty mapping.
3. **LLM-1 (Domain-aware Refinement)**: Corrects terminology in disputed spans without altering verbatim meaning.
4. **LLM-2 (Meeting Minutes & Action Items)**: Extracts summaries, decisions, and actionable tasks with explicit speaker attribution.
5. **Interactive UI & Exports**: Live audio review, transcript inspection, and document export.

---

## Step 1: Foundation Components

- `stt/schema.py`: Pydantic models for word tokens (`Word`), segments (`Segment`), engine responses (`EngineResult`), disputed spans (`UncertainSpan`), and consolidated outputs (`STTResult`).
- `stt/errors.py`: Safe, user-friendly domain exceptions (`UnsupportedFormatError`, `EmptyFileError`, `CorruptAudioError`, `NoAudioStreamError`, etc.).
- `stt/validator.py`: Strict audio inspection via `ffprobe`, verifying existence, allowlisted formats, decodability, presence of audio tracks, and duration limits.
- `stt/audio.py`: Loudness normalization using ffmpeg's `loudnorm` filter (16 kHz mono WAV) and silence-based chunking (`split_on_silence`).
- `config.py`: Environment-driven configuration via Pydantic settings and `.env`.

---

## Setup & Running Tests

### 1. Requirements
- Python 3.10+
- FFmpeg and FFprobe installed on system PATH (or auto-detected)

### 2. Installation
```bash
cd meeting_assistant
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 4. Running Tests
Run pytest from the `meeting_assistant` directory:
```bash
python -m pytest tests/ -v
```

## CLI Usage

### Run the STT Pipeline
Run the complete STT pipeline on an audio file to generate the transcript, JSON result, and uncertainty report.
```bash
python scripts/run_stt.py path/to/audio.wav --out outputs/
```

### Evaluation
Evaluate the pipeline's Word Error Rate (WER) against a reference transcript:
```bash
python scripts/eval_wer.py ref.txt outputs/stt_result.json
```
