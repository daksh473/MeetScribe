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
## CLI Usage

### Complete End-to-End Pipeline
Run the full transcription, refinement, ledger compilation, and meeting summary pipeline:
```bash
python scripts/run_all.py path/to/audio.wav --out outputs/
```
This generates 6 files in `outputs/`:
1. `raw_transcript.txt`: The unrefined consensus transcript.
2. `uncertainty_report.md`: Spans where STT engines disagreed.
3. `refined_transcript.txt`: The transcript after LLM-1 refinement.
4. `refinement_audit.json`: Log of all accepted/rejected LLM-1 text edits.
5. `meeting_record.md`: The human-readable summary, minutes, and verified ledger.
6. `meeting_record.json`: The machine-readable version of the meeting record.

*(Note: If the LLM goes down or fails safety checks, the system degrades gracefully and preserves earlier stages automatically.)*

### Run only STT (Stage 1)
```bash
python scripts/run_stt.py path/to/audio.wav --out outputs/
```

### Evaluation
Evaluate Word Error Rate (WER) against a reference text:
```bash
python scripts/eval_wer.py ref.txt outputs/stt_result.json
```

Evaluate Ledger outputs against a ground truth JSON:
```bash
python scripts/eval_ledger.py samples/ground_truth/test_gt_1.json outputs/meeting_record.json
```

## Setup Instructions

### 1. Requirements
- Python 3.10+
- FFmpeg and FFprobe installed on system PATH (or auto-detected)

### 2. Installation
```bash
cd meeting_assistant
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in the API keys yourself. Never commit your `.env` file!

#### LLM Configuration & Setup
MeetScribe uses robust LLM logic requiring external providers.
- **API Keys**: Configure `GROQ_API_KEY` and `OPENROUTER_API_KEY` in `.env`. (GROQ is required for LLM stages, OpenRouter is a fallback).
- **Model IDs**: Model names shift rapidly. Always verify the exact model ID strings (e.g., `llama3-70b-8192`) on the provider's live dashboard/docs.
- **Rate Limits**: Free-tier accounts have strict requests-per-minute (RPM) limits. Adjust `LLM_RATE_LIMIT_RPM` to prevent failures.
- **Privacy Caveat**: Audio transcripts are sent to third-party APIs for processing. Do NOT use this tool for highly confidential or proprietary meetings unless you have signed data privacy agreements with your chosen LLM providers.

### 4. Setup Checks & Secrets Scanning
Before running a demo or committing code, run the setup checker to verify dependencies and API keys securely (this will not print your full keys):
```bash
python scripts/check_setup.py
```

To prevent accidental API key leaks, install the secrets scanner as a git pre-commit hook. Run this from the project root:
```bash
cat << 'EOF' > .git/hooks/pre-commit
#!/bin/sh
python meeting_assistant/scripts/scan_secrets.py
EOF
chmod +x .git/hooks/pre-commit
```

### 5. Running Tests
Run pytest from the `meeting_assistant` directory to ensure core logic is intact (works without API keys):
```bash
python -m pytest tests/ -v
```

## Deployment Notes
**Streamlit Community Cloud:** If you are deploying this as a Streamlit app to the Community Cloud, put your API keys in the app's Secrets settings online, not in the repository. The application will securely read from `st.secrets` if available. Do NOT create `.streamlit/secrets.toml` with real values locally if you intend to commit it.

## API Server
Run the web API with: uvicorn app.main:app --reload
