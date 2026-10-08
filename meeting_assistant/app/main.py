import asyncio
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt.validator import validate_audio
from stt.pipeline import run_stt
from stt.errors import STTError
from refine.refiner import refine
from llm.client import LLMClient
from minutes.segmenter import segment_utterances
from minutes.annotator import annotate_segment
from minutes.compiler import compile_ledger
from minutes.verifier import verify_ledger
from minutes.writer import write_minutes
from llm.schemas import MeetingMetadata
from config import settings

app = FastAPI(title="MeetScribe API")

# Security: CORS off by default, but let's allow localhost for dev if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = Path("outputs/jobs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job state. In prod, use DB + Redis/Celery.
JOBS: Dict[str, Dict[str, Any]] = {}

class JobResponse(BaseModel):
    job_id: str

import logging
from logging.handlers import RotatingFileHandler
import re

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("app")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1024*1024, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

STAGE_RANGES = {
    "queued": (0.0, 0.0),
    "validating": (0.0, 0.05),
    "transcribing": (0.05, 0.5),
    "consensus": (0.5, 0.55),
    "refining": (0.55, 0.65),
    "documenting": (0.65, 0.8),
    "compiling": (0.8, 0.85),
    "verifying": (0.85, 0.95),
    "finalizing": (0.95, 1.0)
}

def _run_pipeline(job_id: str, audio_path: Path, out_dir: Path):
    job = JOBS[job_id]
    job.setdefault("messages", [])
    
    def cb(stage: str, frac: float, msg: str):
        job["stage"] = stage
        start, end = STAGE_RANGES.get(stage, (0.0, 1.0))
        job["progress"] = start + (end - start) * frac
        
        if msg and (not job["messages"] or job["messages"][-1] != msg):
            job["messages"].append(msg)
            job["message"] = msg
            job["stage_label"] = msg

    try:
        if os.environ.get("DEMO_FIXTURE") == "1":
            import time
            for stage, msg in [("validating", "Validating audio"), 
                               ("transcribing", "Transcribing..."),
                               ("consensus", "Building consensus"),
                               ("refining", "Refining transcript"),
                               ("documenting", "Generating minutes"),
                               ("verifying", "Verifying ledger"),
                               ("finalizing", "Finalizing")]:
                cb(stage, 0.5, msg)
                time.sleep(0.5)
                
            (out_dir / "raw_transcript.txt").write_text("Fake raw transcript")
            (out_dir / "refined_transcript.txt").write_text("Fake refined transcript")
            (out_dir / "meeting_record.md").write_text("# Fake Minutes")
            (out_dir / "meeting_record.json").write_text('{"summary": "Fake summary"}')
            (out_dir / "uncertainty_report.md").write_text("# Fake Uncertainty")
            
            job["status"] = "done"
            job["demo"] = True
            job["result_json"] = {
                "raw_transcript": {
                    "raw_text": "Hey everyone, let's talk about the new Kubernetes cluster. It costs like $50,000. Wait, no, maybe fifteen thousand? Anyway, John, please finish the migration by next Friday.",
                    "segments": [], "uncertain_spans": [], "flags": [], "metadata": {}
                },
                "demo": True
            }
            return

        # STT
        cb("transcribing", 0.0, "Starting STT engines")
        stt_res = run_stt(str(audio_path), progress_cb=cb)
        
        (out_dir / "raw_transcript.txt").write_text(stt_res.raw_text, encoding="utf-8")
        with open(out_dir / "uncertainty_report.md", "w", encoding="utf-8") as f:
            f.write("# Uncertainty Report\n")
            for s in stt_res.uncertain_spans:
                f.write(f"- {s.category} ({s.start:.1f}s-{s.end:.1f}s): {s.alternatives}\n")
                
        try:
            client = LLMClient()
        except Exception as e:
            job["warnings"].append(f"LLM Client Init Failed: {e}")
            raise RuntimeError(f"LLM Client init failed: {e}")

        cb("refining", 0.0, "Refining transcript via LLM-1")
        ref_res = refine(stt_res, client)
        (out_dir / "refined_transcript.txt").write_text(ref_res.refined_text, encoding="utf-8")
        
        cb("documenting", 0.0, "Extracting dialogue acts")
        segments = segment_utterances(ref_res.utterances, max_size=60)
        all_acts = []
        for i, seg in enumerate(segments):
            cb("documenting", i / max(1, len(segments)), f"Extracting acts ({i+1}/{len(segments)})")
            acts = annotate_segment(seg, client)
            all_acts.extend(acts)
                
        cb("compiling", 0.0, "Compiling ledger")
        decisions, tasks = compile_ledger(all_acts, ref_res.utterances)
        
        cb("verifying", 0.0, "Verifying ledger against evidence")
        v_decs, v_tasks, v_flags = verify_ledger(decisions, tasks, ref_res.utterances, stt_res.uncertain_spans, client)
        job["warnings"].extend(v_flags)
        
        cb("finalizing", 0.0, "Drafting final meeting minutes")
        meta = MeetingMetadata(
            models_used=[c["model"] for c in client.call_log],
            call_log_summary={"total_calls": len(client.call_log)},
            warnings=ref_res.flags + v_flags
        )
        record = write_minutes(v_decs, v_tasks, meta, client)
        
        from minutes.render import render_record
        render_record(record, out_dir)
        
        job["result_json"] = {
            "raw_transcript": stt_res.model_dump(),
            "refined_transcript": ref_res.model_dump(),
            "minutes": record.model_dump(),
            "metadata": meta.model_dump(),
            "demo": False
        }
        
        job["status"] = "done"
        cb("finalizing", 1.0, "Finished successfully")
        
    except Exception as e:
        import traceback
        import uuid
        error_id = str(uuid.uuid4())[:8]
        tb = traceback.format_exc()
        
        tb = re.sub(r'(sk-or-v1-[a-zA-Z0-9]{40,})', '***', tb)
        tb = re.sub(r'(sk-[a-zA-Z0-9]{32,})', '***', tb)
        tb = re.sub(r'(gsk_[a-zA-Z0-9]{20,})', '***', tb)
        tb = re.sub(r"('Authorization':\s*')Bearer\s+[^']+", r"\1Bearer ***", tb)
        
        stage_name = job.get("stage", "queued")
        logger.error(f"Pipeline error (ID: {error_id}) during {stage_name}: {e.__class__.__name__}\n{tb}")
        
        reason = str(e)

        # For STTError (which already lists all engine reasons), preserve the full message.
        # For other errors, apply minimal sanitization.
        if not isinstance(e, STTError):
            if "429" in reason:
                reason = "rate limit reached, retry later"
            elif "json" in reason.lower() or "validationerror" in reason.lower():
                reason = "model returned invalid JSON"
            elif not reason or reason == "None":
                reason = "internal error"
            
        # Never expose full API keys in error messages
        reason = re.sub(r'(sk-or-v1-[a-zA-Z0-9]{40,})', '***', reason)
        reason = re.sub(r'(sk-[a-zA-Z0-9]{32,})', '***', reason)
        reason = re.sub(r'(gsk_[a-zA-Z0-9]{20,})', '***', reason)
        reason = re.sub(r'(xi-[a-zA-Z0-9]{20,})', '***', reason)
            
        job["error"] = f"Failed during {stage_name}: {reason} (error id {error_id})"
        job["status"] = "failed"
        
        # Populate partial result
        res = {
            "metadata": {"warnings": [job["error"]], "file_name": Path(job["audio_path"]).name},
            "demo": False,
            "error": job["error"]
        }
        if 'stt_res' in locals(): res["raw_transcript"] = stt_res.model_dump()
        if 'ref_res' in locals(): res["refined_transcript"] = ref_res.model_dump()
        if 'record' in locals(): res["minutes"] = record.model_dump()
        job["result_json"] = res


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        
    job_id = str(uuid.uuid4())
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path = job_dir / file.filename
    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to save uploaded file"})
        
    try:
        # Validate synchronously before accepting job
        validate_audio(str(audio_path))
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"error": str(e)})
        
    JOBS[job_id] = {
        "status": "queued",
        "stage": "queued",
        "stage_label": "Waiting in queue...",
        "progress": 0.0,
        "message": "",
        "engines": [],
        "warnings": [],
        "error": None,
        "job_dir": str(job_dir),
        "audio_path": str(audio_path)
    }
    
    # Run in thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline, job_id, audio_path, job_dir)
    
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = JOBS[job_id]
    resp = {
        "status": job["status"],
        "stage": job["stage"],
        "stage_label": job["stage_label"],
        "progress": job["progress"],
        "message": job["message"],
        "messages": job.get("messages", []),
        "engines": job["engines"],
        "warnings": job["warnings"],
        "error": job["error"]
    }
    if job.get("demo"):
        resp["demo"] = True
    return resp


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = JOBS[job_id]
    if job["status"] not in ("done", "failed"):
        raise HTTPException(status_code=400, detail="Job is not done yet")
        
    resp = job.get("result_json", {})
    if job.get("demo"):
        resp["demo"] = True
    return resp


@app.get("/api/jobs/{job_id}/download/{kind}")
async def download_result(job_id: str, kind: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_dir = Path(JOBS[job_id]["job_dir"])
    
    file_map = {
        "raw_txt": "raw_transcript.txt",
        "refined_txt": "refined_transcript.txt",
        "record_md": "meeting_record.md",
        "record_json": "meeting_record.json",
        "uncertainty_md": "uncertainty_report.md"
    }
    
    if kind == "all_zip":
        zip_path = job_dir / "all_results.zip"
        if not zip_path.exists():
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in file_map.values():
                    fp = job_dir / f
                    if fp.exists():
                        zf.write(fp, f)
        return FileResponse(zip_path, media_type="application/zip", filename=f"meetscribe_{job_id}.zip")
        
    if kind not in file_map:
        raise HTTPException(status_code=404, detail="Unknown download kind")
        
    target_file = job_dir / file_map[kind]
    if not target_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(target_file)


@app.get("/api/jobs/{job_id}/audio")
async def get_audio(job_id: str, request: Request):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
        
    audio_path = Path(JOBS[job_id]["audio_path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    # FileResponse automatically handles HTTP Range requests
    return FileResponse(audio_path, media_type="audio/wav")

# Mount static files at /
import os
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
