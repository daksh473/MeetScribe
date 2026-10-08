"""Main pipeline logic connecting all STT components."""

from __future__ import annotations

import concurrent.futures
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List

from config import settings
from stt.audio import normalize_audio, split_on_silence
from stt.consensus import build_consensus
from stt.engines.base import STTEngine
from stt.engines.scribe import ScribeEngine
from stt.engines.whisper_local import WhisperLocalEngine
from stt.errors import NoSpeechError, STTError
from stt.postchecks import run_postchecks
from stt.schema import EngineResult, STTMetadata, STTResult
from stt.uncertainty import generate_spans
from stt.validator import validate_audio


from stt.engines.groq_whisper import GroqWhisperEngine

def _get_engines() -> List[STTEngine]:
    """Initialize STT engines based on configuration."""
    engines: List[STTEngine] = []
    for eng_name in settings.engines_list:
        if eng_name == "elevenlabs":
            engines.append(ScribeEngine())
        elif eng_name == "whisper":
            engines.append(WhisperLocalEngine())
        elif eng_name == "groq_whisper":
            engines.append(GroqWhisperEngine())
    return engines


def _merge_chunk_results(chunk_results: List[EngineResult]) -> EngineResult:
    """Merge multiple chunks from a single engine on the global timeline.
    
    Drops duplicate words in the overlap regions.
    """
    if not chunk_results:
        return EngineResult(engine_name="unknown", success=False, error="No chunks")
        
    engine_name = chunk_results[0].engine_name
    all_words = []
    
    # Very simple deduplication: ensure word start times are strictly increasing.
    # In overlap regions, we just keep words from the new chunk that start AFTER
    # the last word of the previous chunk.
    last_end = -1.0
    
    for chunk in chunk_results:
        if not chunk.success:
            continue
            
        for w in chunk.words:
            if w.start >= last_end - 0.05:  # small tolerance
                all_words.append(w)
                last_end = w.end
                
    if not all_words:
        # Return a successful result with empty words, representing silence/no speech.
        return EngineResult(
            engine_name=engine_name,
            success=True,
            text="",
            segments=[],
            words=[]
        )
    text = " ".join([w.text for w in all_words])
    from stt.schema import Segment
    segment = Segment(text=text, start=all_words[0].start, end=all_words[-1].end, words=all_words)
    
    return EngineResult(
        engine_name=engine_name,
        success=True,
        text=text,
        segments=[segment],
        words=all_words
    )


def run_stt(audio_path: str, progress_cb: Callable[[str, float, str], None] | None = None) -> STTResult:
    """Run the complete STT pipeline on an audio file.
    
    Args:
        audio_path: Path to the input audio file.
        progress_cb: Optional callback func(stage, fraction, message) for UI updates.
        
    Returns:
        STTResult containing the consensus transcript, uncertain spans, and metadata.
    """
    if not progress_cb:
        progress_cb = lambda s, f, m: None

    t0 = time.perf_counter()
    metadata = STTMetadata()
    
    # 1. Validate
    progress_cb("validating", 0.0, "Validating audio file...")
    audio_info = validate_audio(audio_path)
    metadata.audio_duration_s = audio_info.duration
    
    # Setup temp dir
    temp_dir = tempfile.mkdtemp(prefix="meetscribe_")
    try:
        # 2. Normalize
        progress_cb("validating", 0.3, "Normalizing audio format...")
        norm_path = str(Path(temp_dir) / "normalized.wav")
        normalize_audio(audio_path, norm_path)
        
        # 3. Chunk
        progress_cb("validating", 0.6, "Chunking audio...")
        chunks = split_on_silence(norm_path, out_dir=str(Path(temp_dir)))
        metadata.chunk_count = len(chunks)
        
        # 4. Run Engines
        engines = _get_engines()
        if not engines:
            raise STTError("No STT engines configured.")
            
        engine_results = {e.name: [] for e in engines}
        
        progress_cb("transcribing", 0.0, "Transcribing audio chunks...")
        
        # Run engines on chunks in parallel
        # To keep it safe, we'll iterate engines, and within engine parallelize chunks,
        # or parallelize everything.
        total_tasks = len(engines) * len(chunks)
        completed = 0
        # Track how many chunks had at least one engine succeed
        chunks_with_success = set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, total_tasks)) as executor:
            future_to_task = {}
            for eng in engines:
                for chunk_idx, (chunk_path, offset_s) in enumerate(chunks):
                    future = executor.submit(eng.transcribe, str(chunk_path), offset_s)
                    future_to_task[future] = (eng.name, chunk_idx)
                    
            for future in concurrent.futures.as_completed(future_to_task):
                eng_name, chunk_idx = future_to_task[future]
                try:
                    res = future.result()
                    engine_results[eng_name].append((chunk_idx, res))
                    if res.success:
                        chunks_with_success.add(chunk_idx)
                        
                        # Speed guard for Whisper CPU
                        if eng_name == "whisper":
                            # We can find the engine instance
                            actual_eng = next((e for e in engines if e.name == "whisper"), None)
                            if actual_eng and getattr(actual_eng, "resolved_device", None) == "cpu":
                                if metadata.audio_duration_s > 1200 and getattr(actual_eng, "resolved_model_size", "small") not in ("tiny", "base", "small"):
                                    msg = "Running on CPU: transcription may be slow for long audio."
                                    if msg not in metadata.warnings:
                                        metadata.warnings.append(msg)
                except Exception as e:
                    # Blanket catch to prevent one failed engine/chunk from crashing pipeline
                    err_res = EngineResult(engine_name=eng_name, success=False, error=str(e), error_class=e.__class__.__name__)
                    engine_results[eng_name].append((chunk_idx, err_res))
                
                completed += 1
                frac = completed / total_tasks
                
                # Build per-chunk status message
                chunk_status = []
                for e in engines:
                    r = next((cr for idx, cr in engine_results[e.name] if idx == chunk_idx), None)
                    if r:
                        if r.success:
                            chunk_status.append(f"{e.name} OK")
                        else:
                            # Truncate long error messages for display
                            err_short = (r.error or "unknown")[:80]
                            chunk_status.append(f"{e.name} failed: {err_short}")
                    else:
                        chunk_status.append(f"{e.name} pending")
                
                msg = f"chunk {chunk_idx + 1}/{len(chunks)}: " + ", ".join(chunk_status)
                progress_cb("transcribing", frac, msg)
                
        # Merge chunks per engine
        final_engine_results = []
        engine_errors = []
        for eng_name, chunk_res_list in engine_results.items():
            # Sort by chunk index
            chunk_res_list.sort(key=lambda x: x[0])
            res_list = [r[1] for r in chunk_res_list]
            
            # Check if all failed
            if all(not r.success for r in res_list):
                # Collect ALL distinct error reasons for this engine
                reasons = []
                for r in res_list:
                    if r.error and r.error not in reasons:
                        reasons.append(r.error)
                err_msg = "; ".join(reasons) if reasons else "unknown error"
                err_class = next((r.error_class for r in res_list if r.error_class), None)
                if err_class:
                    err_str = f"{err_msg} ({err_class})"
                    engine_errors.append(f"{eng_name}: {err_str}")
                else:
                    err_str = err_msg
                    engine_errors.append(f"{eng_name}: {err_str}")
                
                metadata.engine_statuses[eng_name] = f"blocked / failed ({err_str})"
            else:
                merged = _merge_chunk_results(res_list)
                if merged.success:
                    final_engine_results.append(merged)
                    metadata.engines_used.append(eng_name)
                    metadata.engine_statuses[eng_name] = "OK"
                    for r in res_list:
                        for w in getattr(r, 'warnings', []):
                            if w not in metadata.warnings:
                                metadata.warnings.append(w)
                    
        if not final_engine_results:
            # List EVERY engine with its own reason
            err_details = " | ".join(engine_errors) if engine_errors else "unknown error"
            raise STTError(f"All STT engines failed. {err_details}")
            
        if len(final_engine_results) < len(engines):
            metadata.fallback_used = True
            # Record EACH failed engine with its specific reason as a warning
            for err in engine_errors:
                metadata.warnings.append(f"Single-engine mode: {err}. Uncertainty map will be empty.")
            
        # 5. Consensus
        progress_cb("consensus", 0.0, "Building consensus...")
        consensus_res = build_consensus(final_engine_results)
        
        if not consensus_res.words:
            raise NoSpeechError()
            
        # 6. Uncertainty Map
        progress_cb("consensus", 0.5, "Mapping uncertainty...")
        spans = generate_spans(consensus_res.disputed_slots, final_engine_results)
        
        # Build STTResult
        from stt.schema import Segment
        from stt.segmenter import build_segments
        
        raw_text = " ".join([w.text for w in consensus_res.words])
        segments = build_segments(consensus_res.words)
        
        if metadata.audio_duration_s == 0.0 and consensus_res.words:
            metadata.audio_duration_s = consensus_res.words[-1].end
            
        metadata.processing_time_s = time.perf_counter() - t0
        
        result = STTResult(
            raw_text=raw_text,
            segments=segments,
            uncertain_spans=spans,
            flags=consensus_res.flags,
            metadata=metadata
        )
        
        # 7. Post-checks
        progress_cb("consensus", 0.9, "Running post-checks...")
        run_postchecks(result)
        
        progress_cb("consensus", 1.0, "Transcription complete.")
        return result
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
