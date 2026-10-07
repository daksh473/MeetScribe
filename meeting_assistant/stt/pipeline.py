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


def _get_engines() -> List[STTEngine]:
    """Initialize STT engines based on configuration."""
    engines: List[STTEngine] = []
    for eng_name in settings.engines_list:
        if eng_name == "elevenlabs":
            engines.append(ScribeEngine())
        elif eng_name == "whisper":
            engines.append(WhisperLocalEngine())
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
    progress_cb("validate", 0.0, "Validating audio file...")
    audio_info = validate_audio(audio_path)
    metadata.audio_duration_s = audio_info.duration
    
    # Setup temp dir
    temp_dir = tempfile.mkdtemp(prefix="meetscribe_")
    try:
        # 2. Normalize
        progress_cb("normalize", 0.1, "Normalizing audio format...")
        norm_path = str(Path(temp_dir) / "normalized.wav")
        normalize_audio(audio_path, norm_path)
        
        # 3. Chunk
        progress_cb("chunk", 0.2, "Chunking audio...")
        chunks = split_on_silence(norm_path, out_dir=str(Path(temp_dir)))
        metadata.chunk_count = len(chunks)
        
        # 4. Run Engines
        engines = _get_engines()
        if not engines:
            raise STTError("No STT engines configured.")
            
        engine_results = {e.name: [] for e in engines}
        
        progress_cb("transcribe", 0.3, "Transcribing audio chunks...")
        
        # Run engines on chunks in parallel
        # To keep it safe, we'll iterate engines, and within engine parallelize chunks,
        # or parallelize everything.
        total_tasks = len(engines) * len(chunks)
        completed = 0
        
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
                except Exception as e:
                    # Blanket catch to prevent one failed engine/chunk from crashing pipeline
                    err_res = EngineResult(engine_name=eng_name, success=False, error=str(e))
                    engine_results[eng_name].append((chunk_idx, err_res))
                
                completed += 1
                frac = 0.3 + 0.5 * (completed / total_tasks)
                progress_cb("transcribe", frac, f"Transcribed {completed}/{total_tasks} chunks")
                
        # Merge chunks per engine
        final_engine_results = []
        for eng_name, chunk_res_list in engine_results.items():
            # Sort by chunk index
            chunk_res_list.sort(key=lambda x: x[0])
            res_list = [r[1] for r in chunk_res_list]
            
            # Check if all failed
            if all(not r.success for r in res_list):
                metadata.warnings.append(f"Engine {eng_name} failed on all chunks.")
            else:
                merged = _merge_chunk_results(res_list)
                if merged.success:
                    final_engine_results.append(merged)
                    metadata.engines_used.append(eng_name)
                    
        if not final_engine_results:
            raise STTError("All STT engines failed to transcribe the audio.")
            
        if len(final_engine_results) < len(engines):
            metadata.fallback_used = True
            
        # 5. Consensus
        progress_cb("consensus", 0.85, "Building consensus...")
        consensus_res = build_consensus(final_engine_results)
        
        if not consensus_res.words:
            raise NoSpeechError()
            
        # 6. Uncertainty Map
        progress_cb("uncertainty", 0.9, "Mapping uncertainty...")
        spans = generate_spans(consensus_res.disputed_slots, final_engine_results)
        
        # Build STTResult
        from stt.schema import Segment
        
        raw_text = " ".join([w.text for w in consensus_res.words])
        overall_segment = Segment(
            text=raw_text,
            start=consensus_res.words[0].start,
            end=consensus_res.words[-1].end,
            words=consensus_res.words
        )
        
        metadata.processing_time_s = time.perf_counter() - t0
        
        result = STTResult(
            raw_text=raw_text,
            segments=[overall_segment],
            uncertain_spans=spans,
            flags=consensus_res.flags,
            metadata=metadata
        )
        
        # 7. Post-checks
        progress_cb("postchecks", 0.95, "Running post-checks...")
        run_postchecks(result)
        
        progress_cb("done", 1.0, "Transcription complete.")
        return result
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
