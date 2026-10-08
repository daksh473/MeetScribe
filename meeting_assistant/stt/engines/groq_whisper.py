"""Groq Whisper STT engine.

Uses the Groq API (https://api.groq.com/openai/v1/audio/transcriptions).

SDK references
--------------
* Docs: https://console.groq.com/docs/speech-text
* Endpoint: POST https://api.groq.com/openai/v1/audio/transcriptions
* Parameters: file, model, language, response_format="verbose_json", timestamp_granularities=["word"], temperature=0.
* Size limit: 25 MB per request.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from llm.client import RateLimiter
from ..schema import EngineResult, Segment, Word
from .base import STTEngine

logger = logging.getLogger(__name__)

# Shared rate limiter (respects LLM_RATE_LIMIT_RPM)
_groq_stt_rate_limiter = RateLimiter(settings.llm_rate_limit_rpm)


class GroqWhisperEngine(STTEngine):
    """Groq-hosted Whisper transcription engine."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self._api_key = api_key or settings.groq_api_key
        self._model = model or getattr(settings, "groq_stt_model", "whisper-large-v3-turbo")

    @property
    def name(self) -> str:
        return "groq_whisper"

    def _do_transcribe(self, wav_path: str) -> EngineResult:
        if not self._api_key:
            return EngineResult(
                engine_name=self.name,
                success=False,
                error="Groq API key is not set.",
            )

        try:
            _groq_stt_rate_limiter.wait()
            response = self._call_api(wav_path)
            return self._parse_response(response)
        except Exception as e:
            err_str = str(e)
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=err_str[:200],
                error_class=e.__class__.__name__,
            )

    def _call_api(self, wav_path: str):
        headers = {
            "Authorization": f"Bearer {self._api_key}"
        }

        # Check if file > 25MB? (Actually we chunk to 10mins = ~19MB max, so we are safe)
        # But we'll just let the API fail and return 400/413 if it exceeds.

        @retry(
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.WriteError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _do_call():
            with httpx.Client(timeout=120.0) as client:
                with open(wav_path, "rb") as fh:
                    files = {"file": ("audio.wav", fh, "audio/wav")}
                    data = {
                        "model": self._model,
                        "language": "en",
                        "response_format": "verbose_json",
                        "temperature": "0",
                        "timestamp_granularities[]": "word",
                    }
                    resp = client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        data=data,
                        files=files
                    )
                    
                    if resp.status_code in (401, 403):
                        # Do not retry permanent errors
                        raise ValueError(f"Groq auth error ({resp.status_code}): {resp.text}")
                        
                    resp.raise_for_status()
                    return resp.json()

        try:
            return _do_call()
        except ValueError as e:
            # Re-raise ValueError so it's caught as permanent
            raise
        except httpx.HTTPStatusError as e:
            # If we exhausted retries on 429/5xx, it will raise this
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}") from e

    def _parse_response(self, data: dict) -> EngineResult:
        words: List[Word] = []
        segments: List[Segment] = []

        # Groq returns words inside data["words"] if requested via timestamp_granularities
        # Wait, the OpenAI spec puts "words" at top level when verbose_json is used.
        groq_words = data.get("words", [])
        for w in groq_words:
            words.append(
                Word(
                    text=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=None  # Groq API doesn't return per-word confidence currently
                )
            )

        # Build segments from words or use data["segments"] if provided
        groq_segments = data.get("segments", [])
        if groq_segments:
            for s in groq_segments:
                seg_words = [w for w in words if w.start >= s.get("start", 0.0) and w.end <= s.get("end", 0.0)]
                segments.append(
                    Segment(
                        text=s.get("text", "").strip(),
                        start=s.get("start", 0.0),
                        end=s.get("end", 0.0),
                        words=seg_words
                    )
                )
        else:
            # Fallback if no segments provided
            if words:
                segments.append(
                    Segment(
                        text=data.get("text", ""),
                        start=words[0].start,
                        end=words[-1].end,
                        words=words
                    )
                )

        return EngineResult(
            engine_name=self.name,
            success=True,
            text=data.get("text", "").strip(),
            segments=segments,
            words=words,
        )
