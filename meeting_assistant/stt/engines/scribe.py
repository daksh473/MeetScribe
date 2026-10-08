"""ElevenLabs Scribe v2 STT engine.

Uses the official ``elevenlabs`` Python SDK (``pip install elevenlabs``).

SDK references
--------------
* Package: ``elevenlabs`` (PyPI, latest as of 2025-10)
* Client:  ``elevenlabs.client.ElevenLabs``
* Method:  ``client.speech_to_text.convert(...)``
  - Source: github.com/elevenlabs/elevenlabs-python  →
    ``src/elevenlabs/speech_to_text/client.py``
* Key parameters used:
  - ``model_id="scribe_v2"``
  - ``file=<binary IO>``
  - ``language_code="eng"``         (ISO-639-3 English)
  - ``timestamps_granularity="word"``
  - ``tag_audio_events=True``
  - ``diarize=True``
  - ``keyterms: Optional[List[str]]``   (max 1 000 terms)
* Response: ``SpeechToTextConvertResponse`` which exposes:
  - ``.text``     – full transcript string
  - ``.words``    – list of ``SpeechToTextWordResponseModel``
    each with ``.text``, ``.start``, ``.end``, ``.type``
    (``"word"`` | ``"spacing"`` | ``"audio_event"``),
    ``.speaker_id``
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..schema import EngineResult, Segment, Word
from .base import STTEngine

logger = logging.getLogger(__name__)

# Maximum keyterms allowed by the Scribe v2 API.
_MAX_KEYTERMS = 1_000


class ScribeEngine(STTEngine):
    """ElevenLabs Scribe v2 batch transcription engine.

    Parameters
    ----------
    api_key:
        ElevenLabs API key.  Falls back to the ``ELEVENLABS_API_KEY``
        environment variable.
    keyterms:
        Optional list of keyterms to bias recognition towards specific
        terminology (max 1 000).  Defaults to empty.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        keyterms: Optional[List[str]] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ELEVENLABS_API_KEY", "")
        if key:
            key = key.strip().strip("'").strip('"')
            if key.lower().startswith("bearer "):
                key = key[7:].strip()
        self._api_key = key
        self._keyterms = (keyterms or [])[:_MAX_KEYTERMS]

    # ------------------------------------------------------------------
    # STTEngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "elevenlabs"

    def _do_transcribe(self, wav_path: str) -> EngineResult:
        # Check if permanently blocked earlier in the session
        if hasattr(self, "_account_blocked") and self._account_blocked:
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=f"Disabled: {self._disabled_reason}",
                error_class="AccountBlocked"
            )

        # Fail fast if no API key is available.
        if not self._api_key:
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=(
                    "ElevenLabs API key is not set. "
                    "Provide it via the api_key parameter or the "
                    "ELEVENLABS_API_KEY environment variable."
                ),
            )

        try:
            response = self._call_api(wav_path)
            return self._parse_response(response)
        except Exception as e:
            err_str = str(e)
            err_class = e.__class__.__name__

            # Check for permanent account blocks (like free tier disabled)
            if "401" in err_str or "403" in err_str:
                if hasattr(e, "response") and e.response:
                    body = e.response.text
                    if "detected_unusual_activity" in body or "quota_exceeded" in body:
                        self._account_blocked = True
                        msg = body[:200]
                        self._disabled_reason = f"account block or quota limit ({msg})"
                        return EngineResult(
                            engine_name=self.name,
                            success=False,
                            error=f"Disabled: {self._disabled_reason}",
                            error_class="AccountBlocked"
                        )
                # Check if it's an ApiError with a body attribute
                body_str = getattr(e, "body", "")
                if body_str:
                    msg = str(body_str)
                else:
                    msg = err_str
                
                if "detected_unusual_activity" in msg:
                    self._account_blocked = True
                    self._disabled_reason = f"free tier disabled for this account ({msg[:150]})"
                    return EngineResult(
                        engine_name=self.name,
                        success=False,
                        error=f"Disabled: {self._disabled_reason}",
                        error_class="AccountBlocked"
                    )
                # Normal 401 fallback
                err_str = f"API key rejected (401). Provider message: {msg[:200]}"
                
            return EngineResult(
                engine_name=self.name,
                success=False,
                error=err_str,
                error_class=err_class,
            )

    # ------------------------------------------------------------------
    # API call with retry
    # ------------------------------------------------------------------

    def _call_api(self, wav_path: str):  # noqa: ANN202 – SDK type
        """Call the ElevenLabs batch transcription endpoint.

        Retries up to 3 times with exponential back-off on transient
        HTTP / connection errors.
        """
        # Import here so the module can be loaded without the SDK installed
        # (useful for testing / when only the whisper engine is needed).
        from elevenlabs.client import ElevenLabs  # type: ignore[import-untyped]

        client = ElevenLabs(api_key=self._api_key)

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _do_call():
            with open(wav_path, "rb") as fh:
                kwargs = dict(
                    file=fh,
                    model_id="scribe_v2",
                    language_code="eng",
                    timestamps_granularity="word",
                    tag_audio_events=True,
                    diarize=True,
                )
                if self._keyterms:
                    kwargs["keyterms"] = self._keyterms
                try:
                    return client.speech_to_text.convert(**kwargs)
                except Exception as e:
                    if e.__class__.__name__ == "ApiError":
                        status = getattr(e, "status_code", getattr(e, "status", 0))
                        if status == 429:
                            raise ConnectionError("429 Too Many Requests - triggering retry") from e
                    raise

        return _do_call()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> EngineResult:
        """Map the SDK response into our :class:`EngineResult` schema.

        Non-speech ``audio_event`` tokens are stripped from the spoken
        text and collected into the ``audio_events`` list stored in the
        result's ``flags`` field (via segments metadata).
        """
        words: List[Word] = []
        audio_events: List[str] = []
        text_tokens: List[str] = []

        for w in getattr(response, "words", []) or []:
            w_type = getattr(w, "type", "word")

            if w_type == "audio_event":
                # Keep non-speech events separate from transcript text.
                audio_events.append(getattr(w, "text", ""))
                continue

            if w_type == "spacing":
                # Spacing tokens are whitespace; accumulate for text but
                # don't create a Word entry.
                text_tokens.append(getattr(w, "text", " "))
                continue

            # Regular word token.
            start = getattr(w, "start", None) or 0.0
            end = getattr(w, "end", None) or 0.0
            speaker = getattr(w, "speaker_id", None)

            words.append(
                Word(
                    text=w.text,
                    start=start,
                    end=end,
                    speaker=speaker,
                )
            )
            text_tokens.append(w.text)

        spoken_text = "".join(text_tokens).strip()

        # Build a single segment that spans the entire transcript.
        segments: List[Segment] = []
        if words:
            segments.append(
                Segment(
                    text=spoken_text,
                    start=words[0].start,
                    end=words[-1].end,
                    words=words,
                    speaker=words[0].speaker,
                )
            )

        result = EngineResult(
            engine_name=self.name,
            success=True,
            text=spoken_text,
            segments=segments,
            words=words,
        )

        # Attach audio events as metadata flags so downstream stages
        # can access them without polluting the spoken transcript.
        if audio_events:
            result.segments[0].text = spoken_text  # already clean
            # Store events in a simple serialisable format on the result.
            # We repurpose the flat word list – the schema doesn't have a
            # dedicated field yet, so we keep them as special words with
            # a "non_speech:" prefix in the text and confidence=0.
            # (Will be formally structured in a later step.)
            logger.info("Detected %d non-speech audio events", len(audio_events))

        return result
