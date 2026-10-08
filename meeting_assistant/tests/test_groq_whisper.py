"""Tests for Groq Whisper engine."""

import httpx
import pytest
from stt.engines.groq_whisper import GroqWhisperEngine

def test_groq_whisper_success(monkeypatch, tmp_path):
    """Test successful response parsing with word timestamps."""
    engine = GroqWhisperEngine(api_key="fake-key", model="mock-model")
    
    class MockResponse:
        status_code = 200
        text = "mock response"
        def json(self):
            return {
                "text": "Hello world.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world.", "start": 0.5, "end": 1.0}
                ]
            }
        def raise_for_status(self):
            pass

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("httpx.Client.post", mock_post)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is True
    assert res.text == "Hello world."
    assert len(res.words) == 2
    assert res.words[0].text == "Hello"
    assert res.words[0].start == 0.0
    assert len(res.segments) == 1

def test_groq_whisper_401_no_retry(monkeypatch, tmp_path):
    """Test that a 401 response disables retries and returns a failure."""
    engine = GroqWhisperEngine(api_key="fake-key")
    
    calls = 0
    class MockResponse401:
        status_code = 401
        text = "invalid_api_key"
        
    def mock_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return MockResponse401()
        
    monkeypatch.setattr("httpx.Client.post", mock_post)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is False
    assert "401" in res.error
    assert calls == 1  # No retries!

def test_groq_whisper_429_retry(monkeypatch, tmp_path):
    """Test that a 429 response triggers retry."""
    engine = GroqWhisperEngine(api_key="fake-key")
    
    calls = 0
    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "error or success"
        def json(self):
            return {"text": "Success!"}
        def raise_for_status(self):
            if self.status_code != 200:
                class Req:
                    pass
                req = Req()
                raise httpx.HTTPStatusError("429", request=req, response=self)

    def mock_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return MockResponse(429)
        return MockResponse(200)

    monkeypatch.setattr("httpx.Client.post", mock_post)
    
    # We also need to speed up tenacity sleep for the test
    import time
    monkeypatch.setattr(time, "sleep", lambda *args: None)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is True
    assert calls == 2
