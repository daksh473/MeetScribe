import pytest
import httpx
from unittest.mock import patch, MagicMock

from stt.engines.scribe import ScribeEngine

def test_scribe_http_key_sent_correctly(monkeypatch, tmp_path):
    """Test key is stripped and sent correctly."""
    engine = ScribeEngine(api_key="  Bearer fake-key  \n")

    def mock_send(self, request, *args, **kwargs):
        headers = request.headers
        assert headers.get("xi-api-key") == "fake-key"
        return httpx.Response(200, json={"language_code": "en", "text": "OK", "words": []}, request=request)

    monkeypatch.setattr("httpx.Client.send", mock_send)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is True

def test_scribe_401_surfaces_message_no_retry(monkeypatch, tmp_path):
    """401/403 should surface provider message and not retry."""
    engine = ScribeEngine(api_key="fake-key")
    
    calls = 0
        
    def mock_send(self, request, *args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"message": "missing_permissions or detected_unusual_activity"}, request=request)
        
    monkeypatch.setattr("httpx.Client.send", mock_send)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is False
    assert "missing_permissions" in res.error
    assert calls == 1  # No retries!

def test_scribe_429_retries(monkeypatch, tmp_path):
    """429 should trigger a retry."""
    engine = ScribeEngine(api_key="fake-key")
    
    calls = 0
                
    def mock_send(self, request, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(200, json={"language_code": "en", "text": "OK", "words": []}, request=request)
        
    monkeypatch.setattr("httpx.Client.send", mock_send)
    import time
    monkeypatch.setattr(time, "sleep", lambda *args: None)
    
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_text("fake")
    
    res = engine.transcribe(str(dummy_wav))
    assert res.success is True
    assert calls == 2
