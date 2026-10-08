"""Tests for configuration and secure setup."""

import os
import sys
import subprocess
from pathlib import Path

from config import _clean_env, mask, is_placeholder, parse_model_spec

def test_is_placeholder():
    assert is_placeholder("REPLACE_ME")
    assert is_placeholder("your_api_key_here")
    assert is_placeholder("<model-id>")
    assert is_placeholder("free-model-id")
    assert is_placeholder("")
    assert not is_placeholder("gsk_valid_key")
    assert not is_placeholder("llama3-8b-8192")

def test_clean_env():
    # Whitespace and quotes stripping
    assert _clean_env(' "test"  ') == "test"
    assert _clean_env(" 'test' ") == "test"
    assert _clean_env("Bearer token") == "token"
    assert _clean_env("token\n\r") == "token"
    
    # Placeholders
    assert _clean_env("your_key_here") == ""
    assert _clean_env("<model-id>") == ""
    assert _clean_env(None) == ""

def test_mask():
    assert mask("") == "***"
    assert mask("short") == "***"
    
    # Must never return more than first 4 and last 2 characters
    val = mask("sk-or-v1-abcdef1234567890")
    assert val == "sk-o...90"
    assert len(val.split("...")[0]) == 4
    assert len(val.split("...")[1]) == 2
    
    val2 = mask("gsk_tNMH4l8aw6qsX")
    assert val2 == "gsk_...sX"

def test_parse_model_spec():
    assert parse_model_spec("groq:llama-x") == ("groq", "llama-x")
    assert parse_model_spec("openrouter:vendor/model:free") == ("openrouter", "vendor/model:free")
    
    import pytest
    with pytest.raises(ValueError):
        parse_model_spec("invalid_spec")
        
    with pytest.raises(ValueError):
        parse_model_spec("")
        
    assert parse_model_spec("  groq : model-x  ") == ("groq", "model-x")

def test_scan_secrets_detects_fake_key(tmp_path):
    test_file = tmp_path / "leak.py"
    test_file.write_text('key = "gsk_12345678901234567890"', encoding="utf-8")
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "scan_secrets.py"
    res = subprocess.run(["python", str(script_path)], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Possible leaked secret found" in res.stdout

def test_scan_secrets_passes_clean_file(tmp_path):
    test_file = tmp_path / "clean.py"
    test_file.write_text('key = "normal_string"', encoding="utf-8")
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "scan_secrets.py"
    res = subprocess.run(["python", str(script_path)], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Possible leaked secret found" not in res.stdout

# Test check_setup.py API call mocking
def test_check_setup_http_mock(monkeypatch, capsys):
    import scripts.check_setup as cs
    
    # Mock settings
    cs.settings.groq_api_key = "gsk_mock_12345"
    cs.settings.llm1_model = "groq:mock-model"
    cs.settings.llm2_model = "groq:mock-model"
    cs.settings.groq_stt_model = "mock-model"
    cs.settings.fallback_models = ""
    cs.settings.elevenlabs_api_key = ""
    cs.settings.openrouter_api_key = ""
    
    # Mock raw environment reading
    def mock_dotenv_values(path):
        return {"GROQ_API_KEY": "gsk_mock_12345"}
    monkeypatch.setattr("dotenv.dotenv_values", mock_dotenv_values)
    
    # Mock original environment
    cs.original_environ = {}
    
    # Test 200
    def mock_api_call_200(url, headers, data=None):
        if "chat/completions" in url:
            return 200, {"choices": [{}]}
        return 200, {"data": [{"id": "mock-model"}]}
    monkeypatch.setattr(cs, "_api_call", mock_api_call_200)
    assert cs.check_groq() == True
    
    # Test 401
    def mock_api_call_401(url, headers, data=None):
        return 401, {"error_body": "Invalid API key"}
    monkeypatch.setattr(cs, "_api_call", mock_api_call_401)
    assert cs.check_groq() == False
    out, err = capsys.readouterr()
    assert "401" in out
    assert "create a new key" in out
    
    # Test 403
    def mock_api_call_403(url, headers, data=None):
        return 403, {"error_body": "Permission Denied"}
    monkeypatch.setattr(cs, "_api_call", mock_api_call_403)
    assert cs.check_groq() == False
    out, err = capsys.readouterr()
    assert "403" in out
    assert "lacks permission" in out

    # Test Network Error (timeout / DNS)
    def mock_api_call_network(url, headers, data=None):
        return 0, {"network_error": "TimeoutError: Request timed out"}
    monkeypatch.setattr(cs, "_api_call", mock_api_call_network)
    assert cs.check_groq() == False
    out, err = capsys.readouterr()
    assert "0 -" in out
    assert "request did not reach the provider" in out
    assert "TimeoutError" in out
