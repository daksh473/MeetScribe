"""Tests for Step 16: CPU Safe Fallbacks."""

import pytest
import os
from unittest.mock import patch, MagicMock

from stt.engines.whisper_local import _resolve_device_and_compute, WhisperLocalEngine

def test_resolve_auto_cpu_without_cuda():
    """Test 'auto' falls back to CPU when CUDA is missing."""
    with patch("torch.cuda.is_available", return_value=False):
        device, compute_type, model_size = _resolve_device_and_compute(
            "auto", "default", "large-v3"
        )
        assert device == "cpu"
        assert compute_type == "int8"
        assert model_size == "medium"

def test_resolve_cuda_with_notice():
    """Test 'cuda' uses float16 and large model."""
    device, compute_type, model_size = _resolve_device_and_compute(
        "cuda", "default", "large-v3"
    )
    assert device == "cuda"
    assert compute_type == "float16"
    assert model_size == "large-v3"

def test_exact_env_variable_reads(monkeypatch):
    """Test exact exact environment variable reads."""
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "base")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    
    eng = WhisperLocalEngine()
    assert eng._cfg_model_size == "base"
    assert eng._cfg_device == "cpu"
    assert eng._cfg_compute_type == "int8"

def test_failing_load_retry_path(monkeypatch):
    """Test fallback on TypeError/load fail."""
    eng = WhisperLocalEngine(model_size="large", device="cpu", compute_type="int8")
    
    def mock_init(*args, **kwargs):
        if args[0] == "medium":  # because it downgrades large to medium
            raise RuntimeError("Fake load error")
        return MagicMock()
        
    monkeypatch.setattr("faster_whisper.WhisperModel", mock_init)
    
    model = eng._get_model()
    assert eng.resolved_model_size == "small"
    assert eng.resolved_device == "cpu"
