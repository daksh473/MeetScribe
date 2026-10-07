"""Tests for the LLM Client."""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError
from pydantic import BaseModel

from llm.client import LLMClient, LLMClientError
from llm.json_utils import LLMOutputError

class DummySchema(BaseModel):
    name: str
    age: int


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("llm.client.settings.groq_api_key", "fake-groq-key")
    monkeypatch.setattr("llm.client.settings.openrouter_api_key", "fake-or-key")
    monkeypatch.setattr("llm.client.settings.llm1_model", "groq:test-model-1")
    monkeypatch.setattr("llm.client.settings.fallback_models", "openrouter:test-model-2")
    monkeypatch.setattr("llm.client.settings.llm_rate_limit_rpm", 0)  # Disable rate limiter wait


def test_missing_key_clean_error(mock_settings, monkeypatch):
    monkeypatch.setattr("llm.client.settings.groq_api_key", "")
    client = LLMClient()
    
    with pytest.raises(LLMClientError, match="Missing API key for provider 'groq'"):
        client.complete_json("sys", "user", DummySchema)


@patch("llm.client.OpenAI")
def test_invalid_json_retries_once_then_succeeds(mock_openai, mock_settings):
    client = LLMClient()
    
    # Setup mock
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    
    # First response: bad JSON
    bad_msg = MagicMock()
    bad_msg.message.content = "```json\n{ bad json \n```"
    bad_resp = MagicMock()
    bad_resp.choices = [bad_msg]
    
    # Second response: good JSON
    good_msg = MagicMock()
    good_msg.message.content = '{"name": "Alice", "age": 30}'
    good_resp = MagicMock()
    good_resp.choices = [good_msg]
    
    mock_client_instance.chat.completions.create.side_effect = [bad_resp, good_resp]
    
    result = client.complete_json("sys", "user", DummySchema)
    
    assert result.name == "Alice"
    assert result.age == 30
    assert mock_client_instance.chat.completions.create.call_count == 2
    assert len(client.call_log) == 2
    assert client.call_log[0]["success"] is True  # API call succeeded (though JSON was bad)


@patch("llm.client.OpenAI")
def test_invalid_json_twice_fails(mock_openai, mock_settings):
    client = LLMClient()
    
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    
    bad_msg = MagicMock()
    bad_msg.message.content = "This is just text, no JSON."
    bad_resp = MagicMock()
    bad_resp.choices = [bad_msg]
    
    mock_client_instance.chat.completions.create.side_effect = [bad_resp, bad_resp]
    
    with pytest.raises(LLMOutputError, match="Failed to generate valid JSON after 2 attempts"):
        client.complete_json("sys", "user", DummySchema)
        
    assert mock_client_instance.chat.completions.create.call_count == 2


@patch("llm.client.OpenAI")
def test_rate_limit_fallback_to_next_model(mock_openai, mock_settings):
    client = LLMClient()
    
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    
    # Model 1 (Groq) fails with 429 RateLimitError (tenacity will retry it up to 3 times)
    response = MagicMock()
    response.status_code = 429
    err = RateLimitError("Too Many Requests", response=response, body=None)

    # First 3 calls fail with 429 (Groq)
    # 4th call is the fallback model (OpenRouter), which succeeds
    good_msg = MagicMock()
    good_msg.message.content = '{"name": "Bob", "age": 40}'
    good_resp = MagicMock()
    good_resp.choices = [good_msg]
    
    mock_client_instance.chat.completions.create.side_effect = [
        err,
        err,
        err,
        good_resp,
    ]
    
    # For testing, we want to speed up Tenacity's wait_exponential
    with patch("llm.client.wait_exponential", return_value=lambda x: 0.0):
        # We need to re-apply the retry decorator with the mocked wait? 
        # Actually, it's easier to patch time.sleep to not wait.
        with patch("time.sleep"):
            result = client.complete_json("sys", "user", DummySchema)
            
    assert result.name == "Bob"
    
    # Verify call_log shows the fallback attempts
    assert len(client.call_log) == 4
    
    # First 3 should be groq:test-model-1 (failures)
    for i in range(3):
        assert client.call_log[i]["provider"] == "groq"
        assert client.call_log[i]["model"] == "test-model-1"
        assert client.call_log[i]["success"] is False
        assert "Too Many Requests" in client.call_log[i]["error"]
        
    # 4th should be openrouter:test-model-2 (success)
    assert client.call_log[3]["provider"] == "openrouter"
    assert client.call_log[3]["model"] == "test-model-2"
    assert client.call_log[3]["success"] is True
