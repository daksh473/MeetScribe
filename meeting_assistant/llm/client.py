"""LLM Client for OpenAI-compatible endpoints with fallback and structured outputs."""

import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from .json_utils import LLMOutputError, validate_json

T = TypeVar("T", bound=BaseModel)

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

class LLMClientError(Exception):
    pass


class RateLimiter:
    """Simple requests-per-minute rate limiter."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0.0

    def wait(self):
        if self.rpm <= 0:
            return
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


class LLMClient:
    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []
        self.rate_limiters: Dict[str, RateLimiter] = {
            "groq": RateLimiter(settings.llm_rate_limit_rpm),
            "openrouter": RateLimiter(settings.llm_rate_limit_rpm),
        }
        
    def _get_api_key(self, provider: str) -> str:
        if provider == "groq":
            key = settings.groq_api_key
        elif provider == "openrouter":
            key = settings.openrouter_api_key
        else:
            raise LLMClientError(f"Unsupported provider: {provider}")
            
        if not key:
            raise LLMClientError(f"Missing API key for provider '{provider}'. Please set it in .env.")
        return key

    def _call_model(
        self,
        provider: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        response_format: Optional[dict] = None
    ) -> str:
        """Call the specific model using the OpenAI SDK."""
        self.rate_limiters[provider].wait()
        
        base_url = PROVIDER_URLS.get(provider)
        if not base_url:
            raise LLMClientError(f"Unknown base URL for provider: {provider}")
            
        api_key = self._get_api_key(provider)
        
        # Initialize client per call to allow dynamic base_url and api_key
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
            
        t0 = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            
            content = response.choices[0].message.content or ""
            
            self.call_log.append({
                "provider": provider,
                "model": model_id,
                "latency_s": latency,
                "success": True,
            })
            return content
            
        except Exception as e:
            latency = time.time() - t0
            self.call_log.append({
                "provider": provider,
                "model": model_id,
                "latency_s": latency,
                "success": False,
                "error": str(e),
            })
            raise

    def complete_json(self, system: str, user: str, schema: Type[T], temperature: float = 0.0, target_model: Optional[str] = None) -> T:
        """Get validated JSON from the LLM, falling back to other models on API errors."""
        
        # Determine model list (primary + fallbacks)
        primary_model_str = target_model or settings.llm1_model
        if not primary_model_str:
            raise LLMClientError("Primary model not configured. Please set LLM1_MODEL or LLM2_MODEL in .env.")
            
        models_to_try = [primary_model_str] + settings.fallback_models_list
        
        last_api_err = None
        
        for model_str in models_to_try:
            if ":" not in model_str:
                raise LLMClientError(f"Invalid model format '{model_str}'. Expected 'provider:model_id'.")
            provider, model_id = model_str.split(":", 1)
            
            # Setup response_format. 
            # Note: Groq supports {"type": "json_object"}. OpenRouter supports it for many models.
            # We'll use json_object mode, but still rely on our prompt instructions and parser.
            response_format = {"type": "json_object"}
            
            # Instruct the model in the system prompt to output JSON matching the schema
            schema_json = schema.model_json_schema()
            augmented_system = (
                f"{system}\n\n"
                f"You MUST respond with a valid JSON object matching this JSON Schema:\n"
                f"{schema_json}\n"
                "Do not include any text outside the JSON object."
            )
            
            @retry(
                retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(3),
                reraise=True
            )
            def attempt_call(current_user: str) -> str:
                return self._call_model(provider, model_id, augmented_system, current_user, temperature, response_format)
                
            # We'll try the model, and if JSON validation fails, retry ONCE with the error.
            current_user_prompt = user
            
            for parse_attempt in range(2):
                try:
                    # This will retry network/429/5xx errors internally up to 3 times
                    raw_content = attempt_call(current_user_prompt)
                    
                    # Validate JSON
                    return validate_json(raw_content, schema)
                    
                except LLMOutputError as e:
                    if parse_attempt == 0:
                        # Validation failed, append error to prompt and try once more on SAME model
                        current_user_prompt = (
                            f"{user}\n\n"
                            f"Your previous response failed validation:\n{e}\n"
                            "Please correct the JSON and ensure it strictly follows the schema."
                        )
                    else:
                        # Failed twice on parsing, we raise this to bubble up (don't fallback to next model for parse errors usually, 
                        # but we could. For now, bubble up as required).
                        raise LLMOutputError(f"Failed to generate valid JSON after 2 attempts. Last error: {e}")
                        
                except (RateLimitError, APIConnectionError, APIStatusError) as e:
                    # The network/API retry exhausted its 3 attempts. Move to NEXT model.
                    last_api_err = e
                    break
                except LLMClientError as e:
                    # Missing API key or configuration error. Do not fallback, bubble up immediately.
                    raise
                    
            # If we break out of the loop due to API errors, the outer loop continues to the next model.
            
        # If we exhausted all models
        raise LLMClientError(f"All models failed due to API errors. Last error: {last_api_err}")
