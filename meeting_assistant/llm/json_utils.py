"""JSON extraction and validation utilities for LLM outputs."""

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

class LLMOutputError(Exception):
    """Raised when LLM output cannot be parsed or validated."""
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a string, stripping markdown."""
    # Remove markdown code fences if present
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    
    # Try to find the first '{' and last '}'
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise LLMOutputError("No JSON object found in output.")
        
    json_str = text[start_idx : end_idx + 1]
    
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise LLMOutputError("Parsed JSON is not an object.")
        return data
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"Failed to parse JSON: {e}") from e


def validate_json(text: str, schema: Type[T]) -> T:
    """Extract JSON from text and validate against a Pydantic schema."""
    data = extract_json(text)
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise LLMOutputError(f"JSON validation failed: {e}") from e
