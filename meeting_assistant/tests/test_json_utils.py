import pytest
from llm.json_utils import extract_json, LLMOutputError

def test_extract_json_booleans():
    """Verify that JSON extraction handles true/false/null without using eval."""
    llm_output = """
    ```json
    {
        "status": true,
        "finished": false,
        "owner": null
    }
    ```
    """
    data = extract_json(llm_output)
    assert data["status"] is True
    assert data["finished"] is False
    assert data["owner"] is None
