#!/usr/bin/env python
"""LLM Live Connectivity Test."""

import sys
from pathlib import Path
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from llm.client import LLMClient, LLMClientError

class OKResponse(BaseModel):
    reply: str

def test_model(client: LLMClient, model_env_key: str, model_spec: str) -> bool:
    print(f"\n--- Testing {model_env_key} ({model_spec}) ---")
    if not model_spec:
        print(f"FAIL - {model_env_key} is not configured.")
        return False
        
    try:
        # We pass target_model dynamically to test specific models using the abstraction
        resp = client.complete_json(
            system="You are a strict testing bot.",
            user="Reply with exactly: OK",
            schema=OKResponse,
            temperature=0.0,
            target_model=model_spec
        )
        if "OK" in resp.reply.upper():
            print("PASS - Successfully received structured JSON response containing OK.")
            return True
        else:
            print(f"FAIL - INVALID_RESPONSE - Expected OK but got: {resp.reply}")
            return False
            
    except Exception as e:
        err_str = str(e)
        if "RateLimitError" in err_str or "429" in err_str:
            print(f"FAIL - RATE_LIMITED - {err_str[:200]}")
        elif "AuthenticationError" in err_str or "401" in err_str or "403" in err_str:
            print(f"FAIL - AUTH_ERROR - {err_str[:200]}")
        elif "NotFoundError" in err_str or "404" in err_str:
            print(f"FAIL - MODEL_NOT_FOUND - {err_str[:200]}")
        elif "Connection" in err_str or "Timeout" in err_str:
            print(f"FAIL - NETWORK_ERROR - {err_str[:200]}")
        else:
            print(f"FAIL - API ERROR - {err_str[:200]}")
        return False

def main():
    print("=== LLM Connectivity Test ===")
    
    try:
        client = LLMClient()
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)
        
    m1 = test_model(client, "LLM1_MODEL", settings.llm1_model)
    m2 = test_model(client, "LLM2_MODEL", settings.llm2_model)
    
    if not (m1 and m2):
        print("\n[ERROR] Connectivity test failed. One or more required models failed.")
        sys.exit(1)
    
    print("\n[SUCCESS] All LLM models are successfully connected and responding with structured JSON.")
    sys.exit(0)

if __name__ == "__main__":
    main()
