#!/usr/bin/env python
"""Pre-demo secure setup and key checker."""

import os
import sys
import shutil
import urllib.request
import urllib.error
import json
import argparse
import difflib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings, mask, original_environ, is_placeholder, parse_model_spec

def print_row(name: str, status: str, detail: str = ""):
    color = "\033[92m" if status == "PASS" else ("\033[93m" if status == "WARN" else "\033[91m")
    reset = "\033[0m"
    print(f"| {name:<25} | {color}{status:<6}{reset} | {detail}")

def check_deps() -> bool:
    success = True
    print_row("Python version", "PASS", sys.version.split(" ")[0])
    
    ffmpeg = shutil.which("ffmpeg") or settings.ffmpeg_path
    if ffmpeg and Path(ffmpeg).is_file() or shutil.which("ffmpeg"):
        print_row("ffmpeg", "PASS", "Found")
    else:
        print_row("ffmpeg", "FAIL", "Not found on PATH")
        success = False
        
    try:
        import faster_whisper
        print_row("faster-whisper", "PASS", f"Importable (device: {settings.whisper_device})")
    except ImportError:
        print_row("faster-whisper", "FAIL", "Not importable")
        success = False
    return success

def check_git() -> bool:
    import subprocess
    try:
        res = subprocess.run(["git", "check-ignore", ".env"], capture_output=True, text=True)
        if ".env" in res.stdout:
            print_row(".env gitignore", "PASS", ".env is ignored")
            return True
        else:
            print_row(".env gitignore", "FAIL", ".env is NOT ignored")
            return False
    except Exception:
        print_row(".env gitignore", "WARN", "git not found or error")
        return True

def _api_call(url: str, headers: dict, data: dict = None) -> tuple[int, dict]:
    if "User-Agent" not in headers:
        headers["User-Agent"] = "meetscribe-check/1.0"
    req = urllib.request.Request(url, headers=headers)
    if data is not None:
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        msg = ""
        try:
            raw_body = e.read().decode("utf-8")
            import re
            redacted = re.sub(r'(gsk_[A-Za-z0-9]+|sk-or-v1-[A-Za-z0-9]+|sk_[A-Za-z0-9]+)', '***REDACTED***', raw_body)
            msg = redacted[:200]
        except:
            msg = str(e)
        return e.code, {"error_body": msg}
    except Exception as e:
        return 0, {"network_error": f"{e.__class__.__name__}: {str(e)}"}

def _check_key_diagnostics(var_name: str, expected_prefix: str, required: bool = True) -> tuple[bool, str]:
    from dotenv import dotenv_values
    raw_env = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    
    raw_val = raw_env.get(var_name, "")
    
    if is_placeholder(raw_val):
        print_row(var_name, "FAIL" if required else "WARN", f"Placeholder not replaced in .env: {var_name}")
        return False, ""
        
    if not raw_val:
        print_row(var_name, "FAIL" if required else "WARN", "Missing")
        return False, ""
        
    # Check OS env differing
    sys_val = original_environ.get(var_name)
    if sys_val and sys_val != raw_val:
        print_row(var_name + " Env", "WARN", f"System env differs from .env; .env is used.")
        
    clean_val = raw_val.strip().strip("'\"").replace("\n", "").replace("\r", "")
    if clean_val.startswith("Bearer "):
        clean_val = clean_val[7:].strip()
    clean_val = "".join(c for c in clean_val if ord(c) < 128)
    
    flags = []
    if raw_val != raw_val.strip(): flags.append("whitespace")
    if raw_val.startswith('"') or raw_val.startswith("'"): flags.append("quotes")
    if raw_val.startswith("Bearer "): flags.append("Bearer")
    if "\n" in raw_val or "\r" in raw_val: flags.append("newlines")
    if not all(ord(c) < 128 for c in raw_val): flags.append("non-ASCII")
    
    flag_str = f" [stripped: {', '.join(flags)}]" if flags else ""
    length = len(clean_val)
    
    print_row(var_name, "PASS", f"Len: {length}, Mask: {mask(clean_val)}{flag_str}")
    if expected_prefix and not clean_val.startswith(expected_prefix):
        print_row(var_name + " Prefix", "WARN", f"Key does not start with {expected_prefix}")
        
    return True, clean_val

def check_groq() -> bool:
    success = True
    ok, key = _check_key_diagnostics("GROQ_API_KEY", "gsk_")
    if not ok:
        return False
        
    status, resp = _api_call("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
    if status == 200:
        print_row("Groq API Live", "PASS", "OK")
        
        available_models = [m["id"] for m in resp.get("data", [])]
        
        def check_model(m_spec, var_name):
            if is_placeholder(m_spec):
                print_row(var_name, "FAIL", f"Placeholder not replaced in .env: {var_name}")
                return False
            try:
                provider, model_id = parse_model_spec(m_spec)
            except ValueError as e:
                print_row(var_name, "FAIL", str(e))
                return False
            
            if provider != "groq":
                return True
                
            if model_id in available_models:
                print_row(f"Groq {model_id}", "PASS", "Found")
                return True
            else:
                matches = difflib.get_close_matches(model_id, available_models, n=5)
                sug = f" (Suggestions: {', '.join(matches)})" if matches else ""
                print_row(f"Groq {model_id}", "FAIL", f"Not found{sug}")
                return False

        if not check_model(settings.llm1_model, "LLM1_MODEL"): success = False
        if not check_model(settings.llm2_model, "LLM2_MODEL"): success = False
        if not check_model("groq:" + settings.groq_stt_model, "GROQ_STT_MODEL"): success = False
        for i, fallback in enumerate(settings.fallback_models_list):
            if fallback.startswith("groq:"):
                if not check_model(fallback, f"FALLBACK_MODELS[{i}]"): success = False
                
        # Tiny JSON mode test
        model_id = ""
        try:
            _, model_id = parse_model_spec(settings.llm2_model)
        except ValueError:
            pass
            
        if model_id in available_models:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return exactly {'ok': true}"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 10
            }
            st2, resp2 = _api_call("https://api.groq.com/openai/v1/chat/completions", 
                                   {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                   payload)
            if st2 == 200 and "choices" in resp2:
                print_row("Groq JSON-mode", "PASS", "OK")
            else:
                print_row("Groq JSON-mode", "WARN", f"Failed (code {st2})")
                
    elif status == 401:
        msg = resp.get("error_body", "")
        print_row("Groq API Live", "FAIL", f"401 - {msg}. Hint: key is rejected by the provider: create a new key in the provider dashboard and paste it into .env")
        success = False
    elif status == 403:
        msg = resp.get("error_body", "")
        print_row("Groq API Live", "FAIL", f"403 - {msg}. Hint: request was blocked or the key lacks permission")
        success = False
    elif status == 429:
        print_row("Groq API Live", "WARN", "Rate limited (429)")
    elif status == 0:
        msg = resp.get("network_error", "")
        print_row("Groq API Live", "FAIL", f"0 - {msg}. Hint: request did not reach the provider: check internet, VPN or proxy, firewall, or try another network")
        success = False
    else:
        print_row("Groq API Live", "FAIL", f"Network error or code {status}")
        success = False
        
    return success

def check_openrouter() -> bool:
    success = True
    ok, key = _check_key_diagnostics("OPENROUTER_API_KEY", "sk-or-", required=False)
    if not ok:
        return True
        
    status, resp = _api_call("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"})
    if status == 200:
        print_row("OpenRouter API Live", "PASS", "OK")
        
        available_models = {m["id"]: m for m in resp.get("data", [])}
        
        for i, fallback in enumerate(settings.fallback_models_list):
            if is_placeholder(fallback):
                print_row(f"FALLBACK_MODELS[{i}]", "FAIL", f"Placeholder not replaced in .env: FALLBACK_MODELS")
                success = False
                continue
                
            try:
                provider, model_id = parse_model_spec(fallback)
            except ValueError as e:
                print_row(f"FALLBACK_MODELS[{i}]", "FAIL", str(e))
                success = False
                continue
                
            if provider != "openrouter":
                continue
                
            if model_id in available_models:
                pricing = available_models[model_id].get("pricing", {})
                is_free = pricing.get("prompt") == "0" or pricing.get("prompt") == 0 or pricing.get("prompt") == "0.0"
                
                if ":free" in model_id or is_free:
                    if is_free:
                        print_row(f"OR {model_id}", "PASS", "Found and is Free")
                    else:
                        print_row(f"OR {model_id}", "WARN", "Found but NOT free")
                else:
                    print_row(f"OR {model_id}", "PASS", "Found")
            else:
                matches = difflib.get_close_matches(model_id, available_models.keys(), n=5)
                sug = f" (Suggestions: {', '.join(matches)})" if matches else ""
                print_row(f"OR {model_id}", "FAIL", f"Not found{sug}")
                success = False
                
    elif status == 401:
        msg = resp.get("error_body", "")
        print_row("OpenRouter API Live", "FAIL", f"401 - {msg}. Hint: key is rejected by the provider: create a new key in the provider dashboard and paste it into .env")
        success = False
    elif status == 403:
        msg = resp.get("error_body", "")
        print_row("OpenRouter API Live", "FAIL", f"403 - {msg}. Hint: request was blocked or the key lacks permission")
        success = False
    elif status == 429:
        print_row("OpenRouter API Live", "WARN", "Rate limited (429)")
    elif status == 0:
        msg = resp.get("network_error", "")
        print_row("OpenRouter API Live", "FAIL", f"0 - {msg}. Hint: request did not reach the provider: check internet, VPN or proxy, firewall, or try another network")
        success = False
    else:
        print_row("OpenRouter API Live", "FAIL", f"Network error or code {status}")
        success = False
    return success

def check_elevenlabs() -> bool:
    success = True
    ok, key = _check_key_diagnostics("ELEVENLABS_API_KEY", "sk_", required=False)
    if not ok:
        return True
        
    status, resp = _api_call("https://api.elevenlabs.io/v1/user", {"xi-api-key": key})
    if status == 200:
        print_row("ElevenLabs API Live", "PASS", "OK")
    elif status == 401:
        msg = resp.get("error_body", "").lower()
        if "detected_unusual_activity" in msg:
            print_row("ElevenLabs API Live", "WARN", "Account Blocked (detected_unusual_activity)")
        else:
            print_row("ElevenLabs API Live", "FAIL", "Invalid API Key")
        success = False
    elif status == 403:
        print_row("ElevenLabs API Live", "FAIL", "Permission denied")
        success = False
    else:
        print_row("ElevenLabs API Live", "FAIL", f"HTTP {status}")
        success = False
    return success

def list_models(provider: str):
    if provider == "groq":
        ok, key = _check_key_diagnostics("GROQ_API_KEY", "gsk_", required=True)
        if not ok: return
        status, resp = _api_call("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
        if status == 200:
            print(f"--- Groq Available Models ---")
            for m in resp.get("data", []):
                print(m["id"])
    elif provider == "openrouter":
        ok, key = _check_key_diagnostics("OPENROUTER_API_KEY", "sk-or-", required=True)
        if not ok: return
        status, resp = _api_call("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"})
        if status == 200:
            print(f"--- OpenRouter Free Models ---")
            for m in resp.get("data", []):
                pricing = m.get("pricing", {})
                is_free = pricing.get("prompt") == "0" or pricing.get("prompt") == 0 or pricing.get("prompt") == "0.0"
                if is_free:
                    print(m["id"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", choices=["groq", "openrouter"], help="List models for a provider")
    args = parser.parse_args()
    
    if args.list_models:
        list_models(args.list_models)
        sys.exit(0)
        
    print("=== Secure Setup Check ===")
    s1 = check_deps()
    s2 = check_git()
    s3 = check_groq()
    s4 = check_openrouter()
    s5 = check_elevenlabs()
    
    if not all([s1, s2, s3, s4, s5]):
        print("\n[ERROR] Setup check failed. Please resolve the failing items above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
