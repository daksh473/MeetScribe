#!/usr/bin/env python3
import os
import re
import struct
import wave
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

raw_key = os.getenv("ELEVENLABS_API_KEY", "")

def clean_key(val):
    if not val:
        return ""
    val = val.strip()
    val = val.strip("'\"")
    if val.startswith("Bearer "):
        val = val[7:].strip()
    return val

key = clean_key(raw_key)

def print_masked(k):
    if not k:
        return "empty"
    if len(k) < 5:
        return f"len={len(k)}, too short"
    return f"len={len(k)}, {k[:3]}...{k[-2:]}"

print("=== ELEVENLABS DEBUG ===")
print("Key info:", print_masked(key))

def redact(text):
    return re.sub(r'(sk_[a-zA-Z0-9]{20,})', '***', text)

# 1. Call User Endpoint
headers = {"xi-api-key": key}
with httpx.Client() as client:
    resp = client.get("https://api.elevenlabs.io/v1/user", headers=headers)
    status_user = resp.status_code
    print(f"User endpoint status: {status_user}")
    print(f"User body (first 300): {redact(resp.text[:300])}")

# 2. Call STT Endpoint
wav_path = "debug_temp.wav"
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(struct.pack("<32000h", *([0] * 32000)))

with httpx.Client() as client:
    with open(wav_path, "rb") as fh:
        files = {"file": fh}
        data = {
            "model_id": "scribe_v2",
            "language_code": "eng"
        }
        resp = client.post("https://api.elevenlabs.io/v1/speech-to-text", headers=headers, data=data, files=files)
        status_stt = resp.status_code
        body_stt = resp.text
        print(f"STT endpoint status: {status_stt}")
        print(f"STT body (first 400): {redact(body_stt[:400])}")

if os.path.exists(wav_path):
    os.remove(wav_path)

# 3. Check OS Env
original_env_key = os.environ.get("ELEVENLABS_API_KEY")
env_differs = False
if original_env_key is not None:
    print("OS ELEVENLABS_API_KEY exists.")
    # Check if OS env value differs from the .env value we loaded
    # Actually load_dotenv(override=True) modifies os.environ!
    # So os.environ["ELEVENLABS_API_KEY"] will be the .env value unless the file is missing
    pass
else:
    print("OS ELEVENLABS_API_KEY does not exist (or overridden).")

# We need a proper way to check if a system env var differs from .env
# Let's parse .env manually
env_file_key = ""
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("ELEVENLABS_API_KEY="):
                env_file_key = line.split("=", 1)[1].strip()
                break

clean_env_file_key = clean_key(env_file_key)
if clean_env_file_key and key != clean_env_file_key:
    env_differs = True
    print(f"STALE ENV WARNING: Loaded key differs from .env file! loaded: {print_masked(key)} vs file: {print_masked(clean_env_file_key)}")

# Verdict
print("========================")
if env_differs:
    print("Verdict: STALE_ENV")
elif status_stt in (401, 403):
    lower_body = body_stt.lower()
    if "detected_unusual_activity" in lower_body:
        print(f"Verdict: ACCOUNT_BLOCKED - {body_stt[:150]}")
    elif "missing_permissions" in lower_body or "permission" in lower_body:
        print("Verdict: MISSING_PERMISSION")
    elif "invalid_api_key" in lower_body:
        print("Verdict: KEY_INVALID")
    else:
        print("Verdict: KEY_INVALID (or other auth error)")
elif status_stt == 429 or "quota_exceeded" in body_stt.lower():
    print("Verdict: QUOTA_OR_PLAN")
elif status_stt == 200:
    print("Verdict: ALL_OK")
else:
    print("Verdict: CODE_BUG")
