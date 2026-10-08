#!/usr/bin/env python
"""Pre-commit hook to scan for accidentally committed API keys."""

import sys
import re
from pathlib import Path
import subprocess

PATTERNS = [
    r"sk-[a-zA-Z0-9]{30,}",
    r"sk-or-v1-[a-f0-9]{50,}",
    r"gsk_[a-zA-Z0-9]{20,}",
    r"sk_[a-f0-9]{30,}"
]

def scan_files():
    try:
        # Get list of tracked and staged files
        res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
        files = res.stdout.splitlines()
        if not files:
            raise subprocess.CalledProcessError(1, "git")
    except subprocess.CalledProcessError:
        # Fallback to current dir if not in git or no files tracked
        files = [str(p) for p in Path(".").rglob("*") if p.is_file() and ".git" not in p.parts]
        
    found_secrets = False
    
    for f in files:
        if not Path(f).is_file():
            continue
            
        if Path(f).name == "scan_secrets.py" or Path(f).name == ".env.example":
            continue
            
        try:
            content = Path(f).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
            
        for pattern in PATTERNS:
            if re.search(pattern, content):
                print(f"[ERROR] Possible leaked secret found in: {f}")
                found_secrets = True
                
    if found_secrets:
        print("Commit rejected: Remove secrets before committing. If these are false positives, ignore.")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    scan_files()
