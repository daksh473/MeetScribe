#!/usr/bin/env python
import os
import sys

def main():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed.")
        sys.exit(1)
        
    model_size = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
    print(f"Prefetching faster-whisper model: {model_size}...")
    
    download_root = os.environ.get("WHISPER_MODEL_PATH")
    try:
        # Initializing without device will still download it
        WhisperModel(model_size, device="cpu", compute_type="int8", download_root=download_root)
        
        # Check if HF_HOME or similar is used for caching
        if download_root:
            cache_dir = download_root
        else:
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            
        print(f"Model successfully downloaded/cached to: {cache_dir}")
    except Exception as e:
        print(f"Failed to prefetch model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
