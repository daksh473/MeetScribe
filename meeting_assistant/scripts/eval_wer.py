#!/usr/bin/env python
"""Evaluate STT pipeline outputs against a reference transcript using WER.

Usage:
    python scripts/eval_wer.py ref.txt hyp.txt
    python scripts/eval_wer.py ref.txt outputs/stt_result.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import jiwer
except ImportError:
    print("Error: 'jiwer' not installed. Please run: pip install jiwer")
    sys.exit(1)


def count_category_errors(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int]:
    """Return an approximate count of errors involving numbers and negations."""
    # A simple approach: use jiwer's alignment
    alignment = jiwer.process_words(ref_words, hyp_words).alignments[0]
    
    num_errors = 0
    neg_errors = 0
    
    neg_words = {"not", "no", "never", "cannot", "without", "neither", "nor", "dont", "wont", "cant", "shouldnt", "wouldnt", "couldnt", "isnt", "arent", "aint", "doesnt", "didnt", "hasnt", "havent", "hadnt"}
    
    for op in alignment:
        if op.type == "equal":
            continue
            
        # Collect all words involved in the error
        err_words = []
        if op.type in ("substitute", "delete"):
            err_words.extend([ref_words[i] for i in range(op.ref_start_idx, op.ref_end_idx)])
        if op.type in ("substitute", "insert"):
            err_words.extend([hyp_words[i] for i in range(op.hyp_start_idx, op.hyp_end_idx)])
            
        is_num = False
        is_neg = False
        
        for w in err_words:
            clean_w = re.sub(r'[^\w\s]', '', w.lower())
            if re.search(r'\d', w):
                is_num = True
            if clean_w in neg_words:
                is_neg = True
                
        if is_num:
            num_errors += 1
        if is_neg:
            neg_errors += 1
            
    return num_errors, neg_errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate STT output.")
    parser.add_argument("reference", help="Reference transcript text file.")
    parser.add_argument("hypothesis", help="Hypothesis file (raw text or stt_result.json).")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    hyp_path = Path(args.hypothesis)

    if not ref_path.exists() or not hyp_path.exists():
        print("Error: Missing input files.")
        sys.exit(1)

    ref_text = ref_path.read_text(encoding="utf-8").strip()
    
    hyp_text = ""
    if hyp_path.suffix == ".json":
        try:
            data = json.loads(hyp_path.read_text(encoding="utf-8"))
            hyp_text = data.get("raw_text", "")
        except Exception:
            hyp_text = hyp_path.read_text(encoding="utf-8").strip()
    else:
        hyp_text = hyp_path.read_text(encoding="utf-8").strip()

    # Pre-process for basic WER (lowercase, strip punctuation)
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    ref_norm = transform(ref_text)
    hyp_norm = transform(hyp_text)
    
    out = jiwer.process_words(ref_norm, hyp_norm)
    
    num_err, neg_err = count_category_errors(ref_norm.split(), hyp_norm.split())
    
    print("\n# STT Evaluation Report\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Word Error Rate (WER) | {out.wer * 100:.2f}% |")
    print(f"| Word Info Lost (WIL)  | {out.wil * 100:.2f}% |")
    print(f"| Total Words (Ref)     | {len(ref_norm.split())} |")
    print(f"| Number Errors         | {num_err} |")
    print(f"| Negation Errors       | {neg_err} |")
    print()


if __name__ == "__main__":
    main()
