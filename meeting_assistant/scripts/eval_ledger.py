#!/usr/bin/env python
"""Evaluate ledger outputs against ground truth."""

import argparse
import json
import sys
from pathlib import Path


def eval_ledger(gt_path: Path, pred_path: Path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
        
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
        
    gt_decisions = {d["id"]: d for d in gt.get("decisions", [])}
    gt_tasks = {t["id"]: t for t in gt.get("tasks", [])}
    
    pred_decisions = {d["id"]: d for d in pred.get("decisions", [])}
    pred_tasks = {t["id"]: t for t in pred.get("action_items", [])}
    
    # False Confirmation Rate
    total_confirmed = 0
    false_confirmed = 0
    
    for pid, d in pred_decisions.items():
        if d["status"] == "AGREED":
            total_confirmed += 1
            if pid not in gt_decisions or gt_decisions[pid]["status"] != "AGREED":
                false_confirmed += 1
                
    for pid, t in pred_tasks.items():
        if t["status"] == "CONFIRMED":
            total_confirmed += 1
            if pid not in gt_tasks or gt_tasks[pid]["status"] != "CONFIRMED":
                false_confirmed += 1
                
    fc_rate = false_confirmed / total_confirmed if total_confirmed > 0 else 0.0
    
    # Invented Owner/Deadline Rate
    total_owner_deadlines = 0
    invented_owner_deadlines = 0
    
    for pid, t in pred_tasks.items():
        if t.get("owner") and t["owner"] != "Unspecified":
            total_owner_deadlines += 1
            if pid not in gt_tasks or t["owner"] != gt_tasks[pid].get("owner"):
                invented_owner_deadlines += 1
                
        if t.get("deadline") and t["deadline"] != "Unspecified":
            total_owner_deadlines += 1
            if pid not in gt_tasks or t["deadline"] != gt_tasks[pid].get("deadline"):
                invented_owner_deadlines += 1
                
    inv_rate = invented_owner_deadlines / total_owner_deadlines if total_owner_deadlines > 0 else 0.0
    
    # Ungrounded Item Rate (measured via warnings in our implementation)
    total_items = len(pred_decisions) + len(pred_tasks)
    ungrounded_items = 0
    for w in pred.get("metadata", {}).get("warnings", []):
        if "Dropped ungrounded" in w:
            # It was dropped before making it to output, so we need to add to total items 
            # to be accurate, or just count the warnings as a ratio.
            ungrounded_items += 1
            total_items += 1 
            
    ungrounded_rate = ungrounded_items / total_items if total_items > 0 else 0.0
    
    print("# Ledger Evaluation Report")
    print(f"| Metric | Rate |")
    print(f"|---|---|")
    print(f"| False Confirmation Rate | {fc_rate:.1%} ({false_confirmed}/{total_confirmed}) |")
    print(f"| Invented Owner/Deadline Rate | {inv_rate:.1%} ({invented_owner_deadlines}/{total_owner_deadlines}) |")
    print(f"| Ungrounded Item Rate | {ungrounded_rate:.1%} ({ungrounded_items}/{total_items}) |")
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", help="Ground truth JSON")
    parser.add_argument("pred", help="Predicted meeting_record.json")
    args = parser.parse_args()
    
    eval_ledger(Path(args.gt), Path(args.pred))
