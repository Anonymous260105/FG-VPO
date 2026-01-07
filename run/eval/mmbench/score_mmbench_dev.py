#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMBench-dev Local Scoring Script
- Supports EN / CN dev (both are .tsv, containing public answers)
- Prediction file is in LLaVA-style .jsonl (each line contains question_id and text)
- Outputs two types of accuracy: Overall accuracy and valid sample accuracy (only those with A-D choices will be scored)
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional, Tuple

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Score MMBench dev predictions locally.")
    ap.add_argument("--tsv", required=True, help="Path to MMBench dev tsv (EN or CN).")
    ap.add_argument("--pred", required=True, help="Path to predictions .jsonl.")
    ap.add_argument("--dump-errors", default=None, help="Optional: path to write CSV of mismatches/unparsed.")
    return ap.parse_args()


def find_answer_column(df: pd.DataFrame) -> str:
    """
    Compatible with different column names: answer / gt / label / answer_letter
    """
    candidates = ["answer", "gt", "label", "answer_letter"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Answer column not found, existing columns: {list(df.columns)}")


def find_index_column(df: pd.DataFrame) -> str:
    """
    Compatible with different question index columns: index / question_id / qid
    """
    candidates = ["index", "question_id", "qid"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Question index column not found, existing columns: {list(df.columns)}")


_letter_pat = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def pick_letter(text: str) -> Optional[str]:
    """
    Extract the first A-D option letter from the model's output.
    Compatible with common formats: 'A', '(A)', 'Answer: A', '选A', 'I choose B', etc., as long as A-D appears independently.
    """
    if text is None:
        return None
    m = _letter_pat.search(str(text))
    return m.group(1).upper() if m else None


def load_gt(tsv_path: str) -> Tuple[Dict, pd.DataFrame, str, str]:
    df = pd.read_csv(tsv_path, sep="\t")
    idx_col = find_index_column(df)
    ans_col = find_answer_column(df)
    # Standardize strings and remove whitespaces
    gtd = dict(zip(df[idx_col], df[ans_col].astype(str).str.strip().str.upper()))
    return gtd, df, idx_col, ans_col


def load_pred(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            yield j


def fmt_pct(x: int, y: int) -> str:
    return f"{(x / y * 100):.2f}%" if y > 0 else "N/A"


def main():
    args = parse_args()

    if not os.path.exists(args.tsv):
        print(f"[Error] TSV does not exist: {args.tsv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pred):
        print(f"[Error] Prediction file does not exist: {args.pred}", file=sys.stderr)
        sys.exit(1)

    gt_map, df, idx_col, ans_col = load_gt(args.tsv)

    tot_all = 0
    correct_all = 0
    tot_valid = 0
    correct_valid = 0

    # Record errors/unparsed for troubleshooting
    err_rows = []

    for j in load_pred(args.pred):
        qid = j.get("question_id")
        # LLaVA commonly uses "text", also supports "prediction"/"answer"
        text = j.get("text", j.get("prediction", j.get("answer", "")))
        pred_letter = pick_letter(text)
        gt_letter = gt_map.get(qid)

        # Count for "All samples"
        tot_all += 1
        if pred_letter == gt_letter:
            correct_all += 1

        # Count for "Valid samples"
        if pred_letter is not None:
            tot_valid += 1
            if pred_letter == gt_letter:
                correct_valid += 1

        # Collect errors or unparsed
        if (pred_letter is None) or (pred_letter != gt_letter):
            err_rows.append({
                "question_id": qid,
                "pred_text": text,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter
            })

    print("== MMBench-dev Local Evaluation Results ==")
    print(f"Total number of questions (prediction records): {tot_all}")
    print(f"Valid predictions (A-D parsed): {tot_valid}")
    print(f"Acc (All samples, unparsed counted as wrong): {fmt_pct(correct_all, tot_all)}  ({correct_all}/{tot_all})")
    print(f"Acc (Only valid samples)       : {fmt_pct(correct_valid, tot_valid)}  ({correct_valid}/{tot_valid})")
    print(f"Entries not parsed as A-D: {tot_all - tot_valid}")

    if args.dump_errors:
        os.makedirs(os.path.dirname(args.dump_errors), exist_ok=True)
        pd.DataFrame(err_rows).to_csv(args.dump_errors, index=False, encoding="utf-8")
        print(f"[Info] Error samples have been saved to: {args.dump_errors}")


if __name__ == "__main__":
    main()
