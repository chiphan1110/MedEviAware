#!/usr/bin/env python3
"""Split the kept cases into train / val / test and write the split as a CSV.

Input:
- --tagged-jsonl: cases_tagged.jsonl from 01_tag_facts.py

Behavior:
- use only cases with keep == true
- split by case (every later version of a case stays in the case's split)
- stratify by evidence type (exam+tests / exam_only / tests_only) so each split has the same mix
- write one row per case: case_id, split, evidence, specialty
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from case_format import evidence_type


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a stratified train/val/test split of kept cases.")
    parser.add_argument("--tagged-jsonl", type=Path, required=True, help="Tagged cases from step 01.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Split CSV.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Share of cases for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Share of cases for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.tagged_jsonl.open(encoding="utf-8") as f:
        cases = [c for c in map(json.loads, f) if c["keep"]]

    by_evidence = defaultdict(list)
    for case in cases:
        by_evidence[evidence_type(case)].append(case)

    rng = random.Random(args.seed)
    rows = []
    for evidence in sorted(by_evidence):
        group = sorted(by_evidence[evidence], key=lambda c: c["case_id"])
        rng.shuffle(group)
        n_test = round(len(group) * args.test_ratio)
        n_val = round(len(group) * args.val_ratio)
        for i, case in enumerate(group):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append({"case_id": case["case_id"], "split": split, "evidence": evidence, "specialty": case["specialty"]})

    rows.sort(key=lambda r: r["case_id"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split", "evidence", "specialty"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["split"], r["evidence"]) for r in rows)
    print(f"[02] {len(rows)} kept cases -> {args.output_csv}")
    for split in SPLITS:
        detail = ", ".join(f"{e} {counts[(split, e)]}" for e in sorted(by_evidence))
        print(f"[02] {split:5s} {sum(counts[(split, e)] for e in by_evidence):4d} ({detail})")


if __name__ == "__main__":
    main()
