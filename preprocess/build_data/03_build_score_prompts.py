#!/usr/bin/env python3
"""Build the scoring prompts: one row per (case, fact set, option order).

Input:
- --tagged-jsonl: cases_tagged.jsonl from 01_tag_facts.py
- --split-csv: case split from 02_split_cases.py

Behavior:
- use only cases listed in the split CSV
- for each case, build a prompt for every fact set it has: H, H+E, H+T, H+E+T
  (exam-only cases get H and H+E, tests-only cases get H and H+T)
- show the options in each order from --orders (rot0..rot3: every option once at every
  position) to average out letter and position bias
- store the shown->original letter map and the gold letter as shown, so the scoring step can
  read the probability of the correct option from the model's next-token logprobs
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from case_format import OPTION_ORDERS, available_fact_sets, build_score_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one scoring prompt per case, fact set and option order.")
    parser.add_argument("--tagged-jsonl", type=Path, required=True, help="Tagged cases from step 01.")
    parser.add_argument("--split-csv", type=Path, required=True, help="Split CSV from step 02.")
    parser.add_argument("--system-prompt-file", type=Path, required=True, help="System prompt for answering.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Scoring prompts.")
    parser.add_argument("--orders", type=str, default=",".join(OPTION_ORDERS), help="Comma-separated option orders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    unknown = set(orders) - set(OPTION_ORDERS)
    if unknown:
        raise ValueError(f"Unknown option orders: {sorted(unknown)}")
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    with args.split_csv.open(newline="", encoding="utf-8") as f:
        split_rows = {int(r["case_id"]): r for r in csv.DictReader(f)}
    with args.tagged_jsonl.open(encoding="utf-8") as f:
        cases = [c for c in map(json.loads, f) if c["case_id"] in split_rows]
    cases.sort(key=lambda c: c["case_id"])

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for case in cases:
            meta = split_rows[case["case_id"]]
            for fact_set in available_fact_sets(case):
                for order in orders:
                    messages, letter_map, gold_shown = build_score_messages(case, fact_set, order, system_prompt)
                    row = {
                        "row_id": f"{case['case_id']}|{fact_set}|{order}",
                        "case_id": case["case_id"],
                        "split": meta["split"],
                        "evidence": meta["evidence"],
                        "fact_set": fact_set,
                        "order": order,
                        "gold_letter": gold_shown,
                        "letter_map": letter_map,
                        "messages": messages,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[fact_set] += 1

    print(f"[03] {len(cases)} cases -> {sum(counts.values())} prompts in {args.output_jsonl}")
    print(f"[03] prompts per fact set: {dict(counts)} (orders: {', '.join(orders)})")


if __name__ == "__main__":
    main()
