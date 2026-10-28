#!/usr/bin/env python3
"""Write the episode items (train / val / test): one plain QA item per (case, availability version).

Input:
- --tagged-jsonl: cases_tagged.jsonl from 01_tag_facts.py (facts per group)
- --labels-jsonl: labels.jsonl from 04_make_labels.py (status, best action and path, p)

Behavior:
- one item per labelled (case, version), written to <split>.jsonl, in the same item style as the
  PathScale QA json (No, question, options, answer, extra_info) so a parquet builder can add the
  instruction prompt; there are no chat roles in the items
- "question" is the full stem the policy sees: the patient information shown at the start
  (history plus every given group), the evidence status lines and the clinical question
- "options" keep their original order and "answer" is the correct option letter
- "target" is what the best first action looks like inside <answer> </answer>: the answer letter,
  "REQUEST: EXAM", "REQUEST: TESTS" or "DEFER"
- extra_info keeps what evaluation and the reward need: split, statuses, held-out flag, best path,
  the score of every action, the p values and the facts that can be revealed on request
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from case_format import render_shown_info, render_status


SPLITS = ("train", "val", "test")
ACTION_TARGET = {"REQUEST_EXAM": "REQUEST: EXAM", "REQUEST_TESTS": "REQUEST: TESTS", "DEFER": "DEFER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one plain QA item per (case, availability version).")
    parser.add_argument("--tagged-jsonl", type=Path, required=True, help="Tagged cases from step 01.")
    parser.add_argument("--labels-jsonl", type=Path, required=True, help="Labels from step 04.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Writes train.jsonl, val.jsonl, test.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.tagged_jsonl.open(encoding="utf-8") as f:
        cases = {c["case_id"]: c for c in map(json.loads, f)}
    with args.labels_jsonl.open(encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {s: (args.output_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    counts = Counter()
    try:
        for no, lab in enumerate(labels):
            case = cases[lab["case_id"]]
            status = {"exam": lab["exam_status"], "tests": lab["tests_status"]}
            stem = "\n\n".join([render_shown_info(case, status), render_status(status), f"Question: {case['question']}"])
            item = {
                "No": no,
                "id": f"{lab['case_id']}|{lab['exam_status']}|{lab['tests_status']}",
                "question": stem,
                "options": case["options"],
                "answer": case["answer_idx"],
                "target": case["answer_idx"] if lab["label"] == "ANSWER" else ACTION_TARGET[lab["label"]],
                "extra_info": {
                    "split": lab["split"],
                    "case_id": lab["case_id"],
                    "evidence": lab["evidence"],
                    "exam_status": lab["exam_status"],
                    "tests_status": lab["tests_status"],
                    "heldout_version": lab["heldout_version"],
                    "label": lab["label"],
                    "best_path": lab["best_path"],
                    "best_value": lab["best_value"],
                    "action_scores": lab["action_scores"],
                    "p": lab["p"],
                    "revealable": {g: case[g] for g in ("exam", "tests") if status[g] == "obtainable"},
                },
            }
            files[lab["split"]].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[(lab["split"], lab["heldout_version"])] += 1
    finally:
        for f in files.values():
            f.close()

    for s in SPLITS:
        seen, ho = counts[(s, False)], counts[(s, True)]
        print(f"[05] {s:5s} {seen + ho:5d} items ({seen} seen versions, {ho} held-out) -> {args.output_dir / f'{s}.jsonl'}")


if __name__ == "__main__":
    main()
