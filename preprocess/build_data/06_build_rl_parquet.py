#!/usr/bin/env python3
"""Build the verl GRPO parquets (train / val) from the episode items.

Input:
- --episodes-dir: train.jsonl and val.jsonl from 05_build_episodes.py

Behavior:
- keep only seen versions (held-out versions stay test-only)
- prompt = one user message rendered by case_format.render_policy_prompt, exactly as the episode
  evaluation renders turn 1 (stem, options, reasoning instruction)
- train: rotate each item's options so the correct letter cycles evenly over A-D (the raw train
  split leans to C); val keeps the original option order, as the evaluation does
- reward_model.ground_truth = the correct letter as shown; extra_info holds what the reward needs:
  the evidence statuses and the score of every action from the labels (null when not allowed)
- columns follow the PathScale parquets: data_source, prompt, ability, reward_model, extra_info
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from case_format import reorder_options, render_policy_prompt


DATA_SOURCE = "medeviaware"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verl GRPO parquets from episode items.")
    parser.add_argument("--episodes-dir", type=Path, required=True, help="Folder with train.jsonl and val.jsonl.")
    parser.add_argument("--instruction-prompt-file", type=Path, required=True, help="Policy reasoning instruction.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Writes train.parquet and val.parquet.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the option rotations of train items.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_row(item: dict, instruction: str, order: str) -> dict:
    extra = item["extra_info"]
    ordered = reorder_options(item["options"], order)
    shown = {new: text for new, _, text in ordered}
    gold_shown = next(new for new, old, _ in ordered if old == item["answer"])
    scores = extra["action_scores"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": render_policy_prompt(item["question"], shown, instruction)}],
        "ability": "clinical_decision",
        "reward_model": {"style": "rule", "ground_truth": gold_shown},
        "extra_info": {
            "index": item["No"],
            "id": item["id"],
            "split": extra["split"],
            "option_order": order,
            "options": "".join(sorted(shown)),
            "exam_status": extra["exam_status"],
            "tests_status": extra["tests_status"],
            "label": extra["label"],
            "best_value": extra["best_value"],
            "score_answer_p": scores["ANSWER"],
            "score_defer": scores["DEFER"],
            "score_request_exam": scores.get("REQUEST_EXAM"),
            "score_request_tests": scores.get("REQUEST_TESTS"),
        },
    }


def balanced_orders(items: list[dict], seed: int) -> list[str]:
    """One rotation per item so the correct letter as shown cycles A, B, C, D over a shuffled item order."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    orders = [""] * len(items)
    for rank, i in enumerate(idx):
        letters = sorted(items[i]["options"])
        target = letters[rank % len(letters)]
        k = (letters.index(items[i]["answer"]) - letters.index(target)) % len(letters)
        orders[i] = f"rot{k}"
    return orders


def main() -> None:
    args = parse_args()
    instruction = args.instruction_prompt_file.read_text(encoding="utf-8").strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        items = [it for it in load_jsonl(args.episodes_dir / f"{split}.jsonl") if not it["extra_info"]["heldout_version"]]
        orders = balanced_orders(items, args.seed) if split == "train" else ["rot0"] * len(items)
        rows = [build_row(it, instruction, order) for it, order in zip(items, orders)]
        out_path = args.output_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)

        gold = Counter(r["reward_model"]["ground_truth"] for r in rows)
        labels = Counter(r["extra_info"]["label"] for r in rows)
        chars = sorted(len(r["prompt"][0]["content"]) for r in rows)
        print(f"[06] {split:5s} {len(rows):5d} rows -> {out_path}")
        print(f"[06]       gold letter {dict(sorted(gold.items()))} | labels {dict(labels)}")
        print(f"[06]       prompt chars median {chars[len(chars) // 2]}, max {chars[-1]} (about {chars[-1] // 3} tokens at most)")


if __name__ == "__main__":
    main()
