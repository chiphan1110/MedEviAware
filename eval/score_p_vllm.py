#!/usr/bin/env python3
"""Score each case's chance of being answered correctly, against a vLLM OpenAI-compatible server.

Input:
- --prompts-jsonl: score_prompts.jsonl from 03_build_score_prompts.py
  (one row per case x fact set x option order, with gold_letter and letter_map)

Behavior:
- send every row as a chat request asking for 1 token with the top --top-logprobs logprobs
- read the probabilities of the option letters from the first token, renormalise them over the
  options, and record the probability of the gold letter plus how much mass the letters had
- write per-row predictions to predictions.jsonl as they finish (predicted letter, correct or not,
  letter probabilities), so an interrupted run resumes
- average the option orders per (case, fact set) on the original letters and write
  scores.jsonl: p (chance of the gold answer) and pred (most likely original letter) per fact set
- write score_stats.json: mean p and accuracy per fact set and split, how often exam / tests raise
  p, and how many rows had little probability on the option letters (a format warning)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


FACT_SETS = ("H", "H+E", "H+T", "H+E+T")
LOW_MASS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score P(correct) per case and fact set via a vLLM server.")
    parser.add_argument("--prompts-jsonl", type=Path, required=True, help="Scoring prompts from step 03.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Writes predictions.jsonl, scores.jsonl, score_stats.json.")
    parser.add_argument("--api-model", type=str, required=True, help="Model name as served by vLLM (--served-model-name).")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"), help="Server API key.")
    parser.add_argument("--top-logprobs", type=int, default=20, help="Logprobs returned for the first token.")
    parser.add_argument("--workers", type=int, default=64, help="Parallel requests.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Seconds per request.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Only score the first N cases (0 = all), for a pilot.")
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Do not pass enable_thinking=False to the chat template (only matters for hybrid-thinking models).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def letter_probs(top_logprobs: list[Any], letters: list[str]) -> tuple[dict[str, float], float, str]:
    """Probabilities of each option letter in the first token, renormalised, plus the raw letter mass."""
    raw = dict.fromkeys(letters, 0.0)
    for item in top_logprobs:
        token = item.token.strip().strip(".:()").upper()
        if token in raw:
            raw[token] += math.exp(item.logprob)
    mass = sum(raw.values())
    probs = {k: v / mass for k, v in raw.items()} if mass > 0 else {k: 1.0 / len(letters) for k in letters}
    top_token = top_logprobs[0].token if top_logprobs else ""
    return probs, mass, top_token


def score_row(client: OpenAI, args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    extra_body = None if args.keep_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(
        model=args.api_model,
        messages=row["messages"],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=args.top_logprobs,
        extra_body=extra_body,
    )
    content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
    top = content[0].top_logprobs if content else []
    letters = sorted(row["letter_map"])
    probs, mass, top_token = letter_probs(top, letters)
    pred = max(probs, key=probs.get)
    return {
        "row_id": row["row_id"],
        "case_id": row["case_id"],
        "split": row["split"],
        "evidence": row["evidence"],
        "fact_set": row["fact_set"],
        "order": row["order"],
        "output": top_token,
        "pred_letter": pred,
        "gold_letter": row["gold_letter"],
        "correct": pred == row["gold_letter"],
        "pred_original": row["letter_map"][pred],
        "gold_original": row["letter_map"][row["gold_letter"]],
        "p_gold": round(probs[row["gold_letter"]], 4),
        "probs": {k: round(v, 4) for k, v in probs.items()},
        "letter_mass": round(mass, 4),
        "letter_map": row["letter_map"],
    }


def aggregate(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average option orders per (case, fact set) on the original letters."""
    grouped: dict[Any, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        grouped[r["case_id"]][r["fact_set"]].append(r)

    cases = []
    for case_id in sorted(grouped):
        first = next(iter(grouped[case_id].values()))[0]
        gold = first["letter_map"][first["gold_letter"]]
        out = {"case_id": case_id, "split": first["split"], "evidence": first["evidence"], "gold": gold,
               "p": {}, "pred": {}, "n_orders": {}, "min_letter_mass": 1.0}
        for fact_set in FACT_SETS:
            rows = grouped[case_id].get(fact_set)
            if not rows:
                continue
            avg = defaultdict(float)
            for r in rows:
                for shown, prob in r["probs"].items():
                    avg[r["letter_map"][shown]] += prob / len(rows)
            out["p"][fact_set] = round(avg[gold], 4)
            out["pred"][fact_set] = max(avg, key=avg.get)
            out["n_orders"][fact_set] = len(rows)
            out["min_letter_mass"] = round(min(out["min_letter_mass"], *(r["letter_mass"] for r in rows)), 4)
        cases.append(out)
    return cases


def summarize(cases: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        stats: dict[str, Any] = {"n_cases": len(subset)}
        for fs in FACT_SETS:
            have = [c for c in subset if fs in c["p"]]
            if have:
                stats[fs] = {
                    "n": len(have),
                    "mean_p": round(sum(c["p"][fs] for c in have) / len(have), 4),
                    "accuracy": round(sum(c["pred"][fs] == c["gold"] for c in have) / len(have), 4),
                }
        for name, fs in (("exam_raises_p", "H+E"), ("tests_raises_p", "H+T"), ("all_raises_p", "H+E+T")):
            have = [c for c in subset if fs in c["p"]]
            if have:
                stats[name] = round(sum(c["p"][fs] - c["p"]["H"] >= 0.1 for c in have) / len(have), 4)
        return stats

    splits = sorted({c["split"] for c in cases})
    return {
        "overall": block(cases),
        "by_split": {s: block([c for c in cases if c["split"] == s]) for s in splits},
        "rows": len(raw_rows),
        "mean_letter_mass": round(sum(r["letter_mass"] for r in raw_rows) / max(len(raw_rows), 1), 4),
        "rows_low_letter_mass": sum(r["letter_mass"] < LOW_MASS for r in raw_rows),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "predictions.jsonl"

    rows = load_jsonl(args.prompts_jsonl)
    if args.limit_cases > 0:
        keep_ids = list(dict.fromkeys(r["case_id"] for r in rows))[: args.limit_cases]
        keep = set(keep_ids)
        rows = [r for r in rows if r["case_id"] in keep]

    done = {r["row_id"] for r in load_jsonl(raw_path)}
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[score] {len(rows)} prompts | {len(rows) - len(todo)} already done | {len(todo)} to score with {args.api_model}")

    if todo:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.request_timeout)
        lock = threading.Lock()
        with raw_path.open("a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(score_row, client, args, r) for r in todo]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="scoring"):
                result = fut.result()
                with lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    wanted = {r["row_id"] for r in rows}
    raw_rows = [r for r in load_jsonl(raw_path) if r["row_id"] in wanted]
    cases = aggregate(raw_rows)
    with (args.out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats = summarize(cases, raw_rows)
    (args.out_dir / "score_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    overall = stats["overall"]
    print(f"[score] {overall['n_cases']} cases scored -> {args.out_dir}")
    for fs in FACT_SETS:
        if fs in overall:
            print(f"[score] {fs:6s} n={overall[fs]['n']:4d}  mean p={overall[fs]['mean_p']:.3f}  accuracy={overall[fs]['accuracy']:.3f}")
    for name in ("exam_raises_p", "tests_raises_p", "all_raises_p"):
        if name in overall:
            print(f"[score] {name}: {overall[name]:.3f} of cases gain >= 0.1")
    print(f"[score] mean letter mass {stats['mean_letter_mass']:.3f}; rows below {LOW_MASS}: {stats['rows_low_letter_mass']}")


if __name__ == "__main__":
    main()
