#!/usr/bin/env python3
"""Score each case's chance of being answered correctly with reasoning, against a vLLM server.

Input:
- --prompts-jsonl: score_prompts.jsonl from 03_build_score_prompts.py
  (one row per case x fact set x option order, with the plain-text prompt, gold_letter and letter_map)

Behavior:
- send every row as one user message and sample --samples reasoning answers
  (<think> ... </think><answer>X</answer>) at --temperature, the same answering style the policy uses
- check each sample's <answer> letter; a sample without a valid letter counts as not correct
- write per-row results to predictions.jsonl as they finish (every sample with its reasoning text
  unless --no-save-responses), so an interrupted run resumes
- per (case, fact set), p = share of correct samples over all option orders; pred = the most common
  answer mapped back to the original letters; write scores.jsonl
- write score_stats.json: mean p and majority accuracy per fact set and split, how often exam / tests
  raise p, the format error rate and the mean reasoning length
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from eval_episodes_vllm import parse_action


FACT_SETS = ("H", "H+E", "H+T", "H+E+T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score P(correct) with reasoning per case and fact set via a vLLM server.")
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
    parser.add_argument("--samples", type=int, default=2, help="Sampled answers per prompt (per option order).")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per sample.")
    parser.add_argument("--workers", type=int, default=64, help="Parallel requests.")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="Seconds per request.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Only score the first N cases (0 = all), for a pilot.")
    parser.add_argument("--no-save-responses", action="store_true", help="Do not store the reasoning text of each sample.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Let hybrid-thinking models use their native thinking mode (default off; reasoning goes in <think> tags).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_row(client: OpenAI, args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    extra_body = None if args.enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(
        model=args.api_model,
        messages=[{"role": "user", "content": row["prompt"]}],
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )
    letters = sorted(row["letter_map"])
    samples = []
    for choice in resp.choices:
        text = choice.message.content or ""
        pred = parse_action(text, letters)
        pred = pred if pred in letters else None
        sample = {
            "pred_letter": pred,
            "pred_original": row["letter_map"][pred] if pred else None,
            "correct": pred == row["gold_letter"],
            "finish_reason": choice.finish_reason,
        }
        if not args.no_save_responses:
            sample["response"] = text
        samples.append(sample)
    n_correct = sum(s["correct"] for s in samples)
    return {
        "row_id": row["row_id"],
        "case_id": row["case_id"],
        "split": row["split"],
        "evidence": row["evidence"],
        "fact_set": row["fact_set"],
        "order": row["order"],
        "gold_letter": row["gold_letter"],
        "gold_original": row["letter_map"][row["gold_letter"]],
        "letter_map": row["letter_map"],
        "n_samples": len(samples),
        "n_correct": n_correct,
        "n_format_error": sum(s["pred_letter"] is None for s in samples),
        "p_gold": round(n_correct / max(len(samples), 1), 4),
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "samples": samples,
    }


def aggregate(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool the samples of every option order per (case, fact set)."""
    grouped: dict[Any, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        grouped[r["case_id"]][r["fact_set"]].append(r)

    cases = []
    for case_id in sorted(grouped):
        first = next(iter(grouped[case_id].values()))[0]
        out = {"case_id": case_id, "split": first["split"], "evidence": first["evidence"], "gold": first["gold_original"],
               "p": {}, "pred": {}, "n_samples": {}, "format_errors": 0}
        for fact_set in FACT_SETS:
            rows = grouped[case_id].get(fact_set)
            if not rows:
                continue
            n = sum(r["n_samples"] for r in rows)
            votes = Counter(s["pred_original"] for r in rows for s in r["samples"] if s["pred_original"])
            out["p"][fact_set] = round(sum(r["n_correct"] for r in rows) / max(n, 1), 4)
            out["pred"][fact_set] = votes.most_common(1)[0][0] if votes else None
            out["n_samples"][fact_set] = n
            out["format_errors"] += sum(r["n_format_error"] for r in rows)
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
                    "majority_accuracy": round(sum(c["pred"][fs] == c["gold"] for c in have) / len(have), 4),
                }
        for name, fs in (("exam_raises_p", "H+E"), ("tests_raises_p", "H+T"), ("all_raises_p", "H+E+T")):
            have = [c for c in subset if fs in c["p"]]
            if have:
                stats[name] = round(sum(c["p"][fs] - c["p"]["H"] >= 0.1 for c in have) / len(have), 4)
                stats[name.replace("raises", "lowers")] = round(sum(c["p"][fs] - c["p"]["H"] <= -0.1 for c in have) / len(have), 4)
        return stats

    n_samples = sum(r["n_samples"] for r in raw_rows)
    tokens = [r["completion_tokens"] for r in raw_rows if r.get("completion_tokens")]
    splits = sorted({c["split"] for c in cases})
    return {
        "overall": block(cases),
        "by_split": {s: block([c for c in cases if c["split"] == s]) for s in splits},
        "rows": len(raw_rows),
        "samples": n_samples,
        "format_error_rate": round(sum(r["n_format_error"] for r in raw_rows) / max(n_samples, 1), 4),
        "truncated_samples": sum(s["finish_reason"] == "length" for r in raw_rows for s in r["samples"]),
        "mean_completion_tokens_per_sample": round(sum(tokens) / max(n_samples, 1), 1) if tokens else None,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "predictions.jsonl"

    rows = load_jsonl(args.prompts_jsonl)
    if args.limit_cases > 0:
        keep = set(list(dict.fromkeys(r["case_id"] for r in rows))[: args.limit_cases])
        rows = [r for r in rows if r["case_id"] in keep]

    previous = load_jsonl(raw_path)
    if any("samples" not in r for r in previous):
        raise SystemExit(f"{raw_path} holds results from the old direct (one-token) scoring. Move that folder away first.")
    done = {r["row_id"] for r in previous}
    todo = [r for r in rows if r["row_id"] not in done]
    print(
        f"[score] {len(rows)} prompts x {args.samples} samples | {len(rows) - len(todo)} prompts already done | "
        f"{len(todo)} to score with {args.api_model}"
    )

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
    stats["settings"] = {"api_model": args.api_model, "samples": args.samples, "temperature": args.temperature,
                         "top_p": args.top_p, "max_tokens": args.max_tokens}
    (args.out_dir / "score_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    overall = stats["overall"]
    print(f"[score] {overall['n_cases']} cases scored -> {args.out_dir}")
    for fs in FACT_SETS:
        if fs in overall:
            print(f"[score] {fs:6s} n={overall[fs]['n']:4d}  mean p={overall[fs]['mean_p']:.3f}  "
                  f"majority accuracy={overall[fs]['majority_accuracy']:.3f}")
    for name in ("exam", "tests", "all"):
        if f"{name}_raises_p" in overall:
            print(f"[score] adding {name:5s}: p rises >= 0.1 in {overall[f'{name}_raises_p']:.3f}, "
                  f"falls >= 0.1 in {overall[f'{name}_lowers_p']:.3f} of cases")
    print(f"[score] format errors {stats['format_error_rate']:.3f} | truncated samples {stats['truncated_samples']} | "
          f"mean tokens/sample {stats['mean_completion_tokens_per_sample']}")


if __name__ == "__main__":
    main()
