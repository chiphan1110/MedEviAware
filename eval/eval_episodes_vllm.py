#!/usr/bin/env python3
"""Multi-turn episode evaluation against a vLLM OpenAI-compatible server.

Input:
- --episodes-jsonl: episode items from 05_build_episodes.py (No, id, question, options, answer, extra_info)
- --instruction-prompt-file: reasoning instruction (<think> </think> then one action in <answer> </answer>)

Behavior:
- turn 1: the item's stem + options + instruction prompt, as one user message
- each turn, read the last <answer> </answer>: an option letter ends the episode (correct or wrong),
  DEFER ends it with score 0, REQUEST: EXAM / REQUEST: TESTS costs --request-cost and the environment
  replies with the revealed facts (only if that group is available on request) or "not available"
- no valid <answer> ends the episode as a format error scored --invalid-score
- after --max-turns turns without an answer or defer, the episode ends as no decision (score 0)
- write every episode (all turns with the full reasoning text) to predictions.jsonl as it finishes,
  so an interrupted run resumes, and the metrics to metrics.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess" / "build_data"))
from case_format import GROUP_TITLES, render_followup, render_policy_prompt, render_reveal, render_status  # noqa: E402


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
LETTER_RE = re.compile(r"^(?:ANSWER\s*[:\-]?\s*)?\(?([A-Z])\)?(?:[.:)\s]|$)")
REQUEST_ALIASES = {
    "REQUESTEXAM": "REQUEST_EXAM",
    "REQUESTEXAMINATION": "REQUEST_EXAM",
    "REQUESTPHYSICALEXAM": "REQUEST_EXAM",
    "REQUESTPHYSICALEXAMINATION": "REQUEST_EXAM",
    "REQUESTTESTS": "REQUEST_TESTS",
    "REQUESTTEST": "REQUEST_TESTS",
    "REQUESTTESTRESULTS": "REQUEST_TESTS",
}
REQUEST_GROUP = {"REQUEST_EXAM": "exam", "REQUEST_TESTS": "tests"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-turn answer / request / defer episodes via a vLLM server.")
    parser.add_argument("--episodes-jsonl", type=Path, required=True, help="Episode items (e.g. test.jsonl).")
    parser.add_argument("--instruction-prompt-file", type=Path, required=True, help="Reasoning instruction prompt.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Writes predictions.jsonl and metrics.json.")
    parser.add_argument("--api-model", type=str, required=True, help="Model name as served by vLLM.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"), help="Server API key.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per turn.")
    parser.add_argument("--max-turns", type=int, default=3, help="Turns per episode before it ends without a decision.")
    parser.add_argument("--wrong-penalty", type=float, default=1.0, help="Score of a wrong answer is -wrong_penalty.")
    parser.add_argument("--request-cost", type=float, default=0.05, help="Cost of each request.")
    parser.add_argument("--invalid-score", type=float, default=-1.0, help="Score when no valid <answer> is given.")
    parser.add_argument("--workers", type=int, default=64, help="Parallel episodes.")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="Seconds per request.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N episodes (0 = all).")
    parser.add_argument("--skip-heldout", action="store_true", help="Leave out held-out versions.")
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


def parse_action(text: str, letters: list[str]) -> str | None:
    """Last <answer> content as an option letter, REQUEST_EXAM, REQUEST_TESTS or DEFER; None if invalid."""
    found = ANSWER_TAG_RE.findall(text or "")
    if not found:
        return None
    raw = found[-1].strip()
    squashed = re.sub(r"[^A-Z]", "", raw.upper())
    if squashed in REQUEST_ALIASES:
        return REQUEST_ALIASES[squashed]
    if squashed == "DEFER":
        return "DEFER"
    m = LETTER_RE.match(raw.upper())
    if m and m.group(1) in letters:
        return m.group(1)
    return None


def run_episode(client: OpenAI, args: argparse.Namespace, instruction: str, item: dict[str, Any]) -> dict[str, Any]:
    extra = item["extra_info"]
    letters = sorted(item["options"])
    status = {"exam": extra["exam_status"], "tests": extra["tests_status"]}
    case_facts = dict(extra.get("revealable", {}))
    messages = [{"role": "user", "content": render_policy_prompt(item["question"], item["options"], instruction)}]
    extra_body = None if args.enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}

    turns, requests = [], []
    outcome, final_action = "no_decision", None
    for turn in range(1, args.max_turns + 1):
        resp = client.chat.completions.create(
            model=args.api_model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        text = resp.choices[0].message.content or ""
        action = parse_action(text, letters)
        turns.append({"turn": turn, "response": text, "action": action, "finish_reason": resp.choices[0].finish_reason})
        messages.append({"role": "assistant", "content": text})

        if action is None:
            outcome = "format_error"
            break
        if action == "DEFER":
            outcome, final_action = "defer", action
            break
        if action in letters:
            outcome = "correct" if action == item["answer"] else "wrong"
            final_action = action
            break

        group = REQUEST_GROUP[action]
        revealed = status[group] == "obtainable"
        requests.append({"turn": turn, "action": action, "status_before": status[group], "revealed": revealed})
        if revealed:
            status[group] = "given"
        if turn < args.max_turns:
            if revealed:
                reveal = render_reveal(case_facts, group, status)
            else:
                note = "already shown above" if status[group] == "given" else "not available"
                verb = "are" if group == "tests" else "is"
                reveal = f"The {GROUP_TITLES[group].lower()} {verb} {note}.\n\n{render_status(status)}"
            messages.append({"role": "user", "content": render_followup(reveal, last_turn=turn + 1 == args.max_turns)})

    score = {
        "correct": 1.0,
        "wrong": -args.wrong_penalty,
        "defer": 0.0,
        "no_decision": 0.0,
        "format_error": args.invalid_score,
    }[outcome] - args.request_cost * len(requests)

    return {
        "No": item["No"],
        "id": item["id"],
        "split": extra["split"],
        "evidence": extra["evidence"],
        "exam_status": extra["exam_status"],
        "tests_status": extra["tests_status"],
        "heldout_version": extra["heldout_version"],
        "answer": item["answer"],
        "target": item["target"],
        "label": extra["label"],
        "best_path": extra["best_path"],
        "first_action": turns[0]["action"] if turns else None,
        "final_action": final_action,
        "outcome": outcome,
        "score": round(score, 4),
        "n_requests": len(requests),
        "requests": requests,
        "n_turns": len(turns),
        "turns": turns,
    }


def summarize(preds: list[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        outcomes = Counter(r["outcome"] for r in rows)
        answered = outcomes["correct"] + outcomes["wrong"]
        n_requests = sum(r["n_requests"] for r in rows)
        label_request = [r for r in rows if r["label"].startswith("REQUEST")]
        label_no_request = [r for r in rows if not r["label"].startswith("REQUEST")]
        first_matches = sum(
            (r["first_action"] == r["label"]) or (r["label"] == "ANSWER" and r["first_action"] == r["answer"]) for r in rows
        )
        return {
            "n": n,
            "avg_score": round(sum(r["score"] for r in rows) / n, 4),
            "accuracy_when_answered": round(outcomes["correct"] / answered, 4) if answered else None,
            "answer_rate": round(answered / n, 4),
            "defer_rate": round(outcomes["defer"] / n, 4),
            "format_error_rate": round(outcomes["format_error"] / n, 4),
            "no_decision_rate": round(outcomes["no_decision"] / n, 4),
            "requests_per_episode": round(n_requests / n, 4),
            "invalid_request_share": round(
                sum(not q["revealed"] for r in rows for q in r["requests"]) / n_requests, 4
            ) if n_requests else None,
            "requested_when_label_requests": round(
                sum(r["n_requests"] > 0 for r in label_request) / len(label_request), 4
            ) if label_request else None,
            "requested_when_label_does_not": round(
                sum(r["n_requests"] > 0 for r in label_no_request) / len(label_no_request), 4
            ) if label_no_request else None,
            "first_action_matches_label": round(first_matches / n, 4),
            "outcomes": dict(outcomes),
        }

    by_version = defaultdict(list)
    for r in preds:
        by_version[f"exam={r['exam_status']}|tests={r['tests_status']}"].append(r)
    return {
        "overall": block(preds),
        "seen_versions": block([r for r in preds if not r["heldout_version"]]),
        "heldout_versions": block([r for r in preds if r["heldout_version"]]),
        "by_version": {k: block(v) for k, v in sorted(by_version.items())},
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "predictions.jsonl"
    instruction = args.instruction_prompt_file.read_text(encoding="utf-8").strip()

    items = load_jsonl(args.episodes_jsonl)
    if args.skip_heldout:
        items = [it for it in items if not it["extra_info"]["heldout_version"]]
    if args.limit > 0:
        items = items[: args.limit]

    done = {p["id"] for p in load_jsonl(pred_path)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[episodes] {len(items)} episodes | {len(items) - len(todo)} already done | {len(todo)} to run with {args.api_model}")

    if todo:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.request_timeout)
        lock = threading.Lock()
        with pred_path.open("a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run_episode, client, args, instruction, it) for it in todo]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="episodes"):
                result = fut.result()
                with lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    wanted = {it["id"] for it in items}
    preds = [p for p in load_jsonl(pred_path) if p["id"] in wanted]
    metrics = summarize(preds)
    metrics["settings"] = {
        "api_model": args.api_model,
        "episodes_jsonl": str(args.episodes_jsonl),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "wrong_penalty": args.wrong_penalty,
        "request_cost": args.request_cost,
        "invalid_score": args.invalid_score,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for name in ("overall", "seen_versions", "heldout_versions"):
        m = metrics[name]
        if not m["n"]:
            continue
        acc = "-" if m["accuracy_when_answered"] is None else f"{m['accuracy_when_answered']:.3f}"
        print(
            f"[episodes] {name:16s} n={m['n']:5d}  score={m['avg_score']:+.3f}  acc_answered={acc}  "
            f"answer={m['answer_rate']:.2f}  defer={m['defer_rate']:.2f}  format_err={m['format_error_rate']:.2f}  "
            f"requests/ep={m['requests_per_episode']:.2f}"
        )
    print(f"[episodes] wrote {pred_path} and {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
