#!/usr/bin/env python3
"""Sort each MediQ case's facts into history / exam / tests and keep patient-specific questions.

Input:
- --input-jsonl: raw MediQ file (e.g. all_dev_good.jsonl), one case per line

Behavior:
- strip the "N. " numbering from facts and flag cut-off facts locally (no API call)
- skip the API for cases that cannot be kept anyway (fewer than --min-facts usable facts)
- one OpenAI call per remaining case; send only the question and the numbered usable facts
  (no options, no answer) and receive a compact JSON {"q": "P"|"G", "E": [...], "T": [...]}
  listing the exam and test fact numbers; every other fact is history
- cache every API reply in --cache-jsonl (keyed by model, prompt and case) so reruns only
  call the API for cases not seen before
- fix obvious tagging mistakes with simple rules: a "history" fact with lab values or imaging
  becomes "tests", a "history" fact with vital signs becomes "exam", and an "exam" fact about
  imaging or ECG (without vital signs) becomes "tests"
- keep a case if its question is patient-specific, it has >= --min-history history facts and
  >= 1 exam or tests fact, and the answer text does not appear in the history
- write every case (kept or dropped, with drop_reason), a review sample and a stats file
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tqdm import tqdm


NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")
END_PUNCT_RE = re.compile(r"[.?!)\]\"'”%°]\s*$")
TRAILING_STOPWORDS = {
    "a", "an", "the", "of", "at", "to", "in", "on", "for", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "as", "from", "that", "which", "her", "his", "their",
}
TEST_RE = re.compile(
    r"(mg/dL|mmol/L|mEq/L|g/dL|U/L|ng/mL|pg/mL|/mm\^?3|/μL|\bserum\b|\blaborator|\bleukocyte"
    r"|\bhemoglobin\b|\bplatelet|\bx-ray|\bradiograph|\bCT\b|\bMRI\b|ultrasound|\bECG\b|\bEKG\b"
    r"|electrocardiogra|biopsy|\bculture\b|urinalysis|\bsmear\b)",
    re.IGNORECASE,
)
VITALS_RE = re.compile(
    r"(\btemperature\b|blood pressure|\bpulse\b|heart rate|\brespirations?\b|respiratory rate"
    r"|oxygen saturation|\bSpO2\b|mm ?Hg|°[FC])",
    re.IGNORECASE,
)
IMAGING_ECG_RE = re.compile(
    r"(\bx-ray|\bradiograph|\bCT\b|\bMRI\b|ultrasound|sonograph|\bECG\b|\bEKG\b|electrocardiogra|cardiac monitor)",
    re.IGNORECASE,
)
LETTER_TO_GROUP = {"H": "history", "E": "exam", "T": "tests"}
QTYPE_TO_NAME = {"P": "patient", "G": "general"}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "enum": ["P", "G"]},
                "E": {"type": "array", "items": {"type": "integer"}},
                "T": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["q", "E", "T"],
            "additionalProperties": False,
        },
    },
}
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
FATAL_ERROR_CODES = {"insufficient_quota", "credit_balance_exhausted", "invalid_api_key", "model_not_found"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag MediQ facts as history / exam / tests with an OpenAI model.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Raw MediQ jsonl.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Tagged cases (kept and dropped).")
    parser.add_argument("--cache-jsonl", type=Path, required=True, help="API reply cache; reused across runs.")
    parser.add_argument("--review-json", type=Path, required=True, help="Random kept cases for manual checking.")
    parser.add_argument("--stats-json", type=Path, required=True, help="Counts and token usage.")
    parser.add_argument("--prompt-file", type=Path, required=True, help="System prompt for the tagger.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model id.")
    parser.add_argument("--base-url", type=str, default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel API requests.")
    parser.add_argument("--max-format-retries", type=int, default=2, help="Re-asks when the reply has the wrong length.")
    parser.add_argument("--min-facts", type=int, default=3, help="Skip the API below this many usable facts.")
    parser.add_argument("--min-history", type=int, default=2, help="Minimum history facts to keep a case.")
    parser.add_argument("--min-answer-chars", type=int, default=5, help="Shortest answer text checked for leakage.")
    parser.add_argument("--review-size", type=int, default=30, help="Number of kept cases in the review sample.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N cases (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the review sample.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def looks_cut_off(text: str) -> bool:
    """A fact with no closing punctuation that ends on a filler word (e.g. "... is at the")."""
    if not text:
        return True
    if END_PUNCT_RE.search(text):
        return False
    words = text.split()
    return len(words) < 4 or words[-1].lower().strip(",;:") in TRAILING_STOPWORDS


def clean_facts(raw_facts: list[str]) -> list[dict[str, Any]]:
    facts = []
    for i, raw in enumerate(raw_facts, start=1):
        m = NUMBER_PREFIX_RE.match(raw)
        fact_id = int(m.group(0).strip().rstrip(".")) if m else i
        text = NUMBER_PREFIX_RE.sub("", raw).strip()
        facts.append({"id": fact_id, "text": text, "cut_off": looks_cut_off(text)})
    return facts


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def build_user_message(question: str, fact_texts: list[str]) -> str:
    lines = [f"Question: {question}", "Facts:"]
    lines += [f"{i}. {text}" for i, text in enumerate(fact_texts, start=1)]
    return "\n".join(lines)


def create_with_backoff(client: OpenAI, model: str, messages: list[dict[str, str]], max_tokens: int):
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_tokens,
                response_format=RESPONSE_FORMAT,
            )
        except RETRYABLE_ERRORS as e:
            # Out of credits or a bad key will not fix itself, so fail now instead of waiting.
            if getattr(e, "code", None) in FATAL_ERROR_CODES or attempt == 5:
                raise
            time.sleep(2**attempt)


def reply_to_letters(reply: dict[str, Any], n: int) -> str | None:
    """Turn {"E": [...], "T": [...]} into one H/E/T letter per fact, or None if the numbers are invalid."""
    exam, tests = set(reply["E"]), set(reply["T"])
    if exam & tests or not (exam | tests) <= set(range(1, n + 1)):
        return None
    return "".join("E" if i in exam else "T" if i in tests else "H" for i in range(1, n + 1))


def tag_case(
    client: OpenAI, model: str, system_prompt: str, question: str, fact_texts: list[str], max_format_retries: int
) -> dict[str, Any]:
    """Ask the tagger once, re-asking only if the reply lists invalid fact numbers."""
    n = len(fact_texts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(question, fact_texts)},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _ in range(max_format_retries + 1):
        resp = create_with_backoff(client, model, messages, max_tokens=24 + 3 * n)
        if resp.usage is not None:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
        content = resp.choices[0].message.content or ""
        try:
            reply = json.loads(content)
            qtype = reply["q"]
            letters = reply_to_letters(reply, n)
        except (json.JSONDecodeError, KeyError, TypeError):
            qtype, letters = None, None
        if qtype in QTYPE_TO_NAME and letters is not None:
            return {"q": qtype, "g": letters, "usage": usage}
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"Fact numbers must be between 1 and {n}, and no fact can be in both E and T. Reply again."},
        ]
    return {"q": None, "g": None, "usage": usage}


def apply_rules(text: str, group: str) -> str:
    if group == "history" and TEST_RE.search(text):
        return "tests"
    if group == "history" and VITALS_RE.search(text):
        return "exam"
    if group == "exam" and IMAGING_ECG_RE.search(text) and not VITALS_RE.search(text):
        return "tests"
    return group


def assemble_case(row: dict[str, Any], facts: list[dict[str, Any]], tags: dict[str, Any] | None, args, model: str) -> dict[str, Any]:
    usable = [f for f in facts if not f["cut_off"]]
    out_facts, drop_reason, qtype = [], None, None

    if not facts:
        drop_reason = "no_facts"
    elif len(usable) < args.min_facts:
        drop_reason = "too_few_facts"
    elif tags is None or tags.get("g") is None:
        drop_reason = "tagging_failed"

    letters = iter(tags["g"]) if (drop_reason is None and tags) else None
    for f in facts:
        if f["cut_off"]:
            group, fixed = "dropped_cut_off", False
        elif letters is None:
            group, fixed = "untagged", False
        else:
            tagged = LETTER_TO_GROUP[next(letters)]
            group = apply_rules(f["text"], tagged)
            fixed = group != tagged
        out_facts.append({"id": f["id"], "text": f["text"], "group": group, "rule_fixed": fixed})

    groups = {g: [f["text"] for f in out_facts if f["group"] == g] for g in ("history", "exam", "tests")}
    if drop_reason is None:
        qtype = QTYPE_TO_NAME[tags["q"]]
        answer_norm = normalize(row.get("answer", ""))
        if qtype != "patient":
            drop_reason = "general_knowledge_question"
        elif len(groups["history"]) < args.min_history:
            drop_reason = "too_little_history"
        elif not groups["exam"] and not groups["tests"]:
            drop_reason = "no_exam_or_tests"
        elif len(answer_norm) >= args.min_answer_chars and answer_norm in normalize(" ".join(groups["history"])):
            drop_reason = "answer_in_history"

    return {
        "case_id": row["id"],
        "specialty": (row.get("patient") or {}).get("gpt_specialty"),
        "question": row["question"],
        "options": row["options"],
        "answer_idx": row["answer_idx"],
        "answer": row.get("answer"),
        "question_type": qtype,
        "facts": out_facts,
        **groups,
        "n_history": len(groups["history"]),
        "n_exam": len(groups["exam"]),
        "n_tests": len(groups["tests"]),
        "keep": drop_reason is None,
        "drop_reason": drop_reason,
        "tagger_model": model,
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:8]

    for path in (args.output_jsonl, args.cache_jsonl, args.review_json, args.stats_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict[str, Any]] = {}
    if args.cache_jsonl.exists():
        for entry in load_jsonl(args.cache_jsonl):
            cache[entry["key"]] = entry

    facts_by_case = {row["id"]: clean_facts(row.get("facts") or []) for row in rows}
    tags_by_case: dict[Any, dict[str, Any]] = {}
    jobs = []
    for row in rows:
        usable = [f["text"] for f in facts_by_case[row["id"]] if not f["cut_off"]]
        if len(usable) < args.min_facts:
            continue
        key = f"{args.model}|{prompt_hash}|{row['id']}"
        cached = cache.get(key)
        if cached and cached.get("n_facts") == len(usable) and cached.get("g"):
            tags_by_case[row["id"]] = cached
        else:
            jobs.append((row, usable, key))

    print(f"[01] {len(rows)} cases | {len(tags_by_case)} from cache | {len(jobs)} API calls with {args.model}")

    run_usage = Counter()
    if jobs:
        # max_retries=0: retries are handled in create_with_backoff, not twice
        client = OpenAI(base_url=args.base_url, max_retries=0) if args.base_url else OpenAI(max_retries=0)
        lock = threading.Lock()
        with args.cache_jsonl.open("a", encoding="utf-8") as cache_f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = {
                pool.submit(tag_case, client, args.model, system_prompt, row["question"], usable, args.max_format_retries): (row, usable, key)
                for row, usable, key in jobs
            }
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="tagging"):
                row, usable, key = futures[fut]
                result = fut.result()
                run_usage.update(result["usage"])
                entry = {"key": key, "case_id": row["id"], "n_facts": len(usable), "q": result["q"], "g": result["g"]}
                tags_by_case[row["id"]] = entry
                with lock:
                    cache_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    cache_f.flush()

    cases = [assemble_case(row, facts_by_case[row["id"]], tags_by_case.get(row["id"]), args, args.model) for row in rows]
    cases.sort(key=lambda c: c["case_id"])
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    kept = [c for c in cases if c["keep"]]
    review = random.Random(args.seed).sample(kept, min(args.review_size, len(kept)))
    review = [
        {
            "case_id": c["case_id"],
            "question": c["question"],
            "history": c["history"],
            "exam": c["exam"],
            "tests": c["tests"],
            "rule_fixed": [f["text"] for f in c["facts"] if f["rule_fixed"]],
            "cut_off": [f["text"] for f in c["facts"] if f["group"] == "dropped_cut_off"],
        }
        for c in sorted(review, key=lambda c: c["case_id"])
    ]
    args.review_json.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    n_kept = max(len(kept), 1)
    stats = {
        "n_cases": len(cases),
        "n_kept": len(kept),
        "drop_reasons": dict(Counter(c["drop_reason"] for c in cases if not c["keep"]).most_common()),
        "question_types": dict(Counter(c["question_type"] for c in cases).most_common()),
        "kept_with_exam_and_tests": sum(1 for c in kept if c["n_exam"] and c["n_tests"]),
        "kept_exam_only": sum(1 for c in kept if c["n_exam"] and not c["n_tests"]),
        "kept_tests_only": sum(1 for c in kept if c["n_tests"] and not c["n_exam"]),
        "mean_facts_per_kept_case": {
            g: round(sum(c[f"n_{g}"] for c in kept) / n_kept, 2) for g in ("history", "exam", "tests")
        },
        "facts_cut_off": sum(f["group"] == "dropped_cut_off" for c in cases for f in c["facts"]),
        "facts_rule_fixed": sum(f["rule_fixed"] for c in cases for f in c["facts"]),
        "api": {
            "model": args.model,
            "prompt_hash": prompt_hash,
            "calls_this_run": len(jobs),
            "cache_hits": len(tags_by_case) - len(jobs),
            "prompt_tokens_this_run": run_usage["prompt_tokens"],
            "completion_tokens_this_run": run_usage["completion_tokens"],
        },
    }
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[01] kept {stats['n_kept']} / {stats['n_cases']} cases")
    print(f"[01] drop reasons: {stats['drop_reasons']}")
    print(
        f"[01] exam+tests {stats['kept_with_exam_and_tests']} | exam only {stats['kept_exam_only']} "
        f"| tests only {stats['kept_tests_only']}"
    )
    print(f"[01] tokens this run: {run_usage['prompt_tokens']} in / {run_usage['completion_tokens']} out")
    print(f"[01] wrote {args.output_jsonl}, {args.review_json}, {args.stats_json}")


if __name__ == "__main__":
    main()
