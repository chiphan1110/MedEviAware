#!/usr/bin/env python3
"""Pick the best action for every (case, availability version) from the measured p values.

Input:
- --tagged-jsonl: cases_tagged.jsonl from 01_tag_facts.py
- --split-csv: case split from 02_split_cases.py
- --scores-jsonl: scores.jsonl from eval/score_p_vllm.py (p per fact set: H, H+E, H+T, H+E+T)

Behavior:
- skip cases whose exam or test facts refer to an image or figure the dataset does not contain
- give each evidence group a status: given / obtainable / unavailable ("none" if the case has no
  facts in that group), so exam+tests cases get 9 versions and exam-only / tests-only cases get 3
- score each action by looking ahead over what could still be revealed:
    ANSWER      = p * 1 + (1 - p) * (-wrong_penalty)      p = chance of being right with the shown facts
    DEFER       = 0
    REQUEST_<g> = -(request_cost + request_margin) + best score after g is revealed   (only if g is obtainable)
  the margin makes a request win only when it beats answering or deferring by at least that much,
  so a one-sample gain in p (one of 8 reasoning samples, i.e. noise) does not count as useful evidence;
  evaluation still charges only request_cost
- label = best-scoring action; ties go to the cheaper action (DEFER over ANSWER at exactly 0, and
  ANSWER/DEFER over REQUEST); between two requests, the one whose evidence raises p more
- record the full best path (e.g. REQUEST_EXAM -> ANSWER) and the score of every action
- flag the held-out versions (--heldout-versions) so training steps can leave them out
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUSES = ("given", "obtainable", "unavailable")
GROUPS = (("exam", "E"), ("tests", "T"))
MISSING_IMAGE_RE = re.compile(
    r"\b(is|are) shown\b|\bshown (below|here)\b|\bseen below\b|\b(image|photograph|figure|exhibit)s?\b|\bpattern (in|on) lead\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label every (case, availability version) with its best action.")
    parser.add_argument("--tagged-jsonl", type=Path, required=True, help="Tagged cases from step 01.")
    parser.add_argument("--split-csv", type=Path, required=True, help="Split CSV from step 02.")
    parser.add_argument("--scores-jsonl", type=Path, required=True, help="Per-case p values from the scoring step.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="One label row per (case, version).")
    parser.add_argument("--stats-json", type=Path, required=True, help="Label counts.")
    parser.add_argument("--skipped-json", type=Path, required=True, help="Cases skipped for missing images.")
    parser.add_argument("--wrong-penalty", type=float, default=1.0, help="Cost of a wrong answer (correct = +1).")
    parser.add_argument("--request-cost", type=float, default=0.05, help="Cost of each request.")
    parser.add_argument(
        "--request-margin",
        type=float,
        default=0.2,
        help="Extra cost per request when labelling, so a request must beat answering or deferring by this much.",
    )
    parser.add_argument(
        "--heldout-versions",
        type=str,
        default="unavailable:obtainable,obtainable:unavailable",
        help="Comma-separated exam_status:tests_status pairs used only for testing.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fact_set_key(revealed: frozenset[str]) -> str:
    return "H" + "".join(f"+{g}" for g in ("E", "T") if g in revealed)


class Planner:
    """Best action and best path for one case, given which groups are revealed and which are obtainable."""

    def __init__(self, p: dict[str, float], wrong_penalty: float, request_cost: float):
        self.p = p
        self.wrong_penalty = wrong_penalty
        self.request_cost = request_cost
        self.memo: dict[tuple[frozenset[str], frozenset[str]], tuple[float, list[str], dict[str, float]]] = {}

    def plan(self, revealed: frozenset[str], obtainable: frozenset[str]) -> tuple[float, list[str], dict[str, float]]:
        key = (revealed, obtainable)
        if key in self.memo:
            return self.memo[key]
        p_now = self.p[fact_set_key(revealed)]
        scores = {"ANSWER": p_now - (1 - p_now) * self.wrong_penalty, "DEFER": 0.0}
        paths = {"ANSWER": ["ANSWER"], "DEFER": ["DEFER"]}
        gains = {}
        for g in sorted(obtainable):
            value, path, _ = self.plan(revealed | {g}, obtainable - {g})
            action = f"REQUEST_{'EXAM' if g == 'E' else 'TESTS'}"
            scores[action] = -self.request_cost + value
            paths[action] = [action] + path
            gains[action] = self.p[fact_set_key(revealed | {g})] - p_now

        # Tie-breaks: DEFER beats ANSWER at exactly 0; ANSWER/DEFER beat requests; bigger gain wins between requests.
        best = "ANSWER" if scores["ANSWER"] > scores["DEFER"] else "DEFER"
        for action in sorted(gains, key=lambda a: -gains[a]):
            if scores[action] > scores[best] + 1e-9:
                best = action
        result = (scores[best], paths[best], {a: round(s, 4) for a, s in scores.items()})
        self.memo[key] = result
        return result


def case_versions(case: dict[str, Any]) -> list[tuple[str, str]]:
    exam = STATUSES if case["n_exam"] else ("none",)
    tests = STATUSES if case["n_tests"] else ("none",)
    return [(e, t) for e in exam for t in tests]


def main() -> None:
    args = parse_args()
    heldout = {tuple(v.split(":")) for v in args.heldout_versions.split(",") if v.strip()}

    with args.split_csv.open(newline="", encoding="utf-8") as f:
        split_rows = {int(r["case_id"]): r for r in csv.DictReader(f)}
    cases = {c["case_id"]: c for c in load_jsonl(args.tagged_jsonl) if c["case_id"] in split_rows}
    scores = {s["case_id"]: s for s in load_jsonl(args.scores_jsonl)}

    skipped = []
    for cid, case in cases.items():
        hits = [t for g in ("exam", "tests") for t in case[g] if MISSING_IMAGE_RE.search(t)]
        if hits:
            skipped.append({"case_id": cid, "reason": "refers_to_missing_image", "facts": hits})
        elif cid not in scores:
            skipped.append({"case_id": cid, "reason": "not_scored", "facts": []})
    skip_ids = {s["case_id"] for s in skipped}

    rows = []
    for cid in sorted(set(cases) - skip_ids):
        case, meta, p = cases[cid], split_rows[cid], scores[cid]["p"]
        planner = Planner(p, args.wrong_penalty, args.request_cost + args.request_margin)
        for exam_status, tests_status in case_versions(case):
            status = dict(zip(("E", "T"), (exam_status, tests_status)))
            revealed = frozenset(g for g, s in status.items() if s == "given")
            obtainable = frozenset(g for g, s in status.items() if s == "obtainable")
            value, path, action_scores = planner.plan(revealed, obtainable)
            ranked = sorted(action_scores.values(), reverse=True)
            rows.append(
                {
                    "case_id": cid,
                    "split": meta["split"],
                    "evidence": meta["evidence"],
                    "exam_status": exam_status,
                    "tests_status": tests_status,
                    "heldout_version": (exam_status, tests_status) in heldout,
                    "label": path[0],
                    "best_path": path,
                    "best_value": round(value, 4),
                    "margin": round(ranked[0] - ranked[1], 4),
                    "action_scores": action_scores,
                    "p_now": p[fact_set_key(revealed)],
                    "p": p,
                    "gold": case["answer_idx"],
                }
            )

    for path in (args.output_jsonl, args.stats_json, args.skipped_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.skipped_json.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    def counts(subset: list[dict[str, Any]]) -> dict[str, Any]:
        c = Counter(r["label"] for r in subset)
        n = max(len(subset), 1)
        return {"n": len(subset), **{k: {"count": v, "share": round(v / n, 4)} for k, v in c.most_common()}}

    by_version = defaultdict(list)
    for r in rows:
        by_version[f"exam={r['exam_status']}|tests={r['tests_status']}"].append(r)
    stats = {
        "settings": {
            "wrong_penalty": args.wrong_penalty,
            "request_cost": args.request_cost,
            "request_margin": args.request_margin,
            "heldout_versions": sorted(":".join(v) for v in heldout),
        },
        "n_cases": len({r["case_id"] for r in rows}),
        "n_skipped": dict(Counter(s["reason"] for s in skipped)),
        "overall": counts(rows),
        "by_split": {
            s: {
                "seen_versions": counts([r for r in rows if r["split"] == s and not r["heldout_version"]]),
                "heldout_versions": counts([r for r in rows if r["split"] == s and r["heldout_version"]]),
            }
            for s in ("train", "val", "test")
        },
        "by_version": {k: counts(v) for k, v in sorted(by_version.items())},
        "best_paths": dict(Counter(" > ".join(r["best_path"]) for r in rows).most_common()),
        "mean_best_value": round(sum(r["best_value"] for r in rows) / max(len(rows), 1), 4),
    }
    args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[04] {stats['n_cases']} cases -> {len(rows)} labelled versions in {args.output_jsonl}")
    print(f"[04] skipped: {stats['n_skipped']} (list in {args.skipped_json})")
    print("[04] labels: " + ", ".join(f"{k} {v['share']:.0%}" for k, v in stats["overall"].items() if k != "n"))
    for s in ("train", "val", "test"):
        seen, ho = stats["by_split"][s]["seen_versions"], stats["by_split"][s]["heldout_versions"]
        print(f"[04] {s:5s} seen versions {seen['n']:5d} | held-out versions {ho['n']:4d}")
    print("[04] best paths: " + ", ".join(f"{k} {v}" for k, v in stats["best_paths"].items()))


if __name__ == "__main__":
    main()
