"""Reward for the MedEviAware GRPO runs (answer / request / defer with reasoning, Qwen3-4B-Instruct).

The response must be <reasoning>...</reasoning><answer>a</answer> with nothing after </answer>, where a
is an option letter, REQUEST: EXAM, REQUEST: TESTS or DEFER. Then

    score = +1                          a is the correct letter
            -WRONG_PENALTY              a is another letter
             0                          a = DEFER
            Q(s, REQUEST_g)             a = REQUEST_g and group g is available on request
            INVALID_REQUEST             a = REQUEST_g and g is not available (not obtainable, absent or
                                        already given)
            FORMAT_PENALTY              malformed, or cut off at the length limit

Q(s, REQUEST_g) = -request_cost + V(s with g revealed) comes from the labels (extra_info
score_request_exam / score_request_tests): V is the best score still reachable after the evidence is
revealed, from the reasoning p measured on the scoring step. It is null when g cannot be requested.
"""

import re

WRONG_PENALTY = 1.0
INVALID_REQUEST = -1.0
FORMAT_PENALTY = -1.0

FORMAT_RE = re.compile(r"^\s*<reasoning>(.+?)</reasoning>\s*<answer>(.*?)</answer>\s*$", re.DOTALL)
LETTER_RE = re.compile(r"^\s*([A-Za-z])\s*$")
REQUEST_ALIASES = {
    "REQUESTEXAM": "REQUEST_EXAM",
    "REQUESTEXAMINATION": "REQUEST_EXAM",
    "REQUESTPHYSICALEXAM": "REQUEST_EXAM",
    "REQUESTPHYSICALEXAMINATION": "REQUEST_EXAM",
    "REQUESTTESTS": "REQUEST_TESTS",
    "REQUESTTEST": "REQUEST_TESTS",
    "REQUESTTESTRESULTS": "REQUEST_TESTS",
}
REQUEST_FIELDS = {"REQUEST_EXAM": ("exam_status", "score_request_exam"), "REQUEST_TESTS": ("tests_status", "score_request_tests")}


def parse_action(solution_str, options):
    """The action in a well-formed response: a letter in options, REQUEST_EXAM, REQUEST_TESTS, DEFER; else None."""
    if solution_str.count("<answer>") != 1 or solution_str.count("<reasoning>") != 1:
        return None
    match = FORMAT_RE.match(solution_str)
    if not match:
        return None
    content = match.group(2).strip()
    letter = LETTER_RE.match(content)
    if letter:
        return letter.group(1).upper() if letter.group(1).upper() in options else None
    squashed = re.sub(r"[^A-Z]", "", content.upper())
    if squashed == "DEFER":
        return "DEFER"
    return REQUEST_ALIASES.get(squashed)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra = extra_info or {}
    options = extra.get("options") or "ABCD"
    action = parse_action(solution_str, options)

    flags = {"format": 0.0, "correct": 0.0, "wrong": 0.0, "defer": 0.0, "request": 0.0, "invalid_request": 0.0}
    if action is None:
        score = FORMAT_PENALTY
    else:
        flags["format"] = 1.0
        if action == "DEFER":
            score = 0.0
            flags["defer"] = 1.0
        elif action in REQUEST_FIELDS:
            flags["request"] = 1.0
            status_key, score_key = REQUEST_FIELDS[action]
            q_value = extra.get(score_key)
            if extra.get(status_key) == "obtainable" and q_value is not None:
                score = float(q_value)
            else:
                score = INVALID_REQUEST
                flags["invalid_request"] = 1.0
        elif action == ground_truth.strip().upper():
            score = 1.0
            flags["correct"] = 1.0
        else:
            score = -WRONG_PENALTY
            flags["wrong"] = 1.0
    return {"score": score, **flags}


if __name__ == "__main__":
    ex = {"options": "ABCD", "exam_status": "obtainable", "tests_status": "unavailable",
          "score_request_exam": 0.95, "score_request_tests": None}
    samples = [
        # (name, solution_str, ground_truth, extra_info, expected score)
        ("correct", "<reasoning>Branchial cleft cyst fits.</reasoning><answer>D</answer>", "D", ex, 1.0),
        ("correct_lowercase_spaces", "<reasoning>...</reasoning>\n<answer> d </answer>\n", "D", ex, 1.0),
        ("wrong_letter", "<reasoning>...</reasoning><answer>A</answer>", "D", ex, -1.0),
        ("defer", "<reasoning>Cannot tell.</reasoning><answer>DEFER</answer>", "D", ex, 0.0),
        ("request_obtainable", "<reasoning>Need the exam.</reasoning><answer>REQUEST: EXAM</answer>", "D", ex, 0.95),
        ("request_variant_spelling", "<reasoning>...</reasoning><answer>Request Physical Exam</answer>", "D", ex, 0.95),
        ("request_not_obtainable", "<reasoning>...</reasoning><answer>REQUEST: TESTS</answer>", "D", ex, -1.0),
        ("request_already_given", "<reasoning>...</reasoning><answer>REQUEST: EXAM</answer>", "D",
         {**ex, "exam_status": "given", "score_request_exam": None}, -1.0),
        ("no_reasoning_tag", "<answer>D</answer>", "D", ex, -1.0),
        ("cut_off", "<reasoning>Comparing D and A again, wait", "D", ex, -1.0),
        ("text_after_answer", "<reasoning>...</reasoning><answer>D</answer> because", "D", ex, -1.0),
        ("two_answers", "<reasoning>...</reasoning><answer>A</answer><answer>D</answer>", "D", ex, -1.0),
        ("option_text", "<reasoning>...</reasoning><answer>D. Branchial cleft cyst</answer>", "D", ex, -1.0),
        ("letter_outside_options", "<reasoning>...</reasoning><answer>E</answer>", "D", ex, -1.0),
        ("think_tags", "<think>...</think><answer>D</answer>", "D", ex, -1.0),
    ]
    failed = 0
    for name, solution_str, gt, extra_info, want in samples:
        result = compute_score("medeviaware", solution_str, gt, extra_info)
        ok = abs(result["score"] - want) < 1e-9
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {result}")
    print(f"{len(samples) - failed}/{len(samples)} passed")
    raise SystemExit(1 if failed else 0)
