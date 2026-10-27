"""Shared text formatting for MedEviAware cases.

Every step that shows a case to a model (scoring, SFT, RL, evaluation) renders it through
this module, so the wording of facts, questions and options is identical everywhere.
"""

from typing import Any


GROUP_TITLES = {"history": "Patient history", "exam": "Physical examination", "tests": "Test results"}
FACT_SET_GROUPS = {
    "H": ("history",),
    "H+E": ("history", "exam"),
    "H+T": ("history", "tests"),
    "H+E+T": ("history", "exam", "tests"),
}
OPTION_ORDERS = ("orig", "rev")


def evidence_type(case: dict[str, Any]) -> str:
    if case["n_exam"] and case["n_tests"]:
        return "exam+tests"
    return "exam_only" if case["n_exam"] else "tests_only"


def available_fact_sets(case: dict[str, Any]) -> list[str]:
    """Fact sets that differ from each other for this case (no H+T when there are no tests, etc.)."""
    sets = ["H"]
    if case["n_exam"]:
        sets.append("H+E")
    if case["n_tests"]:
        sets.append("H+T")
    if case["n_exam"] and case["n_tests"]:
        sets.append("H+E+T")
    return sets


def render_patient_info(case: dict[str, Any], groups: tuple[str, ...]) -> str:
    blocks = []
    for group in groups:
        if case[group]:
            lines = "\n".join(f"- {fact}" for fact in case[group])
            blocks.append(f"{GROUP_TITLES[group]}:\n{lines}")
    return "\n\n".join(blocks)


def reorder_options(options: dict[str, str], order: str) -> list[tuple[str, str, str]]:
    """Return (shown_letter, original_letter, text) in the requested order."""
    original = sorted(options)
    shown = original if order == "orig" else original[::-1]
    return [(new, old, options[old]) for new, old in zip(original, shown)]


def render_question(question: str, ordered_options: list[tuple[str, str, str]]) -> str:
    lines = [f"Question: {question}"] + [f"{new}. {text}" for new, _, text in ordered_options]
    return "\n".join(lines)


def build_score_messages(
    case: dict[str, Any], fact_set: str, order: str, system_prompt: str
) -> tuple[list[dict[str, str]], dict[str, str], str]:
    """Messages asking for the answer letter, the shown->original letter map, and the shown gold letter."""
    ordered = reorder_options(case["options"], order)
    letter_map = {new: old for new, old, _ in ordered}
    gold_shown = next(new for new, old, _ in ordered if old == case["answer_idx"])
    user = "\n\n".join(
        [
            render_patient_info(case, FACT_SET_GROUPS[fact_set]),
            render_question(case["question"], ordered),
            "Answer with one letter.",
        ]
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    return messages, letter_map, gold_shown
