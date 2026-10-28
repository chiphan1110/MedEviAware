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
OPTION_ORDERS = ("rot0", "rot1", "rot2", "rot3")   # rotation k shows original option i+k at position i
STATUS_TEXT = {"given": "GIVEN", "obtainable": "AVAILABLE ON REQUEST", "unavailable": "NOT OBTAINABLE", "none": "NOT OBTAINABLE"}
STATUS_GROUPS = (("exam", "Physical examination"), ("tests", "Test results"))


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
    """Return (shown_letter, original_letter, text) for rotation order "rotK".

    Across rot0..rot3 every option appears once at every position, so averaging the four
    orders cancels any preference the model has for a letter or position.
    """
    original = sorted(options)
    k = int(order.removeprefix("rot")) % len(original)
    shown = original[k:] + original[:k]
    return [(new, old, options[old]) for new, old in zip(original, shown)]


def build_score_prompt(
    case: dict[str, Any], fact_set: str, order: str, instruction: str
) -> tuple[str, dict[str, str], str]:
    """Scoring prompt (facts of one fact set, question, options in one order, reasoning instruction),
    the shown->original letter map, and the gold letter as shown."""
    ordered = reorder_options(case["options"], order)
    letter_map = {new: old for new, old, _ in ordered}
    gold_shown = next(new for new, old, _ in ordered if old == case["answer_idx"])
    stem = f"{render_patient_info(case, FACT_SET_GROUPS[fact_set])}\n\nQuestion: {case['question']}"
    prompt = render_policy_prompt(stem, {new: text for new, _, text in ordered}, instruction)
    return prompt, letter_map, gold_shown


def render_status(status: dict[str, str]) -> str:
    """Evidence status lines; status maps "exam"/"tests" to given/obtainable/unavailable/none."""
    lines = [f"- {title}: {STATUS_TEXT[status[group]]}" for group, title in STATUS_GROUPS]
    return "Evidence status:\n" + "\n".join(lines)


def render_shown_info(case: dict[str, Any], status: dict[str, str]) -> str:
    """Patient information the policy sees at the start: history plus every group whose status is given."""
    shown_groups = ("history",) + tuple(g for g, _ in STATUS_GROUPS if status[g] == "given")
    return render_patient_info(case, shown_groups)


def render_reveal(case: dict[str, Any], group: str, status: dict[str, str]) -> str:
    """Environment reply to a request: the group's facts if it was obtainable, else a refusal.

    `status` is the status after the request (the requested group is "given" if it was revealed).
    """
    if status[group] == "given":
        body = render_patient_info(case, (group,))
    else:
        verb = "are" if group == "tests" else "is"
        body = f"The {GROUP_TITLES[group].lower()} {verb} not available."
    return f"{body}\n\n{render_status(status)}"


FOLLOWUP_INSTRUCTION = "Reason inside <think> </think> again, then give exactly one action inside <answer> </answer>."
LAST_TURN_INSTRUCTION = "This is your last turn: answer with an option letter or DEFER."


def render_policy_prompt(question_stem: str, options: dict[str, str], instruction: str) -> str:
    """First user turn for an episode item: stem, options (one per line), then the instruction prompt.

    Use the same function when building training parquets, so training and evaluation prompts match.
    """
    option_lines = "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))
    return f"{question_stem}\n\nOptions:\n{option_lines}\n\n{instruction}"


def render_followup(reveal_text: str, last_turn: bool) -> str:
    """User turn after a request: what was revealed (or refused), then a short reminder of the format."""
    parts = [reveal_text, FOLLOWUP_INSTRUCTION]
    if last_turn:
        parts.append(LAST_TURN_INSTRUCTION)
    return "\n\n".join(parts)
