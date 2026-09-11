"""Rendering comparisons as markdown.

Markdown so results paste straight into the project report, and so a diff
between two runs is readable.
"""

import math
from collections.abc import Sequence

from tournament.results import RoundResult
from tournament.schedule import CANDIDATE_ARM, CONTROL_ARM
from tournament.stats import ArmSummary, PairedDelta, paired_delta, summarise_arm

_ARM_LABELS = {CANDIDATE_ARM: "candidate", CONTROL_ARM: "control"}


def _num(value: float, places: int = 2) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{places}f}"


def _arms_present(results: Sequence[RoundResult]) -> list[str]:
    seen = {result.arm for result in results}
    ordered = [arm for arm in (CANDIDATE_ARM, CONTROL_ARM) if arm in seen]
    return ordered + sorted(seen - set(ordered))


def summary_table(summaries: Sequence[ArmSummary]) -> str:
    """One row per arm, with the score interval and the diagnostics."""
    columns = [
        ("arm", "---"),
        ("agent", "---"),
        ("rounds", "---:"),
        ("round len", "---:"),
        ("score", "---:"),
        ("95% CI", "---"),
        ("win", "---:"),
        ("tie", "---:"),
        ("survive", "---:"),
        ("suicides", "---:"),
        ("coins", "---:"),
        ("kills", "---:"),
        ("max latency", "---:"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in columns) + " |",
        "|" + "|".join(align for _, align in columns) + "|",
    ]
    for summary in summaries:
        cells = [
            _ARM_LABELS.get(summary.arm, summary.arm),
            f"`{summary.code_name}`",
            str(summary.rounds),
            _num(summary.round_steps, places=0),
            _num(summary.score.mean),
            f"[{_num(summary.score.ci.low)}, {_num(summary.score.ci.high)}]",
            _num(summary.win_rate),
            _num(summary.tie_rate),
            _num(summary.survival_rate),
            _num(summary.suicides),
            _num(summary.coins),
            _num(summary.kills),
            f"{_num(summary.latency_max * 1000.0, places=1)} ms",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_section(delta: PairedDelta) -> str:
    """The headline: candidate minus control on shared boards."""
    ci = delta.delta.ci
    verdict = (
        "excludes zero" if delta.is_significant else "includes zero (not resolved)"
    )
    lines = [
        f"**Paired score difference** (`{delta.candidate}` - `{delta.control}`,"
        f" same board and seat): **{_num(delta.delta.mean)}**"
        f" (95% CI [{_num(ci.low)}, {_num(ci.high)}], {verdict})",
        "",
        f"- pairs: {delta.delta.n}",
    ]
    if delta.unpaired_candidate or delta.unpaired_control:
        lines.append(
            f"- dropped as unpaired: {delta.unpaired_candidate} candidate,"
            f" {delta.unpaired_control} control"
        )
    return "\n".join(lines)


def timeout_warnings(summaries: Sequence[ArmSummary], budget: float) -> list[str]:
    """Flag anything that broke, or came close to breaking, the time limit."""
    warnings: list[str] = []
    for summary in summaries:
        if summary.timeouts:
            warnings.append(
                f"`{summary.code_name}` exceeded the think-time budget"
                f" {summary.timeouts} time(s) - those actions became WAIT."
            )
        elif summary.latency_max > budget / 2.0:
            warnings.append(
                f"`{summary.code_name}` peaked at"
                f" {_num(summary.latency_max * 1000.0, 1)} ms against a"
                f" {_num(budget * 1000.0, 0)} ms budget."
            )
    return warnings


def render(results: Sequence[RoundResult], *, budget: float, title: str = "") -> str:
    """Full markdown report for a set of rounds."""
    if not results:
        return "No rounds to report."

    arms = _arms_present(results)
    summaries = [summarise_arm(results, arm) for arm in arms]

    scenarios = sorted({result.scenario for result in results})
    seeds = {result.seed for result in results}

    sections: list[str] = []
    if title:
        sections.append(f"## {title}\n")
    sections.append(
        f"{len(results)} rounds | scenario(s): {', '.join(scenarios)} |"
        f" {len(seeds)} seeds | lineup: `{'`, `'.join(results[0].lineup)}`\n"
    )
    sections.append(summary_table(summaries))

    if CANDIDATE_ARM in arms and CONTROL_ARM in arms:
        sections.append("\n" + delta_section(paired_delta(results)))

    warnings = timeout_warnings(summaries, budget)
    if warnings:
        sections.append("\n**Warnings**\n")
        sections.extend(f"- {warning}" for warning in warnings)

    return "\n".join(sections)
