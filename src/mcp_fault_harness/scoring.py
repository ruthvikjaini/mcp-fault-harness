"""Scoring: from step outcomes to the reported metrics.

Two scores are reported for every run, on a 0–10 scale like the
task-completion score in Iannillo et al.:

* ``score`` — credit for every correct sub-answer, including correct
  guesses and stale-fallback answers (at 0.6). This is the analogue
  of the paper's completion score, which counted ungrounded answers.
* ``grounded_score`` — credit only for correct answers backed by a
  live, validated tool result. Guesses, fallbacks, and corrupt
  payloads count 0 here, so the gap between the two scores measures
  how much of the headline score is grounded.

Write steps additionally require exactly one applied effect: a
duplicate or a missing effect scores 0 regardless of the value the
agent holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp_fault_harness.agents import (
    PROVENANCE_ABSTAINED,
    PROVENANCE_CORRUPT,
    PROVENANCE_FALLBACK,
    PROVENANCE_GROUNDED,
    PROVENANCE_GUESSED,
    RunRecord,
)


@dataclass(frozen=True)
class RunScores:
    score: float
    grounded_score: float
    complete: bool
    guessed_rate: float
    corrupt_rate: float
    abstain_rate: float
    fallback_rate: float
    duplicate_write: bool


def score_run(record: RunRecord) -> RunScores:
    outcomes = record.outcomes
    n = len(outcomes)
    credits = [o.credit for o in outcomes]
    grounded = [
        1.0 if (o.provenance == PROVENANCE_GROUNDED and o.correct) else 0.0
        for o in outcomes
    ]
    return RunScores(
        score=10.0 * sum(credits) / n if n else 0.0,
        grounded_score=10.0 * sum(grounded) / n if n else 0.0,
        complete=record.complete,
        guessed_rate=sum(1 for o in outcomes if o.provenance == PROVENANCE_GUESSED) / n if n else 0.0,
        corrupt_rate=sum(1 for o in outcomes if o.provenance == PROVENANCE_CORRUPT) / n if n else 0.0,
        abstain_rate=sum(1 for o in outcomes if o.provenance == PROVENANCE_ABSTAINED) / n if n else 0.0,
        fallback_rate=sum(1 for o in outcomes if o.provenance == PROVENANCE_FALLBACK) / n if n else 0.0,
        duplicate_write=any(o.duplicate_effect for o in outcomes),
    )
