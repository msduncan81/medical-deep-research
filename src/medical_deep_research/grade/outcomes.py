"""Node A2 — propose outcomes and certainty targets, then let the user confirm or edit them.

GRADE rates certainty per outcome; outcomes are chosen and ranked by importance before appraisal
(GRADE 1; GRADE Handbook: 1-9 scale, 7-9 critical). A Summary of Findings table holds up to seven
outcomes (Cochrane Handbook ch. 14). Core GRADE 2 sets the certainty target per outcome: the null
("is there any effect?") or the minimal important difference ("is the effect important?").
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from ..languages import language_display_name
from ..models import ApprovalRequest, ApprovalStatus, RunRequest
from ..research.models import ScoredStudy
from .models import CertaintyTarget, OutcomeImportance, OutcomeProposal, OutcomeSpec

TModel = TypeVar("TModel", bound=BaseModel)
StructuredCall = Callable[[str, str, str, type[TModel]], Awaitable[TModel]]

MAX_OUTCOMES = 7
OUTCOME_CONFIRMATION_TYPE = "outcome_confirmation"
DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = 1800

OUTCOME_INSTRUCTIONS = """\
You define the outcomes for a GRADE assessment (Core GRADE) of one comparison.
Return JSON matching the schema. Rules:
- Propose at most 7 outcomes that matter to patients, including the outcome in the question and the
  critical benefits and harms a clinician would weigh. Prefer patient-important outcomes; when a
  surrogate is used (e.g. blood pressure), say so in the rationale.
- importance_score on the GRADE 1-9 scale; importance = "critical" for 7-9, "important" for 4-6.
- kind: "binary", "continuous" or "time_to_event". desirable_direction: "decrease" or "increase".
- target: "mid" ONLY when an established minimal important difference exists for that outcome; then give
  mid_value (absolute, in the outcome's unit), mid_unit and mid_source (citation or guideline). Otherwise
  target "null" and leave the MID fields empty. Never invent a MID.
- Write outcome names and rationales in the report language. Use ids o1, o2, ...
"""


def _study_digest(studies: list[ScoredStudy], limit: int = 25) -> str:
    lines = []
    for study in studies[:limit]:
        abstract = (study.abstract or "").replace("\n", " ").strip()
        lines.append(
            f"[{study.reference_number}] {study.title} ({study.publication_year or 'n.d.'}; "
            f"{study.evidence_level or 'design unknown'})\n    {abstract[:500]}"
        )
    return "\n".join(lines)


def outcome_prompt(request: RunRequest, studies: list[ScoredStudy]) -> str:
    question = request.original_query or request.query
    english = f"\nQuestion in English: {request.query}" if request.original_query else ""
    return (
        f"Question ({request.query_type.upper()}): {question}{english}\n"
        f"Report language: {language_display_name(request.language)}\n\n"
        f"Included studies (titles and abstracts):\n{_study_digest(studies)}"
    )


def normalize_outcomes(proposal: OutcomeProposal) -> list[OutcomeSpec]:
    """Enforce the GRADE rules the model may have bent: at most seven, coherent importance, no
    MID target without a value and a source, unique ids."""
    outcomes: list[OutcomeSpec] = []
    seen: set[str] = set()
    for index, outcome in enumerate(proposal.outcomes[:MAX_OUTCOMES], start=1):
        data = outcome.model_copy(deep=True)
        if not data.id or data.id in seen:
            data.id = f"o{index}"
        seen.add(data.id)
        data.importance = (
            OutcomeImportance.CRITICAL if data.importance_score >= 7 else OutcomeImportance.IMPORTANT
        )
        if data.target == CertaintyTarget.MID and (data.mid_value is None or not (data.mid_source or "").strip()):
            data.target = CertaintyTarget.NULL
            data.mid_value = None
        outcomes.append(data)
    critical_first = sorted(outcomes, key=lambda o: (o.importance != OutcomeImportance.CRITICAL, -o.importance_score))
    return critical_first


async def propose_outcomes(
    structured_call: StructuredCall,
    request: RunRequest,
    studies: list[ScoredStudy],
) -> list[OutcomeSpec]:
    proposal = await structured_call(
        "grade_outcomes", OUTCOME_INSTRUCTIONS, outcome_prompt(request, studies), OutcomeProposal
    )
    return normalize_outcomes(proposal)


# ---------------------------------------------------------------------------
# User confirmation checkpoint (same mechanism as the PDF upload checkpoint)
# ---------------------------------------------------------------------------

def _session(request: RunRequest):
    if not request.database_path:
        return None
    from sqlmodel import Session, create_engine

    return Session(create_engine(f"sqlite:///{request.database_path}"))


def create_outcome_confirmation(request: RunRequest, outcomes: list[OutcomeSpec]) -> str | None:
    session = _session(request)
    if session is None:
        return None
    details = {
        "type": OUTCOME_CONFIRMATION_TYPE,
        "proposed": [o.model_dump(mode="json") for o in outcomes],
        "confirmed": None,
    }
    with session:
        approval = ApprovalRequest(
            run_id=request.run_id,
            summary=f"Confirm {len(outcomes)} outcomes for the GRADE assessment",
            details_json=json.dumps(details),
            status=ApprovalStatus.PENDING.value,
        )
        session.add(approval)
        session.commit()
        return approval.id


def read_outcome_confirmation(request: RunRequest, approval_id: str) -> tuple[str, dict[str, Any]]:
    session = _session(request)
    if session is None:
        return ApprovalStatus.REJECTED.value, {}
    with session:
        approval = session.get(ApprovalRequest, approval_id)
        if approval is None:
            return ApprovalStatus.REJECTED.value, {}
        try:
            details = json.loads(approval.details_json or "{}")
        except json.JSONDecodeError:
            details = {}
        return approval.status, details if isinstance(details, dict) else {}


def _confirmation_timeout() -> float:
    raw = os.getenv("MDR_OUTCOME_CONFIRMATION_TIMEOUT_SECONDS", "")
    return float(raw) if raw.replace(".", "", 1).isdigit() and float(raw) > 0 else DEFAULT_CONFIRMATION_TIMEOUT_SECONDS


async def confirm_outcomes(
    request: RunRequest,
    proposed: list[OutcomeSpec],
    *,
    poll_seconds: float = 1.0,
) -> tuple[list[OutcomeSpec], bool, str]:
    """Wait for the user to confirm/edit the outcomes. Returns (outcomes, confirmed_by_user, note).

    Without a database (headless runs) or after the timeout, the proposal is used and marked
    as not confirmed, so the report can say so.
    """
    approval_id = create_outcome_confirmation(request, proposed)
    if approval_id is None:
        return proposed, False, "Outcomes were not confirmed by the user (no interactive session)."
    deadline = asyncio.get_running_loop().time() + _confirmation_timeout()
    while asyncio.get_running_loop().time() < deadline:
        status, details = read_outcome_confirmation(request, approval_id)
        if status == ApprovalStatus.APPROVED.value:
            confirmed = details.get("confirmed")
            if isinstance(confirmed, list) and confirmed:
                outcomes = normalize_outcomes(
                    OutcomeProposal(outcomes=[OutcomeSpec.model_validate(item) for item in confirmed])
                )
                return outcomes, True, "Outcomes confirmed by the user."
            return proposed, True, "Proposed outcomes accepted by the user."
        if status == ApprovalStatus.REJECTED.value:
            return proposed, False, "The user skipped outcome confirmation; proposed outcomes were used."
        await asyncio.sleep(poll_seconds)
    return proposed, False, "Outcome confirmation timed out; proposed outcomes were used."
