"""Node B6 — extract, per study x outcome, the data the GRADE domains will need.

One model call per study covers all confirmed outcomes. Numbers must come verbatim from the text (a
supporting quote is required); nothing is computed by the model. Risk of bias follows Core GRADE 4:
each study is classified low or high (unclear when the text does not allow a judgement, e.g. abstract
only). Primary studies already included in a chosen systematic review are not extracted again, so the
same trial is not counted twice (Cochrane Handbook ch. V).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from ..research.models import ScoredStudy
from .models import OutcomeSpec, ReviewMap, StudyDesign, StudyExtraction, StudyOutcomeData
from .outcomes import StructuredCall
from .reviews import is_systematic_review

MAX_TEXT_CHARS = 14000

EXTRACTION_INSTRUCTIONS = """\
You extract outcome data from ONE study for a GRADE assessment. Return JSON matching the schema, with
one item per outcome id listed in the prompt (in the same order).
Rules:
- reported=false when the study does not report that outcome; then leave the numeric fields empty.
- Copy numbers exactly as written in the text (effect, CI, n, events). Never compute, convert or guess.
  Put the shortest verbatim passage that contains the numbers in "quote". No quote -> no numbers.
- design: "rct", "non_randomized", "systematic_review" or "other".
- For a systematic review: the pooled estimate for the outcome, studies_in_estimate,
  participants_in_estimate, and review_grade if the review reports a GRADE certainty for it.
- population/intervention/comparator/outcome_definition/follow_up: what the study actually did, briefly.
- risk_of_bias (primary studies, per Core GRADE): "low" or "high" based on randomisation and allocation
  concealment, blinding where it matters for this outcome, missing outcome data and selective reporting;
  "unclear" when the text does not allow a judgement (typical for abstracts). Explain in
  risk_of_bias_rationale. For a systematic review, judge the risk of bias of its included trials as the
  review reports it.
- source_text: "fulltext" if the passage came from the full text section, else "abstract".
"""


def _relevant_fulltext(fulltext: str, outcomes: list[OutcomeSpec], limit: int) -> str:
    """Keep the paragraphs most likely to hold outcome numbers (keywords + digits), within a budget."""
    words = {w for o in outcomes for w in re.findall(r"[a-zA-ZÀ-ÿ]{5,}", o.name.lower())}
    words |= {"result", "outcome", "primary", "secondary", "confidence", "difference", "ratio", "95%"}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z][A-Za-z ]{2,40}\n)", fulltext) if p.strip()]

    def score(p: str) -> int:
        low = p.lower()
        return sum(low.count(w) for w in words) + len(re.findall(r"\d+\.\d+", p))

    ranked = sorted(paragraphs, key=score, reverse=True)
    chosen, total = [], 0
    for paragraph in ranked:
        if total + len(paragraph) > limit:
            continue
        chosen.append(paragraph)
        total += len(paragraph)
    order = {p: i for i, p in enumerate(paragraphs)}
    return "\n\n".join(sorted(chosen, key=lambda p: order[p]))


def extraction_prompt(study: ScoredStudy, outcomes: list[OutcomeSpec], fulltext: str | None) -> str:
    outcome_lines = "\n".join(
        f"- {o.id}: {o.name} ({o.kind}{'; unit ' + o.mid_unit if o.mid_unit else ''})" for o in outcomes
    )
    body = f"ABSTRACT:\n{study.abstract or '(no abstract)'}"
    if fulltext:
        body += f"\n\nFULL TEXT (selected sections):\n{_relevant_fulltext(fulltext, outcomes, MAX_TEXT_CHARS)}"
    return (
        f"Outcome ids:\n{outcome_lines}\n\n"
        f"STUDY [{study.reference_number}]: {study.title}\n"
        f"Journal/year: {study.journal or '?'} {study.publication_year or ''}\n"
        f"Publication types: {', '.join(study.publication_types or []) or 'unknown'}\n\n{body}"
    )


def _sanitize(item: StudyOutcomeData, study: ScoredStudy, outcome_ids: set[str]) -> StudyOutcomeData | None:
    if item.outcome_id not in outcome_ids:
        return None
    item.study_ref = int(study.reference_number or 0)
    if is_systematic_review(study):
        item.design = StudyDesign.SYSTEMATIC_REVIEW
    has_numbers = any(v is not None for v in (item.effect, item.events_intervention, item.n_intervention))
    if has_numbers and not item.quote.strip():
        # Numbers without a supporting passage are not trusted.
        for field in ("effect", "ci_low", "ci_high", "events_intervention", "events_control",
                      "n_intervention", "n_control"):
            setattr(item, field, None)
    if (
        item.effect is not None and item.ci_low is not None and item.ci_high is not None
        and not (min(item.ci_low, item.ci_high) <= item.effect <= max(item.ci_low, item.ci_high))
    ):
        item.ci_low = item.ci_high = None
    return item


def studies_to_extract(studies: list[ScoredStudy], review_map: ReviewMap) -> tuple[list[ScoredStudy], list[str]]:
    chosen_reviews = {r.review_ref for r in review_map.reviews if r.chosen}
    skipped_reviews = {r.review_ref for r in review_map.reviews if not r.chosen}
    selected, notes = [], []
    for study in studies:
        ref = int(study.reference_number or 0)
        if ref in skipped_reviews:
            notes.append(f"[{ref}] not extracted: overlaps with a chosen review.")
            continue
        if ref in review_map.covered_by_chosen and ref not in chosen_reviews:
            notes.append(f"[{ref}] not extracted: already included in review [{review_map.covered_by_chosen[ref]}].")
            continue
        selected.append(study)
    return selected, notes


async def extract_study(
    structured_call: StructuredCall,
    study: ScoredStudy,
    outcomes: list[OutcomeSpec],
    fulltext: str | None,
) -> StudyExtraction:
    raw = await structured_call(
        "grade_extraction", EXTRACTION_INSTRUCTIONS, extraction_prompt(study, outcomes, fulltext), StudyExtraction
    )
    ids = {o.id for o in outcomes}
    items = [clean for item in raw.items if (clean := _sanitize(item, study, ids)) is not None]
    return StudyExtraction(study_ref=int(study.reference_number or 0), items=items)


async def extract_all(
    structured_call: StructuredCall,
    studies: list[ScoredStudy],
    outcomes: list[OutcomeSpec],
    fulltext_for: Callable[[ScoredStudy], str | None],
    *,
    concurrency: int = 2,
    on_progress: Callable[[ScoredStudy], Awaitable[None]] | None = None,
) -> tuple[list[StudyExtraction], list[str]]:
    semaphore = asyncio.Semaphore(concurrency)
    notes: list[str] = []

    async def one(study: ScoredStudy) -> StudyExtraction | None:
        async with semaphore:
            if on_progress:
                await on_progress(study)
            try:
                return await extract_study(structured_call, study, outcomes, fulltext_for(study))
            except Exception as exc:  # noqa: BLE001 - one bad study must not stop the graph
                notes.append(f"[{study.reference_number}] extraction failed: {type(exc).__name__}")
                return None

    results = await asyncio.gather(*(one(s) for s in studies))
    return [r for r in results if r is not None], notes
