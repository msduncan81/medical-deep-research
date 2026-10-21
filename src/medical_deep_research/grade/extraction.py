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
from .models import ComparisonSpec, OutcomeSpec, ReviewMap, StudyDesign, StudyExtraction, StudyOutcomeData
from .outcomes import StructuredCall
from .reviews import is_network_meta_analysis, is_systematic_review

MAX_TEXT_CHARS = 14000

EXTRACTION_INSTRUCTIONS = """\
You extract outcome data from ONE study for a GRADE assessment. Return JSON matching the schema, with
one item per outcome id listed in the prompt (in the same order).
Rules:
- reported=false when the study does not report that outcome; then leave the numeric fields empty.
- Copy numbers exactly as written in the text (effect, CI, n, events). Never compute, convert or guess.
  Put the shortest verbatim passage that contains the numbers in "quote". No quote -> no numbers.
- Only between-group (comparative) effects count. effect_measure must be one of: RR, OR, HR, MD
  (mean difference), SMD, RD (risk difference). A proportion or mean in one group alone is NOT an effect:
  leave effect empty (you may still give n and events per group).
- favours: the group the effect favours according to the text ("intervention", "control", "neither" when
  no difference, "unclear"). Copy the effect with the sign the text uses; do not flip it yourself.
- design: "rct", "non_randomized", "systematic_review" or "other".
- For a systematic review: the pooled estimate for the outcome, studies_in_estimate,
  participants_in_estimate, heterogeneity_i2 if reported, review_grade if the review reports a GRADE
  certainty for it, and is_network_meta_analysis=true for a network meta-analysis.
- population/intervention/comparator/outcome_definition/follow_up: what the study actually did, briefly.
- risk_of_bias is judged PER OUTCOME (Core GRADE 4): "high" only when a limitation is likely to bias THIS
  outcome's estimate. Consider randomisation and allocation concealment, missing outcome data (amount and
  imbalance) and selective reporting. Lack of blinding matters only for outcomes that unblinded people
  can influence (subjective, patient-reported or assessor-judged outcomes). Open-label trials of
  self-management are usually "low" for objective outcomes such as automated blood-pressure readings,
  laboratory values, hospitalisation or death, if randomisation and follow-up are adequate.
  "unclear" when the text does not allow a judgement (typical for abstracts). Explain in
  risk_of_bias_rationale, naming the outcome.
- For a systematic review: summarise the risk of bias of ITS INCLUDED TRIALS FOR THIS OUTCOME as the review
  reports it: "high" only if most of the weight comes from trials at high risk for this outcome, or the
  review itself rated the outcome down for risk of bias; a "high" rating in a single domain (e.g.
  performance bias in unblindable trials) is not enough.
- source_text: "fulltext" if the passage came from the full text section, else "abstract".
- subgroup_levels (study level): for each subgroup id listed, the level this study belongs to, copied
  exactly from the listed levels, or "unclear".
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


def _comparison_text(comparison: ComparisonSpec | None) -> str:
    if comparison is None or not (comparison.intervention or comparison.subgroups):
        return ""
    lines = [f"Intervention in scope: {comparison.intervention or '?'}; comparator: {comparison.comparator or '?'}"]
    for h in comparison.subgroups:
        lines.append(f"Subgroup {h.id}: {h.variable} -> levels {h.levels}")
    return "\n".join(lines) + "\n\n"


def extraction_prompt(
    study: ScoredStudy, outcomes: list[OutcomeSpec], fulltext: str | None, comparison: ComparisonSpec | None = None
) -> str:
    outcome_lines = "\n".join(
        f"- {o.id}: {o.name} ({o.kind}{'; unit ' + o.mid_unit if o.mid_unit else ''})" for o in outcomes
    )
    body = f"ABSTRACT:\n{study.abstract or '(no abstract)'}"
    if fulltext:
        body += f"\n\nFULL TEXT (selected sections):\n{_relevant_fulltext(fulltext, outcomes, MAX_TEXT_CHARS)}"
    return (
        f"{_comparison_text(comparison)}Outcome ids:\n{outcome_lines}\n\n"
        f"STUDY [{study.reference_number}]: {study.title}\n"
        f"Journal/year: {study.journal or '?'} {study.publication_year or ''}\n"
        f"Publication types: {', '.join(study.publication_types or []) or 'unknown'}\n\n{body}"
    )


_MEASURE_ALIASES = {
    "RR": "RR", "RISK RATIO": "RR", "RELATIVE RISK": "RR", "OR": "OR", "ODDS RATIO": "OR",
    "HR": "HR", "HAZARD RATIO": "HR", "MD": "MD", "MEAN DIFFERENCE": "MD", "WMD": "MD",
    "DIFFERENCE": "MD", "SMD": "SMD", "STANDARDISED MEAN DIFFERENCE": "SMD",
    "STANDARDIZED MEAN DIFFERENCE": "SMD", "RD": "RD", "RISK DIFFERENCE": "RD", "ARD": "RD",
    "ABSOLUTE RISK DIFFERENCE": "RD",
}
COMPARATIVE_MEASURES = {"RR", "OR", "HR", "MD", "SMD", "RD"}


def _sanitize(item: StudyOutcomeData, study: ScoredStudy, outcome_ids: set[str]) -> StudyOutcomeData | None:
    if item.outcome_id not in outcome_ids:
        return None
    item.study_ref = int(study.reference_number or 0)
    if is_systematic_review(study):
        item.design = StudyDesign.SYSTEMATIC_REVIEW
    if is_network_meta_analysis(study):
        item.is_network_meta_analysis = True
    measure = _MEASURE_ALIASES.get((item.effect_measure or "").strip().upper())
    item.effect_measure = measure
    if measure is None:
        # Not a between-group effect (e.g. a proportion in one arm): keep counts, drop the "effect".
        item.effect = item.ci_low = item.ci_high = None
    if item.favours not in {"intervention", "control", "neither", "unclear"}:
        item.favours = "unclear"
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
    network_reviews = {r.review_ref for r in review_map.reviews if r.network}
    skipped_reviews = {r.review_ref for r in review_map.reviews if not r.chosen}
    selected, notes = [], []
    for study in studies:
        ref = int(study.reference_number or 0)
        if ref in network_reviews:
            notes.append(f"[{ref}] not extracted: network meta-analysis (outside Core GRADE).")
            continue
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
    comparison: ComparisonSpec | None = None,
) -> StudyExtraction:
    raw = await structured_call(
        "grade_extraction", EXTRACTION_INSTRUCTIONS, extraction_prompt(study, outcomes, fulltext, comparison),
        StudyExtraction,
    )
    ids = {o.id for o in outcomes}
    items = [clean for item in raw.items if (clean := _sanitize(item, study, ids)) is not None]
    levels: dict[str, str] = {}
    for hypothesis in (comparison.subgroups if comparison else []):
        value = (raw.subgroup_levels or {}).get(hypothesis.id, "unclear")
        levels[hypothesis.id] = value if value in hypothesis.levels else "unclear"
    return StudyExtraction(study_ref=int(study.reference_number or 0), subgroup_levels=levels, items=items)


async def extract_all(
    structured_call: StructuredCall,
    studies: list[ScoredStudy],
    outcomes: list[OutcomeSpec],
    fulltext_for: Callable[[ScoredStudy], str | None],
    *,
    comparison: ComparisonSpec | None = None,
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
                return await extract_study(structured_call, study, outcomes, fulltext_for(study), comparison)
            except Exception as exc:  # noqa: BLE001 - one bad study must not stop the graph
                notes.append(f"[{study.reference_number}] extraction failed: {type(exc).__name__}")
                return None

    results = await asyncio.gather(*(one(s) for s in studies))
    return [r for r in results if r is not None], notes
