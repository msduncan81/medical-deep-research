"""Nodes C2 (indirectness), C4-C5 (certainty, modes A and B) and C6 (integration) per outcome.

Mode A: certainty = starting level + rating up - rating down (computed by code from the domain
judgements); the model may move it one level, with a written reason (GRADE: the overall judgement is
holistic, not arithmetic — GRADE 11; GRADE Handbook).
Mode B: the model judges every domain and the final certainty from the same metrics; code only checks
coherence. Both are kept so they can be compared (decision of 20/10/2026).
Integration of randomized and non-randomized bodies follows GRADE guidance 24: with high-certainty RCT
evidence the non-randomized body is not used; otherwise it may complement or replace it when its
certainty is higher.
The certainty a systematic review reported is never shown to these nodes; it is compared afterwards.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from ..models import RunRequest
from .domains import (
    LEVELS,
    DomainJudgement,
    imprecision,
    inconsistency,
    publication_bias,
    rating_up,
    risk_of_bias,
    starting_level,
    sum_certainty,
)
from .models import OutcomeSpec
from .outcomes import StructuredCall
from .synthesis import EvidenceBody

# ---------------------------------------------------------------------------
# Structured outputs for the model
# ---------------------------------------------------------------------------


class ElementJudgement(BaseModel):
    concern: str = Field(default="none", description="'none', 'serious' or 'very serious'")
    rationale: str = ""


class IndirectnessOutput(BaseModel):
    population: ElementJudgement = Field(default_factory=ElementJudgement)
    intervention: ElementJudgement = Field(default_factory=ElementJudgement)
    comparator: ElementJudgement = Field(default_factory=ElementJudgement)
    outcome: ElementJudgement = Field(default_factory=ElementJudgement)
    rating: int = Field(default=0, description="0, -1 or -2")
    rationale: str = ""


class AdjustmentOutput(BaseModel):
    adjustment: int = Field(default=0, description="-1, 0 or +1 relative to the computed certainty")
    rationale: str = ""


class HolisticDomain(BaseModel):
    domain: str
    rating: int = 0
    rationale: str = ""


class HolisticOutput(BaseModel):
    domains: list[HolisticDomain] = Field(default_factory=list)
    certainty: str = Field(default="low", description="'high', 'moderate', 'low' or 'very low'")
    rationale: str = ""


class BodyAssessment(BaseModel):
    outcome_id: str
    design: str
    start: int
    refs: list[int]
    participants: int | None = None
    estimate: dict[str, Any] | None = None
    domains: list[DomainJudgement] = Field(default_factory=list)
    certainty_computed: int = 1
    certainty_a: int = 1
    adjustment_a: AdjustmentOutput = Field(default_factory=AdjustmentOutput)
    certainty_b: int = 1
    holistic_b: HolisticOutput = Field(default_factory=HolisticOutput)
    coherence_flags: list[str] = Field(default_factory=list)
    not_pooled: list[str] = Field(default_factory=list)


class OutcomeProfile(BaseModel):
    outcome: OutcomeSpec
    bodies: list[BodyAssessment] = Field(default_factory=list)
    certainty_a: int | None = None
    certainty_b: int | None = None
    integration_note: str = ""
    review_certainties: list[dict[str, Any]] = Field(default_factory=list)
    comparison_notes: list[str] = Field(default_factory=list)

    @property
    def label_a(self) -> str:
        return LEVELS.get(self.certainty_a or 0, "no evidence")

    @property
    def label_b(self) -> str:
        return LEVELS.get(self.certainty_b or 0, "no evidence")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INDIRECTNESS_INSTRUCTIONS = """\
You judge INDIRECTNESS for one body of evidence in a GRADE assessment (Core GRADE 5).
Compare the target question with what the studies actually did, element by element (population,
intervention, comparator, outcome). For each substantial mismatch, ask how likely it is that the size of
the effect differs substantially because of it; the more likely, the greater the concern.
- Differences in baseline risk or comorbidity seldom change relative effects: usually no concern.
- Often a concern: surrogate outcomes when the target is patient-important, poorly described or different
  interventions (e.g. a broader multicomponent programme), suboptimal comparators, outdated technology.
- Indirect harms evidence from other conditions: at most one level.
- VARIATION BETWEEN STUDIES IS NOT INDIRECTNESS. If some studies match the target and others differ (e.g.
  self-monitoring alone vs self-monitoring plus self-titration), that is an inconsistency/subgroup question,
  rated in the inconsistency domain. Rate indirectness only when most of the weight of the evidence
  differs from the target question in a way likely to change the effect.
- Rating down twice for indirectness is rare (e.g. evidence only from a different population or a surrogate
  far from the target outcome).
Return concern per element ("none", "serious", "very serious") and an overall rating: 0, -1 or -2.
Write rationales in the report language, briefly."""

ADJUSTMENT_INSTRUCTIONS = """\
You review a GRADE certainty computed by summing domain ratings for one body of evidence. GRADE's overall
judgement is holistic: you may move the computed certainty by at most one level (-1, 0 or +1) when the sum
misrepresents the evidence. Check explicitly for double counting: the same problem rated in two domains,
e.g. differences in interventions across studies rated both as inconsistency and as indirectness, or a
wide CI caused by heterogeneity rated both as inconsistency and imprecision. Several borderline concerns
together may justify -1. Default is 0. Explain briefly in the report language."""

HOLISTIC_INSTRUCTIONS = """\
You rate the GRADE certainty of evidence for ONE outcome and ONE body of evidence (Core GRADE), using the
metrics and study details given. Start high for randomized trials, low for non-randomized studies.
Judge the five rating-down domains (risk_of_bias, inconsistency, indirectness, imprecision,
publication_bias: 0, -1 or -2; publication bias at most -1) and, for non-randomized bodies only, rating
up (large effect, dose-response, residual confounding: 0, +1 or +2). Avoid counting the same concern in
two domains. Then give the final certainty: "high", "moderate", "low" or "very low", with a short
rationale in the report language."""


def _target_text(request: RunRequest, outcome: OutcomeSpec) -> str:
    question = request.original_query or request.query
    target = (
        f"MID {outcome.mid_value} {outcome.mid_unit or ''} ({outcome.mid_source or 'no source'})"
        if outcome.target == "mid" else "null (any effect)"
    )
    return (
        f"Question: {question}\nOutcome {outcome.id}: {outcome.name} ({outcome.kind}; better if "
        f"{outcome.desirable_direction}); certainty target: {target}"
    )


def _body_text(body: EvidenceBody) -> str:
    lines = [f"Body of evidence: {'randomized' if body.design == 'rct' else 'non-randomized'}; "
             f"{len(body.refs)} sources; participants {body.participants or 'unknown'}"]
    if body.pooled:
        p = body.pooled
        lines.append(f"Summary estimate ({p.measure}, random effects, k={p.k}): {p.estimate:.3g} "
                     f"[{p.ci_low:.3g}; {p.ci_high:.3g}], I² {p.i2 if p.i2 is not None else 'n/a'}%")
    for item in body.items:
        effect = (f"{item.effect_measure} {item.effect} [{item.ci_low}; {item.ci_high}], favours {item.favours}"
                  if item.effect is not None else "no comparative effect")
        lines.append(
            f"- [{item.study_ref}] {item.design}; P: {item.population or '?'}; I: {item.intervention or '?'}; "
            f"C: {item.comparator or '?'}; O: {item.outcome_definition or '?'}; follow-up {item.follow_up or '?'}; "
            f"{effect}; risk of bias {item.risk_of_bias} ({item.risk_of_bias_rationale or 'no rationale'}); "
            f"from {item.source_text}"
        )
    return "\n".join(lines)


def _domains_text(domains: list[DomainJudgement]) -> str:
    return "\n".join(f"- {d.domain}: {d.rating:+d} ({d.source}) — {d.rationale}" for d in domains)


_LABEL_TO_LEVEL = {"high": 4, "moderate": 3, "low": 2, "very low": 1, "alta": 4, "moderada": 3,
                   "baixa": 2, "muito baixa": 1}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


async def _safe_call(structured_call: StructuredCall, task: str, instructions: str, prompt: str,
                     model: type[BaseModel], flags: list[str]) -> Any:
    try:
        return await structured_call(task, instructions, prompt, model)
    except Exception as exc:  # noqa: BLE001 - keep the code-computed result when the model fails
        flags.append(f"{task} failed ({type(exc).__name__}); default used")
        return model()


# ---------------------------------------------------------------------------
# Per-body assessment
# ---------------------------------------------------------------------------


async def assess_body(
    structured_call: StructuredCall,
    request: RunRequest,
    outcome: OutcomeSpec,
    body: EvidenceBody,
) -> BodyAssessment:
    start = starting_level(body)
    target = _target_text(request, outcome)
    details = _body_text(body)

    flags: list[str] = []
    code_domains = [risk_of_bias(body), inconsistency(body), imprecision(body, outcome), publication_bias(body)]
    indirect = await _safe_call(
        structured_call, "grade_indirectness", INDIRECTNESS_INSTRUCTIONS, f"{target}\n\n{details}",
        IndirectnessOutput, flags,
    )
    concerns = [indirect.population, indirect.intervention, indirect.comparator, indirect.outcome]
    serious_elements = sum(c.concern in ("serious", "very serious") for c in concerns)
    very_serious_po = any(c.concern == "very serious" for c in (indirect.population, indirect.outcome))
    worst = 2 if (very_serious_po or serious_elements >= 2) else 1 if serious_elements else 0
    # The overall rating cannot exceed the element concerns; -2 needs a very serious population/outcome
    # mismatch or concerns in at least two elements (rating down twice is rare — Core GRADE 5).
    indirect_rating = _clamp(indirect.rating, -worst, 0)
    indirectness = DomainJudgement(
        domain="indirectness", rating=indirect_rating, source="model",
        rationale=indirect.rationale or "; ".join(f"{n}: {c.concern}" for n, c in
                                                   zip(("P", "I", "C", "O"), concerns)),
        metrics={n: c.concern for n, c in zip(("population", "intervention", "comparator", "outcome"), concerns)},
    )
    downgrades = [*code_domains[:2], indirectness, *code_domains[2:]]
    up = rating_up(body, downgrades)
    domains = [*downgrades, up]
    computed = sum_certainty(start, domains)

    adjustment = await _safe_call(
        structured_call, "grade_certainty_review", ADJUSTMENT_INSTRUCTIONS,
        f"{target}\n\n{details}\n\nStarting level: {LEVELS[start]}\nDomain ratings:\n{_domains_text(domains)}\n"
        f"Computed certainty: {LEVELS[computed]}",
        AdjustmentOutput, flags,
    )
    adjustment.adjustment = _clamp(adjustment.adjustment, -1, 1)
    certainty_a = _clamp(computed + adjustment.adjustment, 1, 4)

    metrics_only = "\n".join(f"- {d.domain} metrics: {d.metrics}" for d in domains if d.metrics)
    holistic = await _safe_call(
        structured_call, "grade_certainty_holistic", HOLISTIC_INSTRUCTIONS,
        f"{target}\n\n{details}\n\nComputed metrics (no ratings):\n{metrics_only}", HolisticOutput, flags,
    )
    certainty_b = _LABEL_TO_LEVEL.get(holistic.certainty.strip().lower(), computed)
    if any(f.startswith("grade_certainty_holistic failed") for f in flags):
        certainty_b = computed
    ups_b = sum(max(0, d.rating) for d in holistic.domains)
    if body.design == "rct" and ups_b:
        flags.append("mode B rated up a randomized body")
    downs_b = sum(min(0, d.rating) for d in holistic.domains)
    implied_b = _clamp(start + ups_b + downs_b, 1, 4)
    if abs(implied_b - certainty_b) > 1:
        flags.append(f"mode B final ({LEVELS[certainty_b]}) differs by >1 level from its own domains ({LEVELS[implied_b]})")
    if certainty_b > start + ups_b:
        certainty_b = start + ups_b
        flags.append("mode B certainty capped at the starting level plus rating up")

    pooled = body.pooled
    return BodyAssessment(
        outcome_id=outcome.id, design=body.design, start=start, refs=body.refs, participants=body.participants,
        estimate=pooled.model_dump(exclude={"study_estimates"}) if pooled else None,
        domains=domains, certainty_computed=computed, certainty_a=certainty_a, adjustment_a=adjustment,
        certainty_b=certainty_b, holistic_b=holistic, coherence_flags=flags, not_pooled=body.not_pooled,
    )


def integrate(bodies: list[BodyAssessment], attr: str) -> tuple[int | None, str]:
    """Outcome certainty from the randomized and non-randomized bodies (GRADE guidance 24)."""
    rct = next((b for b in bodies if b.design == "rct"), None)
    nrs = next((b for b in bodies if b.design == "nrs"), None)
    if rct and not nrs:
        return getattr(rct, attr), "randomized evidence only"
    if nrs and not rct:
        return getattr(nrs, attr), "non-randomized evidence only"
    if not rct or not nrs:
        return None, "no evidence"
    r, n = getattr(rct, attr), getattr(nrs, attr)
    if r == 4:
        return r, "high-certainty randomized evidence; non-randomized studies not used (GRADE 24)"
    if n > r:
        return n, "non-randomized evidence replaces lower-certainty randomized evidence (GRADE 24; judgement to confirm)"
    return r, "randomized evidence; non-randomized studies complementary (GRADE 24)"


def compare_with_reviews(profile: OutcomeProfile, bodies: list[EvidenceBody]) -> None:
    for body in bodies:
        for item in body.items:
            if item.review_grade:
                reported = _LABEL_TO_LEVEL.get(item.review_grade.strip().lower())
                profile.review_certainties.append({"review_ref": item.study_ref, "reported": item.review_grade})
                if reported and profile.certainty_a:
                    verdict = "agrees" if reported == profile.certainty_a else "differs"
                    profile.comparison_notes.append(
                        f"Review [{item.study_ref}] reported {item.review_grade}; this assessment {verdict} "
                        f"(mode A {profile.label_a}, mode B {profile.label_b})."
                    )


async def assess_outcomes(
    structured_call: StructuredCall,
    request: RunRequest,
    outcomes: list[OutcomeSpec],
    bodies: list[EvidenceBody],
    *,
    concurrency: int = 2,
) -> list[OutcomeProfile]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(outcome: OutcomeSpec, body: EvidenceBody) -> BodyAssessment:
        async with semaphore:
            return await assess_body(structured_call, request, outcome, body)

    tasks = {(b.outcome_id, b.design): one(next(o for o in outcomes if o.id == b.outcome_id), b) for b in bodies}
    results = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    profiles: list[OutcomeProfile] = []
    for outcome in outcomes:
        assessed = [results[key] for key in results if key[0] == outcome.id]
        profile = OutcomeProfile(outcome=outcome, bodies=assessed)
        profile.certainty_a, note = integrate(assessed, "certainty_a")
        profile.certainty_b, _ = integrate(assessed, "certainty_b")
        profile.integration_note = note
        compare_with_reviews(profile, [b for b in bodies if b.outcome_id == outcome.id])
        profiles.append(profile)
    return profiles
