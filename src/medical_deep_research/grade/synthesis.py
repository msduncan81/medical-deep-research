"""Node C0 — bodies of evidence per outcome x design, with a numerical summary.

Certainty is rated per outcome across the body of evidence (GRADE 1). Randomized and non-randomized
evidence form separate bodies with different starting levels (GRADE Handbook; Core GRADE 4; GRADE
guidance 24). Effects are first oriented as intervention vs control using the "favours" field, then
combined with a DerSimonian-Laird random-effects model from the reported confidence intervals. A pooled
estimate from a chosen systematic review enters as one unit; its trials were not extracted again.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from .models import OutcomeSpec, StudyDesign, StudyExtraction, StudyOutcomeData

RATIO_MEASURES = {"RR", "OR", "HR"}
DIFFERENCE_MEASURES = {"MD", "SMD", "RD"}
Z95 = 1.959964


class PooledEstimate(BaseModel):
    measure: str
    estimate: float
    ci_low: float
    ci_high: float
    k: int
    tau2: float = 0.0
    i2: float | None = None
    q_p_value: float | None = None
    study_estimates: list[tuple[int, float, float, float]] = Field(
        default_factory=list, description="(study_ref, estimate, ci_low, ci_high) after orientation"
    )


class EvidenceBody(BaseModel):
    outcome_id: str
    design: str  # "rct" | "nrs"
    items: list[StudyOutcomeData] = Field(default_factory=list)
    pooled: PooledEstimate | None = None
    participants: int | None = None
    control_event_rate: float | None = None
    not_pooled: list[str] = Field(default_factory=list)
    derived: list[str] = Field(default_factory=list)
    participants_complete: bool = False
    from_review: bool = False

    @property
    def refs(self) -> list[int]:
        return sorted({i.study_ref for i in self.items})


def _benefit_is_lower(outcome: OutcomeSpec) -> bool:
    return outcome.desirable_direction.strip().lower() != "increase"


def rr_from_counts(item: StudyOutcomeData) -> tuple[float, float, float] | None:
    """Risk ratio (intervention/control) with 95% CI from 2x2 counts; 0.5 continuity correction for zeros."""
    a, n1, c, n2 = item.events_intervention, item.n_intervention, item.events_control, item.n_control
    if None in (a, n1, c, n2) or not n1 or not n2 or a > n1 or c > n2:  # type: ignore[operator]
        return None
    if a in (0, n1) or c in (0, n2):
        a, c, n1, n2 = a + 0.5, c + 0.5, n1 + 1, n2 + 1  # type: ignore[operator]
    rr = (a / n1) / (c / n2)  # type: ignore[operator]
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)  # type: ignore[operator]
    return rr, math.exp(math.log(rr) - Z95 * se), math.exp(math.log(rr) + Z95 * se)


def orient(item: StudyOutcomeData, outcome: OutcomeSpec) -> tuple[float, float, float] | None:
    """Return (estimate, ci_low, ci_high) as intervention vs control, or None if not usable."""
    if item.effect is None or item.ci_low is None or item.ci_high is None or not item.effect_measure:
        return None
    est, lo, hi = item.effect, min(item.ci_low, item.ci_high), max(item.ci_low, item.ci_high)
    lower_is_better = _benefit_is_lower(outcome)
    if item.effect_measure in RATIO_MEASURES:
        if est <= 0 or lo <= 0:
            return None
        favours_intervention_side = (est < 1) == lower_is_better
        if item.favours == "intervention" and not favours_intervention_side and est != 1:
            est, lo, hi = 1 / est, 1 / hi, 1 / lo
        elif item.favours == "control" and favours_intervention_side and est != 1:
            est, lo, hi = 1 / est, 1 / hi, 1 / lo
    else:
        favours_intervention_side = (est < 0) == lower_is_better
        if item.favours == "intervention" and not favours_intervention_side and est != 0:
            est, lo, hi = -est, -hi, -lo
        elif item.favours == "control" and favours_intervention_side and est != 0:
            est, lo, hi = -est, -hi, -lo
    return est, lo, hi


def _chi2_sf(q: float, df: int) -> float:
    """Survival function of chi-square (Wilson-Hilferty approximation; adequate for heterogeneity P)."""
    if df <= 0:
        return 1.0
    z = ((q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def pool(measure: str, estimates: list[tuple[int, float, float, float]]) -> PooledEstimate | None:
    """DerSimonian-Laird random effects on the log scale for ratios, natural scale for differences."""
    if not estimates:
        return None
    log = measure in RATIO_MEASURES
    ys, ws = [], []
    for _ref, est, lo, hi in estimates:
        y = math.log(est) if log else est
        se = ((math.log(hi) - math.log(lo)) if log else (hi - lo)) / (2 * Z95)
        if se <= 0:
            continue
        ys.append(y)
        ws.append(1 / se**2)
    if not ys:
        return None
    k = len(ys)
    fixed = sum(w * y for w, y in zip(ws, ys)) / sum(ws)
    q = sum(w * (y - fixed) ** 2 for w, y in zip(ws, ys))
    c = sum(ws) - sum(w**2 for w in ws) / sum(ws)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    wr = [1 / (1 / w + tau2) for w in ws]
    mu = sum(w * y for w, y in zip(wr, ys)) / sum(wr)
    se_mu = math.sqrt(1 / sum(wr))
    lo, hi = mu - Z95 * se_mu, mu + Z95 * se_mu
    i2 = max(0.0, (q - (k - 1)) / q * 100) if k > 1 and q > 0 else (0.0 if k > 1 else None)
    back = math.exp if log else (lambda v: v)
    return PooledEstimate(
        measure=measure, estimate=back(mu), ci_low=back(lo), ci_high=back(hi), k=k, tau2=tau2,
        i2=round(i2, 1) if i2 is not None else None,
        q_p_value=round(_chi2_sf(q, k - 1), 4) if k > 1 else None,
        study_estimates=estimates,
    )


def _design_group(item: StudyOutcomeData) -> str | None:
    if item.design in (StudyDesign.RCT, StudyDesign.SYSTEMATIC_REVIEW):
        return "rct"  # reviews are taken as reviews of randomized trials unless stated otherwise
    if item.design == StudyDesign.NON_RANDOMIZED:
        return "nrs"
    return None


def _participants(item: StudyOutcomeData) -> int | None:
    if item.participants_in_estimate:
        return item.participants_in_estimate
    if item.n_intervention and item.n_control:
        return item.n_intervention + item.n_control
    return None


def build_bodies(outcomes: list[OutcomeSpec], extractions: list[StudyExtraction]) -> list[EvidenceBody]:
    by_outcome: dict[tuple[str, str], list[StudyOutcomeData]] = defaultdict(list)
    for extraction in extractions:
        for item in extraction.items:
            group = _design_group(item)
            if item.reported and group:
                by_outcome[(item.outcome_id, group)].append(item)
    bodies: list[EvidenceBody] = []
    for outcome in outcomes:
        for group in ("rct", "nrs"):
            items = by_outcome.get((outcome.id, group), [])
            if not items:
                continue
            body = EvidenceBody(outcome_id=outcome.id, design=group, items=items)
            oriented: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
            pooled_refs: set[int] = set()
            for item in items:
                result = orient(item, outcome)
                measure = item.effect_measure or "?"
                if result is None and (counts := rr_from_counts(item)) is not None:
                    result, measure = counts, "RR"
                    body.derived.append(f"[{item.study_ref}] RR calculated from events per group")
                if result is None:
                    body.not_pooled.append(f"[{item.study_ref}] no usable effect with CI")
                    continue
                oriented[measure].append((item.study_ref, *result))
            if oriented:
                measure = max(oriented, key=lambda m: (len(oriented[m]), m in DIFFERENCE_MEASURES))
                body.pooled = pool(measure, oriented[measure])
                pooled_refs = {row[0] for row in oriented[measure]}
                for other, rows in oriented.items():
                    if other != measure:
                        body.not_pooled.extend(f"[{r[0]}] reported {other}, not combined with {measure}" for r in rows)
            # Information size counts only the estimates that were combined; unknown n -> unknown total.
            in_pool = [i for i in items if i.study_ref in pooled_refs]
            sizes = [_participants(i) for i in in_pool]
            body.participants = sum(x for x in sizes if x) or None
            body.participants_complete = bool(in_pool) and all(sizes)
            ev_c = sum(i.events_control or 0 for i in items if i.events_control is not None and i.n_control)
            n_c = sum(i.n_control or 0 for i in items if i.events_control is not None and i.n_control)
            body.control_event_rate = round(ev_c / n_c, 4) if n_c else None
            body.from_review = any(i.design == StudyDesign.SYSTEMATIC_REVIEW for i in items)
            bodies.append(body)
    return bodies
