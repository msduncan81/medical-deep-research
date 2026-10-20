"""Nodes C1-C3 — starting level, the five rating-down domains and rating up, per body of evidence.

Rules implemented in code follow Core GRADE (BMJ 2025) unless noted:
- Starting level: randomized trials high, non-randomized studies low (Core GRADE 4).
- Risk of bias: each study low/high; rate down when most of the weight comes from high-risk studies
  (Core GRADE 4). "Unclear" is not a limitation by itself (GRADE Handbook) but is flagged as limited
  appraisal.
- Inconsistency: point estimates on opposite sides of the threshold, CI overlap, I² as a secondary
  signal (concern rises above ~30%; serious inconsistency is rare below it). No rating with one study.
  Rating down twice is practically never needed (Core GRADE 3).
- Imprecision: the CI against the target threshold (null or MID); if it does not cross, optimal
  information size: continuous 400 per group (800 total), <30% of that consider -2; binary: CI-limit
  ratio >= 3 (2.5 for OR) -> -2, large relative effects (RRR > ~30%) go to an OIS calculation with
  alpha 0.05, beta 0.20, RRR 25% (Core GRADE 2). At most -2.
- Publication bias: at most -1; funnel-based tests only with >= 10 studies (GRADE Handbook).
- Rating up (large effect): RR > 2 or < 0.5 -> +1, > 5 or < 0.2 -> +2, only for non-randomized
  bodies without serious limitations (GRADE Handbook; Core GRADE 4).
Indirectness is a judgement (Core GRADE 5) made by the model, per PICO element.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from .models import CertaintyTarget, OutcomeSpec, RiskOfBias
from .synthesis import RATIO_MEASURES, EvidenceBody

LEVELS = {4: "high", 3: "moderate", 2: "low", 1: "very low"}
CONTINUOUS_OIS_TOTAL = 800


class DomainJudgement(BaseModel):
    domain: str
    rating: int = 0  # 0, -1, -2 (rating down) or +1, +2 (rating up)
    rationale: str = ""
    source: str = "code"  # "code" or "model"
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)


def starting_level(body: EvidenceBody) -> int:
    return 4 if body.design == "rct" else 2


def _weight(item) -> float:  # noqa: ANN001 - StudyOutcomeData
    n = item.participants_in_estimate or ((item.n_intervention or 0) + (item.n_control or 0))
    return float(n) if n else 1.0


def risk_of_bias(body: EvidenceBody) -> DomainJudgement:
    total = sum(_weight(i) for i in body.items) or 1.0
    high = sum(_weight(i) for i in body.items if i.risk_of_bias == RiskOfBias.HIGH) / total
    unclear = sum(_weight(i) for i in body.items if i.risk_of_bias == RiskOfBias.UNCLEAR) / total
    abstract_only = all(i.source_text != "fulltext" for i in body.items)
    rating = -1 if high > 0.5 else 0
    rationale = (
        f"{high:.0%} of the weight comes from high risk-of-bias studies"
        + (" -> serious limitations" if rating else "; most evidence is not at high risk")
    )
    if unclear > 0.5:
        rationale += f"; {unclear:.0%} could not be judged (limited appraisal)"
    if abstract_only:
        rationale += "; judged from abstracts only"
    return DomainJudgement(domain="risk_of_bias", rating=rating, rationale=rationale,
                           metrics={"share_high": round(high, 3), "share_unclear": round(unclear, 3),
                                    "abstract_only": str(abstract_only)})


def _null_value(measure: str) -> float:
    return 1.0 if measure in RATIO_MEASURES else 0.0


def inconsistency(body: EvidenceBody) -> DomainJudgement:
    pooled = body.pooled
    if pooled is None or pooled.k < 2:
        return DomainJudgement(domain="inconsistency", rationale="single estimate: inconsistency not rated")
    null = _null_value(pooled.measure)
    sides = {"benefit": 0, "harm": 0}
    for _ref, est, _lo, _hi in pooled.study_estimates:
        if est != null:
            sides["benefit" if est < null else "harm"] += 1
    rows = pooled.study_estimates
    non_overlap = any(a[3] < b[2] or b[3] < a[2] for i, a in enumerate(rows) for b in rows[i + 1:])
    both_sides = sides["benefit"] > 0 and sides["harm"] > 0
    i2 = pooled.i2 or 0.0
    serious = (both_sides and (non_overlap or i2 >= 50)) or (i2 >= 75 and non_overlap)
    rationale = (
        f"I² {i2:.0f}%; estimates on {'both sides' if both_sides else 'one side'} of the null; "
        f"{'some' if non_overlap else 'no'} non-overlapping CIs"
    )
    return DomainJudgement(domain="inconsistency", rating=-1 if serious else 0, rationale=rationale,
                           metrics={"i2": i2, "k": pooled.k, "both_sides": str(both_sides),
                                    "non_overlapping_cis": str(non_overlap)})


def _ois_binary(control_rate: float, rrr: float = 0.25, increase: bool = False) -> int:
    """Total optimal information size for two proportions (alpha 0.05 two-sided, power 0.80)."""
    p0 = min(max(control_rate, 1e-4), 0.9999)
    p1 = p0 * (1 + rrr) if increase else p0 * (1 - rrr)
    p1 = min(max(p1, 1e-4), 0.9999)
    z = 1.959964 + 0.841621
    n_group = z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p0 - p1) ** 2
    return int(math.ceil(2 * n_group))


def imprecision(body: EvidenceBody, outcome: OutcomeSpec) -> DomainJudgement:
    pooled = body.pooled
    if pooled is None:
        return DomainJudgement(domain="imprecision", rating=-2, metrics={"quantitative": "no"},
                               rationale="no quantitative estimate could be combined (narrative evidence only)")
    measure, est, lo, hi = pooled.measure, pooled.estimate, pooled.ci_low, pooled.ci_high
    n = body.participants if body.participants_complete else None
    metrics: dict[str, float | int | str | None] = {"estimate": round(est, 4), "ci_low": round(lo, 4),
                                                    "ci_high": round(hi, 4), "participants": body.participants,
                                                    "participants_complete": str(body.participants_complete)}
    use_mid = outcome.target == CertaintyTarget.MID and outcome.mid_value and measure not in RATIO_MEASURES
    if use_mid:
        mid = abs(float(outcome.mid_value or 0))
        benefit_low = outcome.desirable_direction != "increase"
        b_thr, h_thr = (-mid, mid) if benefit_low else (mid, -mid)
        crosses_benefit = min(lo, hi) < b_thr < max(lo, hi)
        crosses_harm = min(lo, hi) < h_thr < max(lo, hi)
        metrics["mid"] = mid
        if crosses_benefit and crosses_harm:
            return DomainJudgement(domain="imprecision", rating=-2, metrics=metrics,
                                   rationale=f"CI crosses both the important-benefit and important-harm thresholds (MID {mid})")
        if crosses_benefit or crosses_harm:
            return DomainJudgement(domain="imprecision", rating=-1, metrics=metrics,
                                   rationale=f"CI crosses the MID ({mid})")
    else:
        null = _null_value(measure)
        if lo < null < hi:
            rating = -1
            reason = "CI includes the null"
            if measure in RATIO_MEASURES and hi / lo >= (2.5 if measure == "OR" else 3):
                rating, reason = -2, f"CI includes the null and is very wide (upper/lower = {hi / lo:.1f})"
            elif measure not in RATIO_MEASURES and n is not None and n < 0.3 * CONTINUOUS_OIS_TOTAL:
                rating, reason = -2, f"CI includes the null with only {n} participants"
            return DomainJudgement(domain="imprecision", rating=rating, rationale=reason, metrics=metrics)
    # CI does not cross the threshold: optimal information size
    if measure in RATIO_MEASURES:
        ratio = hi / lo
        metrics["ci_ratio"] = round(ratio, 2)
        if ratio >= (2.5 if measure == "OR" else 3):
            return DomainJudgement(domain="imprecision", rating=-2, metrics=metrics,
                                   rationale=f"very wide CI (upper/lower = {ratio:.1f})")
        rrr = abs(1 - est)
        if rrr > 0.3 and body.control_event_rate:
            ois = _ois_binary(body.control_event_rate, increase=outcome.desirable_direction == "increase")
            metrics["ois"] = ois
            if n is None:
                return DomainJudgement(domain="imprecision", metrics=metrics,
                                       rationale=f"CI excludes the threshold; OIS ({ois}) could not be verified "
                                                 "(participants missing for some studies)")
            if n < ois:
                return DomainJudgement(domain="imprecision", rating=-1, metrics=metrics,
                                       rationale=f"large relative effect from {n} participants, below the OIS ({ois})")
        return DomainJudgement(domain="imprecision", rationale="CI excludes the threshold; information size adequate",
                               metrics=metrics)
    if n is None:
        return DomainJudgement(domain="imprecision", metrics=metrics,
                               rationale="CI excludes the threshold; OIS could not be verified "
                                         "(participants missing for some studies)")
    if n < 0.3 * CONTINUOUS_OIS_TOTAL:
        return DomainJudgement(domain="imprecision", rating=-2, metrics=metrics,
                               rationale=f"{n} participants, under 30% of the OIS ({CONTINUOUS_OIS_TOTAL})")
    if n < CONTINUOUS_OIS_TOTAL:
        return DomainJudgement(domain="imprecision", rating=-1, metrics=metrics,
                               rationale=f"{n} participants, below the OIS ({CONTINUOUS_OIS_TOTAL})")
    return DomainJudgement(domain="imprecision", rationale=f"CI excludes the threshold; {n} participants >= OIS",
                           metrics=metrics)


def egger_p_value(rows: list[tuple[int, float, float, float]], log: bool) -> float | None:
    """Egger regression intercept test (standard normal deviate on precision); None if < 10 studies."""
    if len(rows) < 10:
        return None
    xs, ys = [], []
    for _ref, est, lo, hi in rows:
        y = math.log(est) if log else est
        se = ((math.log(hi) - math.log(lo)) if log else (hi - lo)) / 3.919928
        if se <= 0:
            continue
        xs.append(1 / se)
        ys.append(y / se)
    n = len(xs)
    if n < 10:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    se_int = math.sqrt(resid * (1 / n + mx**2 / sxx))
    if se_int == 0:
        return None
    t = intercept / se_int
    return round(math.erfc(abs(t) / math.sqrt(2)), 4)  # normal approximation


def publication_bias(body: EvidenceBody) -> DomainJudgement:
    pooled = body.pooled
    rows = pooled.study_estimates if pooled else []
    p = egger_p_value(rows, log=bool(pooled and pooled.measure in RATIO_MEASURES)) if pooled else None
    if p is None:
        return DomainJudgement(domain="publication_bias", metrics={"k": len(rows)},
                               rationale=f"not testable ({len(rows)} estimates; funnel tests need >= 10)")
    rating = -1 if p < 0.10 else 0
    return DomainJudgement(domain="publication_bias", rating=rating, metrics={"egger_p": p, "k": len(rows)},
                           rationale=f"Egger test P = {p}" + (" (small-study effects suspected)" if rating else ""))


def rating_up(body: EvidenceBody, downgrades: list[DomainJudgement]) -> DomainJudgement:
    pooled = body.pooled
    if body.design != "nrs" or pooled is None or pooled.measure not in RATIO_MEASURES:
        return DomainJudgement(domain="rating_up", rationale="not applicable (randomized body or no ratio estimate)")
    if any(d.rating < 0 for d in downgrades):
        return DomainJudgement(domain="rating_up", rationale="not considered: the body has serious limitations")
    est = pooled.estimate
    level = 2 if (est >= 5 or est <= 0.2) else 1 if (est >= 2 or est <= 0.5) else 0
    return DomainJudgement(domain="rating_up", rating=level, metrics={"estimate": round(est, 3)},
                           rationale="very large effect" if level == 2 else "large effect" if level else "effect not large")


def sum_certainty(start: int, judgements: list[DomainJudgement]) -> int:
    downs = sum(min(0, j.rating) for j in judgements)
    ups = sum(max(0, j.rating) for j in judgements)
    return max(1, min(4, start + ups + downs))
