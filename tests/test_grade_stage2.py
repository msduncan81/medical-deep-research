from __future__ import annotations

import unittest
from typing import Any

from medical_deep_research.grade.assessment import (
    AdjustmentOutput,
    BodyAssessment,
    HolisticDomain,
    HolisticOutput,
    IndirectnessOutput,
    ElementJudgement,
    assess_outcomes,
    integrate,
)
from medical_deep_research.grade.domains import (
    _ois_binary,
    imprecision,
    inconsistency,
    rating_up,
    risk_of_bias,
    sum_certainty,
    DomainJudgement,
)
from medical_deep_research.grade.models import (
    CertaintyTarget,
    OutcomeSpec,
    RiskOfBias,
    StudyDesign,
    StudyExtraction,
    StudyOutcomeData,
)
from medical_deep_research.grade.synthesis import EvidenceBody, build_bodies, orient, pool, rr_from_counts
from medical_deep_research.models import RunRequest

SBP = OutcomeSpec(id="o1", name="PA sistólica", kind="continuous", desirable_direction="decrease")
CONTROL = OutcomeSpec(id="o2", name="Controle da PA", kind="binary", desirable_direction="increase")


def _item(ref: int, est: float, lo: float, hi: float, measure: str = "MD", *, design=StudyDesign.RCT,
          favours: str = "intervention", n: int = 200, rob=RiskOfBias.LOW, outcome: str = "o1",
          events_c: int | None = None, n_c: int | None = None, review_grade: str | None = None) -> StudyOutcomeData:
    return StudyOutcomeData(
        study_ref=ref, outcome_id=outcome, reported=True, design=design, effect_measure=measure, effect=est,
        ci_low=lo, ci_high=hi, favours=favours, n_intervention=n // 2, n_control=n_c or n // 2,
        events_control=events_c, risk_of_bias=rob, quote="q", source_text="fulltext", review_grade=review_grade,
    )


def _body(items: list[StudyOutcomeData], outcome: OutcomeSpec = SBP) -> EvidenceBody:
    return build_bodies([outcome], [StudyExtraction(study_ref=i.study_ref, items=[i]) for i in items])[0]


class OrientationAndPoolingTests(unittest.TestCase):
    def test_reduction_reported_as_positive_is_flipped(self) -> None:
        self.assertEqual(orient(_item(1, 3.27, 2.4, 4.15), SBP), (-3.27, -4.15, -2.4))
        self.assertEqual(orient(_item(2, -3.5, -5.8, -1.2), SBP), (-3.5, -5.8, -1.2))

    def test_ratio_for_desirable_increase(self) -> None:
        inverted = orient(_item(1, 0.5, 0.4, 0.625, "RR", favours="intervention"), CONTROL)
        self.assertAlmostEqual(inverted[0], 2.0)
        self.assertAlmostEqual(inverted[1], 1.6)

    def test_random_effects_pool(self) -> None:
        pooled = pool("MD", [(1, -3.0, -5.0, -1.0), (2, -4.0, -6.0, -2.0)])
        self.assertAlmostEqual(pooled.estimate, -3.5, places=6)
        self.assertEqual(pooled.k, 2)
        self.assertEqual(pooled.i2, 0.0)
        heterogeneous = pool("MD", [(1, -10.0, -12.0, -8.0), (2, 2.0, 0.0, 4.0)])
        self.assertGreater(heterogeneous.i2, 90)

    def test_bodies_split_by_design_and_mixed_measures_not_combined(self) -> None:
        items = [_item(1, -3, -5, -1), _item(2, -4, -6, -2), _item(3, -2, -4, 0, design=StudyDesign.NON_RANDOMIZED),
                 _item(4, -0.3, -0.5, -0.1, "SMD")]
        bodies = build_bodies([SBP], [StudyExtraction(study_ref=i.study_ref, items=[i]) for i in items])
        rct = next(b for b in bodies if b.design == "rct")
        self.assertEqual(rct.pooled.k, 2)
        self.assertTrue(any("SMD" in note for note in rct.not_pooled))
        self.assertEqual(next(b for b in bodies if b.design == "nrs").refs, [3])


class CalibrationTests(unittest.TestCase):
    def test_rr_from_counts_when_no_effect_reported(self) -> None:
        item = StudyOutcomeData(study_ref=9, outcome_id="o2", reported=True, design=StudyDesign.RCT,
                                events_intervention=60, n_intervention=100, events_control=40, n_control=100)
        rr, lo, hi = rr_from_counts(item)
        self.assertAlmostEqual(rr, 1.5)
        self.assertLess(lo, 1.5)
        self.assertGreater(hi, 1.5)
        body = build_bodies([CONTROL], [StudyExtraction(study_ref=9, items=[item])])[0]
        self.assertEqual(body.pooled.measure, "RR")
        self.assertTrue(body.derived)

    def test_unknown_participants_do_not_trigger_ois_downgrade(self) -> None:
        known = _item(1, -3, -5, -1, n=300)
        unknown = _item(2, -4, -6, -2, n=0)
        unknown.n_intervention = unknown.n_control = None
        body = _body([known, unknown])
        self.assertFalse(body.participants_complete)
        judgement = imprecision(body, SBP)
        self.assertEqual(judgement.rating, 0)
        self.assertIn("could not be verified", judgement.rationale)


class DomainRuleTests(unittest.TestCase):
    def test_imprecision_continuous_core_grade_flow(self) -> None:
        crosses = _body([_item(1, -2, -4, 0.5, n=1000)])
        self.assertEqual(imprecision(crosses, SBP).rating, -1)
        tiny_crossing = _body([_item(1, -2, -6, 2, n=100)])
        self.assertEqual(imprecision(tiny_crossing, SBP).rating, -2)
        small = _body([_item(1, -3, -5, -1, n=500)])
        self.assertEqual(imprecision(small, SBP).rating, -1)  # below OIS 800
        large = _body([_item(1, -3, -5, -1, n=1200)])
        self.assertEqual(imprecision(large, SBP).rating, 0)

    def test_imprecision_against_mid(self) -> None:
        mid_outcome = SBP.model_copy(update={"target": CertaintyTarget.MID, "mid_value": 5.0, "mid_source": "x"})
        crosses_mid = _body([_item(1, -4, -7, -1, n=1200)], mid_outcome)
        self.assertEqual(imprecision(crosses_mid, mid_outcome).rating, -1)

    def test_imprecision_binary_ratio_rules(self) -> None:
        wide = _body([_item(1, 2.0, 1.1, 3.6, "RR", outcome="o2")], CONTROL)
        self.assertEqual(imprecision(wide, CONTROL).rating, -2)  # upper/lower >= 3
        self.assertGreater(_ois_binary(0.3), 0)

    def test_inconsistency_and_single_study(self) -> None:
        self.assertEqual(inconsistency(_body([_item(1, -3, -5, -1)])).rating, 0)
        opposite = _body([_item(1, -10, -12, -8), _item(2, 3, 1, 5, favours="control")])
        self.assertEqual(inconsistency(opposite).rating, -1)

    def test_risk_of_bias_by_weight(self) -> None:
        mostly_high = _body([_item(1, -3, -5, -1, n=800, rob=RiskOfBias.HIGH), _item(2, -3, -5, -1, n=200)])
        self.assertEqual(risk_of_bias(mostly_high).rating, -1)
        mostly_low = _body([_item(1, -3, -5, -1, n=200, rob=RiskOfBias.HIGH), _item(2, -3, -5, -1, n=800)])
        self.assertEqual(risk_of_bias(mostly_low).rating, 0)

    def test_rating_up_only_for_clean_non_randomized_bodies(self) -> None:
        nrs = _body([_item(1, 0.3, 0.2, 0.45, "RR", design=StudyDesign.NON_RANDOMIZED, favours="intervention")])
        self.assertEqual(rating_up(nrs, []).rating, 1)
        self.assertEqual(rating_up(nrs, [DomainJudgement(domain="x", rating=-1)]).rating, 0)
        rct = _body([_item(1, 0.3, 0.2, 0.45, "RR")])
        self.assertEqual(rating_up(rct, []).rating, 0)
        self.assertEqual(sum_certainty(2, [DomainJudgement(domain="u", rating=2)]), 4)
        self.assertEqual(sum_certainty(4, [DomainJudgement(domain="d", rating=-2),
                                           DomainJudgement(domain="e", rating=-2)]), 1)


class IntegrationTests(unittest.TestCase):
    def _b(self, design: str, certainty: int) -> BodyAssessment:
        return BodyAssessment(outcome_id="o1", design=design, start=4, refs=[], certainty_a=certainty)

    def test_grade24_rules(self) -> None:
        self.assertEqual(integrate([self._b("rct", 4), self._b("nrs", 2)], "certainty_a")[0], 4)
        value, note = integrate([self._b("rct", 1), self._b("nrs", 2)], "certainty_a")
        self.assertEqual(value, 2)
        self.assertIn("replaces", note)
        self.assertEqual(integrate([], "certainty_a"), (None, "no evidence"))


class AssessmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_modes_a_and_b_with_review_comparison_and_coherence(self) -> None:
        request = RunRequest(run_id="r", query="q", query_type="pico", mode="detailed", provider="local", model="m")
        items = [_item(1, -3, -5, -1, n=500, review_grade="High", design=StudyDesign.SYSTEMATIC_REVIEW),
                 _item(2, -4, -6, -2, n=500)]
        bodies = build_bodies([SBP], [StudyExtraction(study_ref=i.study_ref, items=[i]) for i in items])
        seen_prompts: list[str] = []

        async def structured_call(task: str, _instructions: str, prompt: str, model: type) -> Any:
            seen_prompts.append(prompt)
            if model is IndirectnessOutput:
                return IndirectnessOutput(intervention=ElementJudgement(concern="serious", rationale="mixed"),
                                          rating=-2, rationale="multicomponent")
            if model is AdjustmentOutput:
                return AdjustmentOutput(adjustment=3, rationale="too much")
            return HolisticOutput(domains=[HolisticDomain(domain="rating_up", rating=1)], certainty="high")

        profile = (await assess_outcomes(structured_call, request, [SBP], bodies))[0]
        body = profile.bodies[0]

        indirect = next(d for d in body.domains if d.domain == "indirectness")
        self.assertEqual(indirect.rating, -1)  # capped by the element concerns (only "serious")
        self.assertEqual(body.certainty_computed, 3)  # high, -1 indirectness; 1000 participants >= OIS 800
        self.assertEqual(body.adjustment_a.adjustment, 1)  # clamped to +-1
        self.assertEqual(body.certainty_a, 4)
        self.assertIn("mode B rated up a randomized body", body.coherence_flags)
        self.assertFalse(any("High" in p for p in seen_prompts))  # the review's GRADE is never shown
        self.assertIn("agrees", profile.comparison_notes[0])  # review said High; mode A reached high


if __name__ == "__main__":
    unittest.main()


class IndirectnessCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_intervention_only_very_serious_is_capped_at_minus_one(self) -> None:
        request = RunRequest(run_id="r", query="q", query_type="pico", mode="detailed", provider="local", model="m")
        bodies = build_bodies([SBP], [StudyExtraction(study_ref=1, items=[_item(1, -3, -5, -1, n=1000)])])

        async def structured_call(_task: str, _i: str, _p: str, model: type) -> Any:
            if model is IndirectnessOutput:
                return IndirectnessOutput(intervention=ElementJudgement(concern="very serious"), rating=-2)
            return model()

        body = (await assess_outcomes(structured_call, request, [SBP], bodies))[0].bodies[0]
        self.assertEqual(next(d for d in body.domains if d.domain == "indirectness").rating, -1)


class SubgroupTests(unittest.TestCase):
    def _bodies(self, expected: str):
        from medical_deep_research.grade.models import SubgroupHypothesis

        hypothesis = SubgroupHypothesis(id="s1", variable="co-interventions", levels=["alone", "titration"],
                                        expected_larger_benefit=expected)
        rows = [(1, -1.0, -2.0, 0.0, "alone"), (2, -1.2, -2.2, -0.2, "alone"),
                (3, -8.0, -9.0, -7.0, "titration"), (4, -7.5, -8.5, -6.5, "titration")]
        extractions = [StudyExtraction(study_ref=r, subgroup_levels={"s1": lvl},
                                       items=[_item(r, est, lo, hi, n=1000)]) for r, est, lo, hi, lvl in rows]
        return build_bodies([SBP], extractions, [hypothesis])[0]

    def test_credible_subgroup_explains_heterogeneity(self) -> None:
        body = self._bodies("titration")
        self.assertTrue(body.subgroups[0].credible)
        judgement = inconsistency(body)
        self.assertEqual(judgement.rating, 0)
        self.assertIn("explained by the pre-specified subgroup", judgement.rationale)

    def test_subgroup_in_unexpected_direction_does_not_explain(self) -> None:
        body = self._bodies("alone")
        self.assertFalse(body.subgroups[0].credible)
        self.assertEqual(inconsistency(body).rating, -1)  # strong unexplained heterogeneity stays rated down


class ComparisonNormalizationTests(unittest.TestCase):
    def test_subgroups_capped_and_cleaned(self) -> None:
        from medical_deep_research.grade.models import ComparisonSpec, SubgroupHypothesis
        from medical_deep_research.grade.outcomes import normalize_comparison

        spec = ComparisonSpec(intervention="x", subgroups=[
            SubgroupHypothesis(id="", variable="a", levels=["1", "2"], expected_larger_benefit="3"),
            SubgroupHypothesis(id="s2", variable="b", levels=["only one"]),
            *[SubgroupHypothesis(id=f"s{i}", variable=f"v{i}", levels=["p", "q"]) for i in range(5)],
        ])
        cleaned = normalize_comparison(spec)
        self.assertEqual(len(cleaned.subgroups), 2)  # 3 kept by the cap, one dropped for a single level
        self.assertEqual(cleaned.subgroups[0].id, "s1")
        self.assertEqual(cleaned.subgroups[0].expected_larger_benefit, "")  # not one of the levels
