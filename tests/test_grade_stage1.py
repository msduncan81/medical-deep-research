from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from medical_deep_research.grade.extraction import _sanitize, studies_to_extract
from medical_deep_research.grade.graph import run_stage1
from medical_deep_research.grade.models import (
    CertaintyTarget,
    OutcomeImportance,
    OutcomeProposal,
    OutcomeSpec,
    ReviewCoverage,
    StudyDesign,
    StudyExtraction,
    StudyOutcomeData,
)
from medical_deep_research.grade.outcomes import confirm_outcomes, normalize_outcomes
from medical_deep_research.grade.reviews import build_review_map, is_systematic_review, match_references
from medical_deep_research.models import ApprovalRequest, ApprovalStatus, RunRequest
from medical_deep_research.research.models import EvidenceStudy, ScoredStudy, SearchProviderResult


def _study(ref: int, title: str, *, year: str = "2020", pmid: str | None = None, types: list[str] | None = None,
           abstract: str = "") -> ScoredStudy:
    return ScoredStudy(
        source="PubMed", source_id=f"s{ref}", title=title, publication_year=year, pmid=pmid,
        publication_types=types or [], abstract=abstract, sources=["PubMed"], evidence_level_score=0.5,
        citation_score=0.5, recency_score=0.5, composite_score=0.5, reference_number=ref,
    )


def _request(database_path: str | None = None) -> RunRequest:
    return RunRequest(
        run_id="grade-run", query="Population: adults with hypertension; Intervention: home BP monitoring",
        query_type="pico", mode="detailed", provider="local", model="m", language="pt",
        database_path=database_path,
    )


class OutcomeNormalizationTests(unittest.TestCase):
    def test_grade_rules_are_enforced(self) -> None:
        proposal = OutcomeProposal(outcomes=[
            OutcomeSpec(id="o1", name="PA sistólica", importance_score=5, target=CertaintyTarget.MID, mid_value=5),
            OutcomeSpec(id="o1", name="Eventos cardiovasculares", importance_score=9),
            *[OutcomeSpec(id=f"x{i}", name=f"Desfecho {i}", importance_score=4) for i in range(8)],
        ])
        outcomes = normalize_outcomes(proposal)

        self.assertEqual(len(outcomes), 7)
        self.assertEqual(outcomes[0].name, "Eventos cardiovasculares")  # critical first
        self.assertEqual(outcomes[0].importance, OutcomeImportance.CRITICAL)
        self.assertEqual(len({o.id for o in outcomes}), 7)  # ids made unique
        sbp = next(o for o in outcomes if o.name == "PA sistólica")
        self.assertEqual(sbp.target, CertaintyTarget.NULL)  # MID without a source is not accepted
        self.assertIsNone(sbp.mid_value)


class ReviewMapTests(unittest.TestCase):
    def test_overlapping_reviews_choose_most_recent_and_compute_cca(self) -> None:
        primaries = [_study(3, "Trial A of home blood pressure telemonitoring", pmid="111"),
                     _study(4, "Trial B of self measured blood pressure in practice", pmid="222"),
                     _study(5, "Trial C unrelated cohort of hypertension care", pmid="333")]
        refs_old = [EvidenceStudy(source="x", source_id="r1", title="t", pmid="111")]
        refs_new = [EvidenceStudy(source="x", source_id="r2", title="t", pmid="111"),
                    EvidenceStudy(source="x", source_id="r3", title="Trial B of self measured blood pressure in practice")]
        old = ReviewCoverage(review_ref=1, title="SR 2013", year=2013, covered_refs=match_references(refs_old, primaries),
                             reference_count=1)
        new = ReviewCoverage(review_ref=2, title="SR 2021", year=2021, covered_refs=match_references(refs_new, primaries),
                             reference_count=2)
        review_map = build_review_map([old, new])

        self.assertEqual(new.covered_refs, [3, 4])  # matched by PMID and by title
        self.assertEqual(review_map.groups, [[1, 2]])
        self.assertTrue(next(r for r in review_map.reviews if r.review_ref == 2).chosen)
        self.assertFalse(next(r for r in review_map.reviews if r.review_ref == 1).chosen)
        self.assertEqual(review_map.covered_by_chosen, {3: 2, 4: 2})
        self.assertEqual(review_map.corrected_covered_area, 0.5)  # N=3, r=2, c=2

    def test_systematic_review_detection(self) -> None:
        self.assertTrue(is_systematic_review(_study(1, "Self-monitoring: a meta-analysis")))
        self.assertTrue(is_systematic_review(_study(1, "Home BP", types=["Systematic Review"])))
        self.assertFalse(is_systematic_review(_study(1, "A randomised trial of home BP")))


class ExtractionRulesTests(unittest.TestCase):
    def test_covered_primaries_and_non_chosen_reviews_are_skipped(self) -> None:
        studies = [_study(1, "SR 2013 meta-analysis"), _study(2, "SR 2021 meta-analysis"),
                   _study(3, "Trial A"), _study(5, "Trial C")]
        old = ReviewCoverage(review_ref=1, title="", year=2013, covered_refs=[3])
        new = ReviewCoverage(review_ref=2, title="", year=2021, covered_refs=[3])
        selected, notes = studies_to_extract(studies, build_review_map([old, new]))

        self.assertEqual([s.reference_number for s in selected], [2, 5])
        self.assertEqual(len(notes), 2)

    def test_numbers_without_quote_or_outside_ci_are_dropped(self) -> None:
        study = _study(7, "Trial")
        no_quote = StudyOutcomeData(outcome_id="o1", reported=True, effect=0.8, ci_low=0.6, ci_high=1.0,
                                    effect_measure="RR")
        self.assertIsNone(_sanitize(no_quote, study, {"o1"}).effect)

        bad_ci = StudyOutcomeData(outcome_id="o1", reported=True, effect=2.0, ci_low=0.6, ci_high=1.0, quote="RR 2.0",
                                  effect_measure="risk ratio")
        cleaned = _sanitize(bad_ci, study, {"o1"})
        self.assertEqual(cleaned.effect, 2.0)
        self.assertIsNone(cleaned.ci_low)

        self.assertIsNone(_sanitize(StudyOutcomeData(outcome_id="zz"), study, {"o1"}))

    def test_single_group_proportion_is_not_an_effect_and_measures_are_normalized(self) -> None:
        study = _study(23, "Trial")
        proportion = StudyOutcomeData(outcome_id="o1", reported=True, effect=13.0, effect_measure="percentage",
                                      quote="25 (13%) patients had controlled BP", favours="sideways")
        cleaned = _sanitize(proportion, study, {"o1"})
        self.assertIsNone(cleaned.effect)
        self.assertIsNone(cleaned.effect_measure)
        self.assertEqual(cleaned.favours, "unclear")

        md = StudyOutcomeData(outcome_id="o1", reported=True, effect=-3.5, ci_low=-5.8, ci_high=-1.2,
                              effect_measure="mean difference", quote="MD -3.5", favours="intervention")
        self.assertEqual(_sanitize(md, study, {"o1"}).effect_measure, "MD")

    def test_network_meta_analysis_is_never_the_chosen_review(self) -> None:
        nma = ReviewCoverage(review_ref=1, title="NMA", year=2025, covered_refs=[3], network=True)
        pairwise = ReviewCoverage(review_ref=2, title="SR", year=2020, covered_refs=[3])
        review_map = build_review_map([nma, pairwise])
        self.assertEqual([r.review_ref for r in review_map.reviews if r.chosen], [2])

        alone = build_review_map([ReviewCoverage(review_ref=1, title="NMA", year=2025, covered_refs=[3], network=True)])
        self.assertEqual(alone.covered_by_chosen, {})  # its trials are assessed directly


class Stage1GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_runs_end_to_end_without_database(self) -> None:
        studies = [_study(1, "Self-monitoring of blood pressure: a meta-analysis", year="2021", pmid="900",
                          abstract="Pooled MD -3.2 mmHg (95% CI -4.1 to -2.3)."),
                   _study(2, "Randomised trial of telemonitoring", pmid="111", abstract="..."),
                   _study(3, "Cohort of home monitoring", pmid="333", abstract="...")]
        calls: list[str] = []

        async def structured_call(task: str, _instructions: str, prompt: str, model: type) -> Any:
            calls.append(task)
            if model is OutcomeProposal:
                return OutcomeProposal(outcomes=[OutcomeSpec(id="o1", name="PA sistólica", importance_score=7)])
            ref = int(prompt.split("STUDY [")[1].split("]")[0])
            return StudyExtraction(items=[StudyOutcomeData(outcome_id="o1", reported=True, effect=-3.2,
                                                           ci_low=-4.1, ci_high=-2.3, quote="MD -3.2", study_ref=ref)])

        async def fetch_references(study: EvidenceStudy) -> SearchProviderResult:
            refs = [EvidenceStudy(source="x", source_id="r", title="t", pmid="111")] if study.pmid == "900" else []
            return SearchProviderResult(source="snowball", query="q", studies=refs)

        result = await run_stage1(_request(), studies, structured_call, fetch_references, lambda _s: None,
                                  fetch_full=None)

        self.assertFalse(result.outcomes_confirmed_by_user)
        self.assertEqual([o.name for o in result.outcomes], ["PA sistólica"])
        self.assertEqual(result.review_map.covered_by_chosen, {2: 1})
        self.assertEqual(sorted(e.study_ref for e in result.extractions), [1, 3])  # trial 2 is inside the review
        review_item = next(e for e in result.extractions if e.study_ref == 1).items[0]
        self.assertEqual(review_item.design, StudyDesign.SYSTEMATIC_REVIEW)
        self.assertEqual(calls.count("grade_extraction"), 2)


class OutcomeConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_edits_are_used(self) -> None:
        from sqlmodel import Session, SQLModel, create_engine, select

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.sqlite")
            engine = create_engine(f"sqlite:///{db}")
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                from medical_deep_research.models import ResearchRun

                session.add(ResearchRun(id="grade-run", query="q", query_type="pico", mode="detailed",
                                        provider="local", model="m", runtime_name="r"))
                session.commit()
            proposed = [OutcomeSpec(id="o1", name="PA sistólica", importance_score=6)]

            async def user() -> None:
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    with Session(engine) as session:
                        approval = session.exec(select(ApprovalRequest)).first()
                        if approval:
                            details = json.loads(approval.details_json)
                            details["confirmed"] = [{"id": "o1", "name": "Controle da PA", "importance_score": 8,
                                                     "target": "mid", "mid_value": 5, "mid_source": "ref"}]
                            approval.details_json = json.dumps(details)
                            approval.status = ApprovalStatus.APPROVED.value
                            session.add(approval)
                            session.commit()
                            return

            outcomes, _comparison, confirmed, _note = (await asyncio.gather(
                confirm_outcomes(_request(db), proposed, poll_seconds=0.02), user()
            ))[0]

        self.assertTrue(confirmed)
        self.assertEqual(outcomes[0].name, "Controle da PA")
        self.assertEqual(outcomes[0].target, CertaintyTarget.MID)
        self.assertEqual(outcomes[0].importance, OutcomeImportance.CRITICAL)


if __name__ == "__main__":
    unittest.main()
