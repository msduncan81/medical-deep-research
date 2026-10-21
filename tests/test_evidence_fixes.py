from __future__ import annotations

import unittest
from unittest.mock import patch

from medical_deep_research.agentic_tools import AgenticEventBridge, tool_finalize_ranking
from medical_deep_research.models import RunRequest
from medical_deep_research.research.models import ScoredStudy
from medical_deep_research.research.search import cochrane_evidence_level, infer_evidence_level
from medical_deep_research.research.verification import verify_studies


def _request() -> RunRequest:
    return RunRequest(
        run_id="evidence-fixes",
        query="Population: adults with hypertension; Intervention: home BP monitoring; Comparison: usual care; Outcome: BP control",
        query_type="pico",
        mode="detailed",
        provider="local",
        model="qwen",
        language="en",
        api_keys={},
        offline_mode=False,
    )


def _study(i: int, *, pmid: str | None = None, doi: str | None = None) -> ScoredStudy:
    return ScoredStudy(
        source="PubMed",
        source_id=f"s{i}",
        title=f"Study {i} on home blood pressure monitoring",
        pmid=pmid,
        doi=doi,
        evidence_level="Level II",
        sources=["PubMed"],
        evidence_level_score=0.8,
        citation_score=0.5,
        recency_score=0.5,
        composite_score=0.6,
        reference_number=i,
    )


class InferEvidenceLevelTests(unittest.TestCase):
    def test_narrative_review_is_level_v_not_level_i(self) -> None:
        self.assertEqual(infer_evidence_level("Hypertension management in primary care", ["Review"]), "Level V")
        self.assertEqual(infer_evidence_level("Hypertension management", ["review"]), "Level V")  # OpenAlex
        self.assertEqual(infer_evidence_level("Hypertension management", ["review-article"]), "Level V")  # Europe PMC

    def test_systematic_reviews_and_meta_analyses_are_level_i(self) -> None:
        self.assertEqual(infer_evidence_level("Home BP monitoring", ["Systematic Review", "Review"]), "Level I")
        self.assertEqual(infer_evidence_level("Home BP monitoring", ["Meta-Analysis"]), "Level I")
        self.assertEqual(infer_evidence_level("Home BP monitoring: a meta-analysis", ["Review"]), "Level I")

    def test_publication_type_is_used_even_when_title_is_silent(self) -> None:
        self.assertEqual(infer_evidence_level("Home BP monitoring in adults", ["Randomized Controlled Trial"]), "Level II")
        self.assertEqual(infer_evidence_level("Home BP monitoring in adults", ["Observational Study"]), "Level III")
        self.assertEqual(infer_evidence_level("Home BP monitoring in adults", ["Case Reports"]), "Level IV")

    def test_non_randomized_clinical_trial_is_not_level_ii(self) -> None:
        self.assertEqual(infer_evidence_level("A clinical trial of home BP monitoring", []), "Level III")
        self.assertEqual(infer_evidence_level("A randomized clinical trial of home BP monitoring", []), "Level II")

    def test_guidelines_are_not_placed_on_the_level_scale(self) -> None:
        self.assertIsNone(infer_evidence_level("Hypertension guideline 2025", ["Practice Guideline"]))
        self.assertIsNone(infer_evidence_level("Hypertension guideline 2025", ["Guideline"]))

    def test_unknown_designs_stay_unclassified(self) -> None:
        self.assertIsNone(infer_evidence_level("Home BP monitoring in adults", ["Journal Article"]))

    def test_cochrane_protocols_and_withdrawn_reviews_are_not_level_i(self) -> None:
        self.assertEqual(cochrane_evidence_level("Self-monitoring of blood pressure for hypertension"), "Level I")
        self.assertIsNone(cochrane_evidence_level("WITHDRAWN: Self-monitoring of blood pressure"))
        self.assertIsNone(cochrane_evidence_level("Self-monitoring of blood pressure (Protocol)"))


class VerificationCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_cited_studies_are_checked_and_dois_go_to_crossref(self) -> None:
        studies = [_study(i, pmid=str(10_000 + i)) for i in range(1, 13)]
        studies.append(_study(13, doi="10.1590/s0102-311x2024000101"))

        async def fake_pubmed(pmids: list[str], api_key: str | None = None) -> dict[str, tuple[bool | None, str | None]]:
            return {pmid: (True, None) for pmid in pmids}

        async def fake_crossref(dois: list[str]) -> dict[str, tuple[bool | None, str | None]]:
            return {doi: (True, None) for doi in dois}

        with (
            patch("medical_deep_research.research.verification.verify_pmids_in_pubmed", fake_pubmed),
            patch("medical_deep_research.research.verification.verify_dois_in_crossref", fake_crossref),
        ):
            summary = await verify_studies(studies, limit=30)

        self.assertEqual(summary.total_considered, 13)
        self.assertEqual(summary.verified_pmids, 12)
        self.assertEqual(summary.missing_pmids, 1)
        self.assertIn("No PMID; DOI verified in Crossref", [d.issue for d in summary.details])
        self.assertTrue(any("1 of 1" in note for note in summary.notes))


class FinalizeRankingScreeningTests(unittest.IsolatedAsyncioTestCase):
    def _bridge(self) -> AgenticEventBridge:
        bridge = AgenticEventBridge()
        bridge._pre_scored = [_study(1, pmid="1"), _study(2, pmid="2"), _study(3, pmid="3")]
        return bridge

    async def test_unscreened_pool_is_refused_once_then_only_ranked_studies_are_kept(self) -> None:
        bridge = self._bridge()

        first = await tool_finalize_ranking(_request(), bridge, [2])
        self.assertIn("error", first)
        self.assertEqual(first["unscreened_count"], 2)
        self.assertEqual(bridge.ranked_studies, [])

        second = await tool_finalize_ranking(_request(), bridge, [2])
        self.assertEqual(second["status"], "ok")
        self.assertIn("note", second)
        self.assertEqual([s.title for s in bridge.ranked_studies], ["Study 2 on home blood pressure monitoring"])
        self.assertEqual(bridge.ranked_studies[0].reference_number, 1)

    async def test_screened_pool_keeps_appending_remaining_screened_studies(self) -> None:
        bridge = self._bridge()
        bridge.screening = {"included": [1, 2, 3]}

        result = await tool_finalize_ranking(_request(), bridge, [3, 1])

        self.assertEqual(result["status"], "ok")
        self.assertEqual([s.source_id for s in bridge.ranked_studies], ["s3", "s1", "s2"])
        self.assertEqual([s.reference_number for s in bridge.ranked_studies], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()


class ExclusionTests(unittest.TestCase):
    def test_target_review_excluded_by_cd_number_or_pmid(self) -> None:
        from medical_deep_research.agentic_tools import is_excluded

        request = _request().model_copy(update={"exclude_ids": ["CD006591", "30000001"]})
        review = _study(1, doi="10.1002/14651858.CD006591.pub4")
        older = _study(2, doi="10.1002/14651858.cd006591.pub3")
        by_pmid = _study(3, pmid="30000001")
        other = _study(4, pmid="123", doi="10.1000/x")
        self.assertTrue(is_excluded(review, request))
        self.assertTrue(is_excluded(older, request))
        self.assertTrue(is_excluded(by_pmid, request))
        self.assertFalse(is_excluded(other, request))
        self.assertFalse(is_excluded(review, _request()))
