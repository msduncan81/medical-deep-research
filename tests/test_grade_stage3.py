from __future__ import annotations

import unittest
from typing import Any

from medical_deep_research.grade.models import GradeResult, OutcomeSpec
from medical_deep_research.grade.report import ReportProse, OutcomeSection, assemble_report, build_grade_report, report_issues
from medical_deep_research.grade.sof import informative_statement, summary_of_findings
from medical_deep_research.models import RunRequest
from medical_deep_research.research.models import ScoredStudy


def _profile(oid: str, name: str, certainty: int | None, measure: str = "MD", est: float = -3.3,
             lo: float = -4.3, hi: float = -2.3, kind: str = "continuous", cer: float | None = None,
             domains: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    outcome = OutcomeSpec(id=oid, name=name, kind=kind, importance_score=8).model_dump(mode="json")
    bodies = [] if certainty is None else [{
        "design": "rct", "refs": [1, 2], "participants": 900, "control_event_rate": cer,
        "estimate": {"measure": measure, "estimate": est, "ci_low": lo, "ci_high": hi, "k": 2},
        "domains": domains or [], "adjustment_a": {"adjustment": 0, "rationale": ""}, "certainty_a": certainty,
        "subgroups": [],
    }]
    return {"outcome": outcome, "bodies": bodies, "certainty_a": certainty, "certainty_b": certainty,
            "comparison_notes": []}


def _request() -> RunRequest:
    return RunRequest(run_id="r", query="Q", query_type="pico", mode="detailed", provider="local", model="m",
                      language="pt", query_payload={"intervention": "AMPA"})


def _studies() -> list[ScoredStudy]:
    return [ScoredStudy(source="PubMed", source_id=str(i), title=f"Trial {i}", publication_year="2020", pmid=str(i),
                        sources=["PubMed"], evidence_level_score=0.5, citation_score=0.5, recency_score=0.5,
                        composite_score=0.5, reference_number=i) for i in (1, 2, 3)]


class StatementTests(unittest.TestCase):
    def test_wording_by_certainty(self) -> None:
        self.assertEqual(informative_statement(_profile("o1", "PA sistólica", 4), "AMPA", "pt"), "AMPA reduz PA sistólica")
        self.assertEqual(informative_statement(_profile("o1", "PA sistólica", 3), "AMPA", "pt"),
                         "AMPA provavelmente reduz PA sistólica")
        self.assertEqual(informative_statement(_profile("o1", "PA sistólica", 2), "AMPA", "pt"),
                         "AMPA pode reduzir PA sistólica")
        self.assertIn("muito incerta", informative_statement(_profile("o1", "PA sistólica", 1), "AMPA", "pt"))
        self.assertIn("Nenhum estudo", informative_statement(_profile("o1", "Mortalidade", None), "AMPA", "pt"))

    def test_little_or_no_difference_and_large_effects(self) -> None:
        null_ci = _profile("o2", "Mortalidade", 3, "RR", 1.03, 0.63, 1.69, "binary")
        self.assertIn("pouca ou nenhuma diferença", informative_statement(null_ci, "AMPA", "pt"))
        large = _profile("o3", "Controle da PA", 4, "RR", 2.4, 2.0, 2.9, "binary")
        self.assertEqual(informative_statement(large, "AMPA", "pt"), "AMPA aumenta muito controle da PA")
        self.assertEqual(informative_statement(large, "SMBP", "en"), "SMBP results in a large increase in controle da PA")


class SummaryOfFindingsTests(unittest.TestCase):
    def test_table_with_absolute_effect_and_footnotes(self) -> None:
        domains = [{"domain": "indirectness", "rating": -1, "rationale": "intervenções multicomponentes"}]
        profiles = [_profile("o1", "Controle da PA", 3, "RR", 1.3, 1.1, 1.5, "binary", cer=0.4, domains=domains),
                    _profile("o2", "Mortalidade", None)]
        table = summary_of_findings(profiles, "AMPA", "pt")
        self.assertIn("400 por 1.000", table)
        self.assertIn("520 por 1.000 (120 a mais", table)
        self.assertIn("⊕⊕⊕◯ Moderada (a)", table)
        self.assertIn("a. Controle da PA: rebaixada 1 nível por evidência indireta", table)
        self.assertIn("Sem evidência", table)


class ReportAssemblyTests(unittest.IsolatedAsyncioTestCase):
    def _result(self) -> GradeResult:
        return GradeResult(profiles=[_profile("o1", "PA sistólica", 3), _profile("o2", "Mortalidade", None)])

    def test_code_only_report_has_statements_table_and_references(self) -> None:
        report = assemble_report(self._result(), _request(), _studies(), None)
        self.assertIn("## 3. Resumo dos achados", report)
        self.assertIn("**AMPA provavelmente reduz PA sistólica.**", report)
        self.assertIn("## 7. Referências", report)
        self.assertEqual(report_issues(report, self._result(), _request()), [])

    async def test_model_prose_is_placed_and_unknown_citations_removed(self) -> None:
        async def structured_call(_task: str, _i: str, _p: str, _m: type) -> ReportProse:
            return ReportProse(title="AMPA na APS", summary="Resumo [1] e [99].",
                               outcome_sections=[OutcomeSection(outcome_id="o1", text="Nove ensaios [2].")],
                               discussion="d", conclusions="c")

        report, notes = await build_grade_report(structured_call, self._result(), _request(), _studies())
        self.assertTrue(report.startswith("# AMPA na APS"))
        self.assertIn("Nove ensaios [2].", report)
        self.assertNotIn("[99]", report)
        self.assertTrue(any("99" in n for n in notes))
        self.assertLess(report.index("**AMPA provavelmente reduz PA sistólica.**"), report.index("Nove ensaios"))


if __name__ == "__main__":
    unittest.main()
