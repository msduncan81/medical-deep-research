"""Nodes D4-D5 — report organised by outcome, assembled by code around the model's prose.

The model writes only prose (summary, methods, one section per outcome, discussion, conclusions) as
structured JSON. Code inserts the Summary of Findings table, the informative statements (verbatim, first
sentence of each outcome section), the references and the appendix comparing modes A and B. Certainty is
never changed by the model. If the model fails, a code-only report is produced.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..languages import report_language_instruction
from ..models import RunRequest
from ..research.models import ScoredStudy
from ..research.reporting import render_reference_entries
from .models import GradeResult
from .outcomes import StructuredCall
from .sof import LEVEL_LABELS, informative_statement, modes_table, summary_of_findings


class OutcomeSection(BaseModel):
    outcome_id: str
    text: str = ""


class ReportProse(BaseModel):
    title: str = ""
    summary: str = ""
    methods: str = ""
    outcome_sections: list[OutcomeSection] = Field(default_factory=list)
    discussion: str = ""
    conclusions: str = ""


REPORT_INSTRUCTIONS = """\
You write the prose of an evidence report organised BY OUTCOME, following GRADE. The certainty of each
outcome, the Summary of Findings table and one informative statement per outcome are already fixed by
the assessment and are given to you: never change them, never re-rate certainty, never organise the
report by study design or evidence level.
Return JSON with: title; summary (2-3 paragraphs answering the question with the key statements);
methods (one paragraph: question and comparison, outcomes and whether the user confirmed them, how
reviews were mapped to avoid double counting, full-text availability, Core GRADE per outcome with
randomized and non-randomized evidence rated separately); outcome_sections (one per outcome id given,
explaining the estimate, the number of studies and participants, why certainty was rated down or up
domain by domain, credible subgroups, and abstract-only limitations); discussion (limitations,
applicability to the setting of the question); conclusions (practical takeaways consistent with the
certainty: "probably", "may" and "very uncertain" mean what they say).
Cite studies only as [n] using the reference numbers listed. Do not repeat the informative statement at
the start of an outcome section; it is inserted before your text."""


def _lang(language: str | None) -> str:
    return "pt" if (language or "").lower().startswith("pt") else "en"


_HEADINGS = {
    "pt": ["Resumo", "Pergunta e métodos", "Resumo dos achados", "Resultados por desfecho", "Discussão",
           "Conclusões", "Referências", "Apêndice: certeza nos modos A e B"],
    "en": ["Summary", "Question and methods", "Summary of findings", "Results by outcome", "Discussion",
           "Conclusions", "References", "Appendix: certainty in modes A and B"],
}


def intervention_label(result: GradeResult, request: RunRequest) -> str:
    if result.comparison.intervention:
        return result.comparison.intervention
    payload = request.original_query_payload or request.query_payload
    return payload.get("intervention") or ("a intervenção" if _lang(request.language) == "pt" else "the intervention")


def _report_prompt(result: GradeResult, request: RunRequest, studies: list[ScoredStudy], statements: dict[str, str],
                   sof: str) -> str:
    labels = LEVEL_LABELS[_lang(request.language)]
    outcome_lines = []
    for profile in result.profiles:
        outcome = profile["outcome"]
        domains = []
        for body in profile.get("bodies") or []:
            for d in body.get("domains") or []:
                if d["rating"]:
                    domains.append(f"{body['design']} {d['domain']} {d['rating']:+d}: {d['rationale']}")
            for g in body.get("subgroups") or []:
                domains.append(f"subgroup {g['variable']} (credible={g.get('credible')}): "
                               + "; ".join(f"{lv['level']} {lv['estimate']:.3g} [{lv['ci_low']:.3g}; {lv['ci_high']:.3g}]"
                                           for lv in g["levels"]) + f"; P={g.get('p_interaction')}")
        outcome_lines.append(
            f"- {outcome['id']} {outcome['name']} — certainty {labels[profile.get('certainty_a')]}; "
            f"statement: \"{statements[outcome['id']]}\"; " + (" | ".join(domains) or "no rating changes")
        )
    refs = "\n".join(f"[{s.reference_number}] {s.title} ({s.publication_year or 'n.d.'})" for s in studies[:40])
    extracted = len(result.extractions)
    fulltext = sum(1 for e in result.extractions if any(i.source_text == "fulltext" for i in e.items))
    chosen = [r for r in result.review_map.reviews if r.chosen]
    return (
        f"{report_language_instruction(request.language)}\n\n"
        f"Question: {request.original_query or request.query}\n"
        f"Intervention: {result.comparison.intervention or '?'}; comparator: {result.comparison.comparator or '?'}\n"
        f"Outcomes confirmed by the user: {'yes' if result.outcomes_confirmed_by_user else 'no'}\n"
        f"Studies extracted: {extracted} ({fulltext} with full text); reviews chosen: "
        f"{', '.join(f'[{r.review_ref}]' for r in chosen) or 'none'}; CCA {result.review_map.corrected_covered_area}\n\n"
        f"Outcomes:\n" + "\n".join(outcome_lines) + f"\n\nSummary of Findings table:\n{sof}\n\nReferences:\n{refs}"
    )


def _cited_refs(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d{1,3})\]", text)}


def assemble_report(result: GradeResult, request: RunRequest, studies: list[ScoredStudy], prose: ReportProse | None) -> str:
    lang = _lang(request.language)
    h = _HEADINGS[lang]
    intervention = intervention_label(result, request)
    statements = {p["outcome"]["id"]: informative_statement(p, intervention, request.language) for p in result.profiles}
    sof = summary_of_findings(result.profiles, intervention, request.language)
    prose = prose or ReportProse()
    title = prose.title.strip() or (request.original_query or request.query)
    by_id = {s.outcome_id: s.text.strip() for s in prose.outcome_sections}
    parts = [f"# {title}", "", f"## 1. {h[0]}", "",
             prose.summary.strip() or "\n".join(f"- {s}." for s in statements.values()), "",
             f"## 2. {h[1]}", "", prose.methods.strip() or "—", "", f"## 3. {h[2]}", "", sof, "",
             f"## 4. {h[3]}", ""]
    for profile in result.profiles:
        outcome = profile["outcome"]
        parts += [f"### {outcome['name']}", "", f"**{statements[outcome['id']]}.**", ""]
        if by_id.get(outcome["id"]):
            parts += [by_id[outcome["id"]], ""]
    parts += [f"## 5. {h[4]}", "", prose.discussion.strip() or "—", "", f"## 6. {h[5]}", "",
              prose.conclusions.strip() or "\n".join(f"- {s}." for s in statements.values()), ""]
    body_text = "\n".join(parts)
    cited = _cited_refs(body_text)
    in_bodies = {ref for p in result.profiles for b in p.get("bodies") or [] for ref in b.get("refs") or []}
    listed = [s for s in studies if int(s.reference_number or 0) in (cited | in_bodies)]
    parts += [f"## 7. {h[6]}", "", render_reference_entries(listed) if listed else "—", "",
              f"## {h[7]}", "", modes_table(result.profiles, request.language)]
    return "\n".join(parts).strip() + "\n"


async def build_grade_report(
    structured_call: StructuredCall, result: GradeResult, request: RunRequest, studies: list[ScoredStudy]
) -> tuple[str, list[str]]:
    """Return (markdown report, notes)."""
    notes: list[str] = []
    intervention = intervention_label(result, request)
    statements = {p["outcome"]["id"]: informative_statement(p, intervention, request.language) for p in result.profiles}
    sof = summary_of_findings(result.profiles, intervention, request.language)
    prose: ReportProse | None = None
    try:
        prose = await structured_call("grade_report", REPORT_INSTRUCTIONS,
                                      _report_prompt(result, request, studies, statements, sof), ReportProse)
    except Exception as exc:  # noqa: BLE001 - fall back to the code-only report
        notes.append(f"Report prose failed ({type(exc).__name__}); code-only report produced.")
    valid_refs = {int(s.reference_number or 0) for s in studies}
    if prose is not None:
        text = " ".join([prose.summary, prose.methods, prose.discussion, prose.conclusions,
                         *(s.text for s in prose.outcome_sections)])
        unknown = sorted(_cited_refs(text) - valid_refs)
        if unknown:
            notes.append(f"Citations to unknown references removed: {unknown}")
            pattern = re.compile(r"\[(" + "|".join(str(n) for n in unknown) + r")\]")
            prose.summary, prose.methods, prose.discussion, prose.conclusions = (
                pattern.sub("", x) for x in (prose.summary, prose.methods, prose.discussion, prose.conclusions)
            )
            for section in prose.outcome_sections:
                section.text = pattern.sub("", section.text)
    return assemble_report(result, request, studies, prose), notes


def report_issues(report: str, result: GradeResult, request: RunRequest) -> list[str]:
    """D5 — consistency checks between the report, the table and the profiles."""
    intervention = intervention_label(result, request)
    issues = []
    for profile in result.profiles:
        statement = informative_statement(profile, intervention, request.language)
        if statement not in report:
            issues.append(f"statement missing for {profile['outcome']['id']}")
    if "|---|" not in report:
        issues.append("summary of findings table missing")
    return issues

