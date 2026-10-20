from __future__ import annotations

import re

from .models import QueryPlan, ScoredStudy, SearchProviderResult, VerificationSummary

# Maximum number of ranked studies that flow into synthesized reports.
#
# v2.9.5 widened database searches to 25 results/source, so the post-screened
# evidence set can exceed the old 20-study report window. Keep this high enough
# that a typical expanded run, such as 26 ranked studies, reaches synthesis
# without silently dropping citable references.
MAX_REPORT_STUDIES = 30


def _vancouver_author(name: str) -> str:
    """Normalize a single author name to Vancouver style ("Surname II")."""
    name = (name or "").strip()
    if not name:
        return ""
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    # Already in "Surname Initials" form (e.g. PubMed esummary "Ahmed O", "Walsh EI").
    if last.isalpha() and last.isupper() and len(last) <= 3:
        return name
    surname = parts[-1]
    initials = "".join(part[0].upper() for part in parts[:-1] if part and part[0].isalpha())
    return f"{surname} {initials}".strip()


def _format_authors(authors: list[str]) -> str:
    """Format an author list per Vancouver rules: up to 6 names, then 'et al.'."""
    formatted = [author for author in (_vancouver_author(a) for a in (authors or [])) if author]
    if not formatted:
        return ""
    if len(formatted) > 6:
        return ", ".join(formatted[:6]) + ", et al."
    return ", ".join(formatted)


def format_vancouver_citation(study: ScoredStudy) -> str:
    """Render one reference deterministically from structured metadata.

    Format: Authors. Title. JournalAbbrev. Year;Volume(Issue):Pages. doi:DOI. PMID: x.
    Missing fields are simply omitted — never fabricated.
    """
    parts: list[str] = []

    authors = _format_authors(study.authors)
    if authors:
        parts.append(authors if authors.endswith(".") else authors + ".")

    title = (study.title or "").strip()
    if title:
        parts.append(title if title.endswith(".") else title + ".")

    journal = study.journal_abbrev or study.journal
    if journal:
        journal_part = journal.rstrip(". ")
        if study.publication_year:
            journal_part += f". {study.publication_year}"
            if study.volume:
                journal_part += f";{study.volume}"
                if study.issue:
                    journal_part += f"({study.issue})"
                if study.pages:
                    journal_part += f":{study.pages}"
        parts.append(journal_part + ".")
    elif study.publication_year:
        parts.append(f"{study.publication_year}.")

    if study.doi:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", study.doi.strip())
        parts.append(f"doi:{doi}.")
    if study.pmid:
        parts.append(f"PMID: {study.pmid}.")

    return " ".join(parts).strip()


def render_reference_entries(studies: list[ScoredStudy]) -> str:
    """Render numbered reference entries ("[n] citation"), keyed by reference_number."""
    entries: list[str] = []
    for index, study in enumerate(studies, start=1):
        number = study.reference_number if study.reference_number is not None else index
        entries.append(f"[{number}] {format_vancouver_citation(study)}")
    return "\n\n".join(entries)


def render_reference_list(studies: list[ScoredStudy]) -> str:
    """Render a complete numbered references section from structured metadata."""
    body = render_reference_entries(studies)
    return "## 7. References\n\n" + body if body else "## 7. References\n"


_REPORT_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "# Research Report",
        "summary": "## 1. Executive Summary",
        "summary_found": "The evidence workflow identified {count} ranked studies relevant to the question. The leading evidence includes {first}{second}.",
        "summary_and": " and ",
        "summary_none": "The evidence workflow completed, but no retrievable studies were ranked. The search should be revised before drawing conclusions.",
        "background": "## 2. Background",
        "question": "Research question: {query}",
        "methods": "## 3. Methods",
        "methods_query": "The query was handled as a {query_type} question in the {domain} domain. The search used the main concepts: {keywords}.",
        "methods_hits": "Searches returned {hits} source records before deduplication and ranking. Executed sources: {sources}.",
        "methods_skipped": "Sources not executed or skipped: {sources}.",
        "methods_failed": "Sources with errors: {sources}.",
        "none": "none",
        "results": "## 4. Results/Findings",
        "study_line": "- [{ref}] {title}. Source: {sources}. Evidence: {level}. Score: {score}. Citations: {citations}.",
        "unknown": "Unknown",
        "abstract": "  Abstract summary: {snippet}",
        "no_studies": "- No ranked studies available.",
        "discussion": "## 5. Discussion",
        "discussion_text": "The ranked evidence should be interpreted with attention to indirectness, study design, and the amount of full text available for appraisal. Automated ranking is not a substitute for duplicate human screening or a formal risk-of-bias review.",
        "verification": "Identifier verification checked the highest-ranked PMID-bearing records: {verified} PMIDs verified, {missing} records without PMIDs in the checked set, and {missing_pubmed} records missing from PubMed.",
        "conclusions": "## 6. Conclusions",
        "conclusions_text": "The available evidence should be treated as a mapped evidence base for review and follow-up screening. Conclusions should be strengthened only after full-text review of the key studies and explicit certainty appraisal.",
        "references": "## 7. References",
        "notes": "Notes:",
    },
    "pt": {
        "title": "# Relatório de Pesquisa",
        "summary": "## 1. Resumo",
        "summary_found": "O fluxo de busca de evidências identificou {count} estudos ranqueados relevantes para a pergunta. As principais evidências incluem {first}{second}.",
        "summary_and": " e ",
        "summary_none": "O fluxo de busca foi concluído, mas nenhum estudo recuperável foi ranqueado. A busca deve ser revista antes de qualquer conclusão.",
        "background": "## 2. Contexto",
        "question": "Pergunta de pesquisa: {query}",
        "methods": "## 3. Métodos",
        "methods_query": "A pergunta foi tratada como {query_type} no domínio {domain}. A busca usou os conceitos principais: {keywords}.",
        "methods_hits": "As buscas retornaram {hits} registros antes da remoção de duplicatas e do ranqueamento. Fontes consultadas: {sources}.",
        "methods_skipped": "Fontes não consultadas ou puladas: {sources}.",
        "methods_failed": "Fontes com erro: {sources}.",
        "none": "nenhum",
        "results": "## 4. Resultados",
        "study_line": "- [{ref}] {title}. Fonte: {sources}. Evidência: {level}. Pontuação: {score}. Citações: {citations}.",
        "unknown": "Não classificado",
        "abstract": "  Resumo do artigo: {snippet}",
        "no_studies": "- Nenhum estudo ranqueado disponível.",
        "discussion": "## 5. Discussão",
        "discussion_text": "As evidências ranqueadas devem ser interpretadas considerando a evidência indireta, o desenho dos estudos e quanto do texto completo esteve disponível para avaliação. O ranqueamento automático não substitui a triagem em duplicidade por pessoas nem uma avaliação formal do risco de viés.",
        "verification": "A conferência de identificadores verificou os registros mais bem ranqueados: {verified} PMIDs confirmados, {missing} registros sem PMID no conjunto conferido e {missing_pubmed} registros não encontrados no PubMed.",
        "conclusions": "## 6. Conclusões",
        "conclusions_text": "As evidências disponíveis devem ser tratadas como um mapa da literatura para revisão e nova triagem. As conclusões só devem ser reforçadas após leitura do texto completo dos estudos principais e avaliação explícita da certeza da evidência.",
        "references": "## 7. Referências",
        "notes": "Observações:",
    },
}


def _report_text(language: str | None) -> dict[str, str]:
    return _REPORT_TEXT.get((language or "en").lower(), _REPORT_TEXT["en"])


def _render_methods(
    plan: QueryPlan, results: list[SearchProviderResult], language: str | None = "en"
) -> list[str]:
    text = _report_text(language)
    searched = []
    skipped = []
    failed = []
    total_hits = 0
    for source in plan.databases:
        matched = next((result for result in results if result.source == source), None)
        if matched is None:
            skipped.append(source)
            continue
        if matched.skipped:
            skipped.append(source)
            continue
        if matched.error:
            failed.append(source)
            continue
        total_hits += len(matched.studies)
        searched.append(f"{source} ({len(matched.studies)})")
    lines = [
        text["methods"],
        "",
        text["methods_query"].format(
            query_type=plan.query_type.upper(),
            domain=plan.domain,
            keywords=", ".join(plan.keywords[:10]) or text["none"],
        ),
        "",
        text["methods_hits"].format(hits=total_hits, sources=", ".join(searched) or text["none"]),
    ]
    if skipped:
        lines.append(text["methods_skipped"].format(sources=", ".join(skipped)))
    if failed:
        lines.append(text["methods_failed"].format(sources=", ".join(failed)))
    return lines


def render_report(
    *,
    query: str,
    plan: QueryPlan,
    search_results: list[SearchProviderResult],
    ranked_studies: list[ScoredStudy],
    verification: VerificationSummary,
    provider: str,
    runtime_name: str,
    language: str | None = "en",
) -> str:
    text = _report_text(language)
    top_studies = ranked_studies[:MAX_REPORT_STUDIES]
    lines = [
        text["title"],
        "",
        text["summary"],
        "",
    ]
    del runtime_name, provider

    if top_studies:
        lines.append(
            text["summary_found"].format(
                count=len(ranked_studies),
                first=top_studies[0].title,
                second=(text["summary_and"] + top_studies[1].title) if len(top_studies) > 1 else "",
            )
        )
    else:
        lines.append(text["summary_none"])

    lines.extend(["", text["background"], "", text["question"].format(query=query)])
    lines.extend(_render_methods(plan, search_results, language))
    lines.extend(["", text["results"], ""])

    if top_studies:
        for study in top_studies:
            lines.append(
                text["study_line"].format(
                    ref=study.reference_number,
                    title=study.title,
                    sources=", ".join(study.sources or [study.source]),
                    level=study.evidence_level or text["unknown"],
                    score=study.composite_score,
                    citations=study.citation_count,
                )
            )
            if study.abstract:
                snippet = study.abstract.replace("\n", " ").strip()
                lines.append(text["abstract"].format(snippet=f"{snippet[:280]}{'...' if len(snippet) > 280 else ''}"))
    else:
        lines.append(text["no_studies"])

    lines.extend(
        [
            "",
            text["discussion"],
            "",
            text["discussion_text"],
            "",
            text["verification"].format(
                verified=verification.verified_pmids,
                missing=verification.missing_pmids,
                missing_pubmed=verification.missing_from_pubmed,
            ),
        ]
    )
    if verification.notes:
        lines.extend(f"- {note}" for note in verification.notes)

    lines.extend(["", text["conclusions"], "", text["conclusions_text"]])

    lines.extend(["", text["references"], ""])
    if top_studies:
        lines.append(render_reference_entries(top_studies))

    if plan.notes:
        lines.extend(["", text["notes"]])
        lines.extend(f"- {note}" for note in plan.notes)

    return "\n".join(lines)
