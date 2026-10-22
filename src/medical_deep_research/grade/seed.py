"""Node B4b — seed the evidence with trials cited by the included systematic reviews.

Keyword searches tend to miss older landmark trials (e.g. the placebo-controlled hypertension trials of
the 1980s-90s), while the reviews found by the search already list them. Reference lists fetched by the
review map are mined for randomized trials not yet in the pool; their records are retrieved from Europe
PMC (abstract, publication types) and they go through full-text retrieval and extraction like any other
study. The comparison filter at extraction decides whether each one belongs to the question. Double
counting with the citing review is avoided per outcome when the bodies of evidence are built.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

import httpx

from ..research.models import EvidenceStudy, ScoredStudy
from ..research.search import USER_AGENT, infer_evidence_level
from .reviews import _norm_doi, _norm_title, is_systematic_review

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_SEEDS = 30
_TRIAL_RE = re.compile(r"randomi[sz]ed|randomised|controlled trial|placebo|double[- ]blind|\btrial\b", re.IGNORECASE)


def _key(study: EvidenceStudy) -> str:
    if study.pmid:
        return f"pmid:{study.pmid}"
    if study.doi:
        return f"doi:{_norm_doi(study.doi)}"
    return f"title:{_norm_title(study.title)}"


def select_seed_candidates(
    references_by_review: dict[int, list[EvidenceStudy]],
    pool: list[ScoredStudy],
    excluded: Callable[[EvidenceStudy], bool],
    limit: int = MAX_SEEDS,
) -> tuple[list[EvidenceStudy], dict[str, set[int]]]:
    """Trial-like references not in the pool, most-cited first. Returns (candidates, key -> citing reviews)."""
    known = {_key(s) for s in pool} | {f"title:{_norm_title(s.title)}" for s in pool}
    cited_by: dict[str, set[int]] = defaultdict(set)
    first: dict[str, EvidenceStudy] = {}
    for review_ref, references in references_by_review.items():
        for ref in references:
            key = _key(ref)
            if key in known or f"title:{_norm_title(ref.title)}" in known or excluded(ref):
                continue
            if is_systematic_review(ref) or not (ref.pmid or ref.doi):
                continue
            cited_by[key].add(review_ref)
            first.setdefault(key, ref)
    ranked = sorted(
        first,
        key=lambda k: (len(cited_by[k]), bool(_TRIAL_RE.search(first[k].title or ""))),
        reverse=True,
    )
    chosen = [k for k in ranked if _TRIAL_RE.search(first[k].title or "") or len(cited_by[k]) >= 2][:limit]
    return [first[k] for k in chosen], {k: cited_by[k] for k in chosen}


def _record_to_study(record: dict) -> EvidenceStudy:
    pub_types = (record.get("pubTypeList") or {}).get("pubType") or []
    if isinstance(pub_types, str):
        pub_types = [pub_types]
    journal = ((record.get("journalInfo") or {}).get("journal") or {}).get("title")
    return EvidenceStudy(
        source="review_reference", source_id=str(record.get("id") or record.get("pmid") or ""),
        title=record.get("title") or "", abstract=record.get("abstractText"),
        publication_year=str(record.get("pubYear") or "") or None, doi=record.get("doi"),
        pmid=record.get("pmid"), pmcid=record.get("pmcid"), journal=journal,
        publication_types=list(pub_types), sources=["review_reference"],
    )


async def fetch_records(candidates: list[EvidenceStudy], batch: int = 20) -> list[EvidenceStudy]:
    """Europe PMC core records (abstract, publication types) for candidates, by PMID or DOI."""
    found: list[EvidenceStudy] = []
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for i in range(0, len(candidates), batch):
            chunk = candidates[i:i + batch]
            terms = [f"(EXT_ID:{c.pmid} AND SRC:MED)" if c.pmid else f'DOI:"{_norm_doi(c.doi)}"' for c in chunk]
            try:
                response = await client.get(EUROPE_PMC_SEARCH, params={
                    "query": " OR ".join(terms), "format": "json", "resultType": "core", "pageSize": batch,
                })
                response.raise_for_status()
                records = (response.json().get("resultList") or {}).get("result") or []
            except Exception:  # noqa: BLE001 - seeding is optional
                continue
            found.extend(_record_to_study(r) for r in records if r.get("title"))
    return found


def is_trial(study: EvidenceStudy) -> bool:
    types = " ".join(study.publication_types).lower()
    if "randomized controlled trial" in types or "clinical trial" in types:
        return True
    return bool(_TRIAL_RE.search(f"{study.title} {(study.abstract or '')[:600]}"))


def to_scored(studies: list[EvidenceStudy], start_ref: int) -> list[ScoredStudy]:
    scored = []
    for offset, study in enumerate(studies):
        scored.append(ScoredStudy(
            **study.model_dump(exclude={"evidence_level"}),
            evidence_level=infer_evidence_level(study.title, study.publication_types),
            evidence_level_score=0.0, citation_score=0.0, recency_score=0.0, composite_score=0.0,
            reference_number=start_ref + offset,
        ))
    return scored
