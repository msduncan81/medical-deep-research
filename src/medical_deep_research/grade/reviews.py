"""Node B4 — map systematic reviews to the primary studies of the pool, to avoid double counting.

Cochrane Handbook ch. V (overviews): map primary studies across reviews with a citation matrix and the
corrected covered area (CCA); counting the same study twice gives it too much weight. When reviews
overlap, choose one per group by pre-specified criteria (here: most recent, then widest coverage of the
pool). Review quality (AMSTAR 2 / ROBIS) is a later stage.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from ..research.models import EvidenceStudy, ScoredStudy, SearchProviderResult
from .models import ReviewCoverage, ReviewMap

_SR_TITLE_RE = re.compile(r"systematic\s*review|meta[\s-]?analys[ie]s|cochrane", re.IGNORECASE)
_SR_TYPES = {"systematic review", "meta analysis", "meta-analysis", "network meta-analysis"}

ReferenceFetcher = Callable[[EvidenceStudy], Awaitable[SearchProviderResult]]


def is_systematic_review(study: EvidenceStudy) -> bool:
    types = {t.lower() for t in (study.publication_types or [])}
    if types & _SR_TYPES:
        return True
    if _SR_TITLE_RE.search(study.title or ""):
        return True
    return "cochrane" in (study.source or "").lower() or "cochrane" in (study.journal or "").lower()


_NMA_RE = re.compile(r"network\s+meta[\s-]?analys", re.IGNORECASE)


def is_network_meta_analysis(study: EvidenceStudy) -> bool:
    """Core GRADE covers pairwise comparisons; network meta-analyses are out of scope (Core GRADE 1)."""
    return bool(_NMA_RE.search(f"{study.title or ''} {study.abstract or ''}"))


def _norm_doi(doi: str | None) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (doi or "").strip().lower())


def _norm_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())[:80]


def _year(study: EvidenceStudy) -> int | None:
    match = re.search(r"\d{4}", str(study.publication_year or ""))
    return int(match.group(0)) if match else None


def match_references(references: list[EvidenceStudy], primaries: list[ScoredStudy]) -> list[int]:
    """Reference numbers of the pool's primary studies that appear in a review's reference list."""
    by_pmid = {r.pmid for r in references if r.pmid}
    by_doi = {_norm_doi(r.doi) for r in references if r.doi}
    by_title = {_norm_title(r.title) for r in references if r.title and len(_norm_title(r.title)) >= 20}
    covered = []
    for study in primaries:
        if (
            (study.pmid and study.pmid in by_pmid)
            or (study.doi and _norm_doi(study.doi) in by_doi)
            or (_norm_title(study.title) in by_title and len(_norm_title(study.title)) >= 20)
        ):
            covered.append(int(study.reference_number or 0))
    return sorted(ref for ref in covered if ref)


def _overlap_groups(reviews: list[ReviewCoverage]) -> list[list[int]]:
    parent = {r.review_ref: r.review_ref for r in reviews}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(reviews):
        for b in reviews[i + 1:]:
            if set(a.covered_refs) & set(b.covered_refs):
                parent[find(a.review_ref)] = find(b.review_ref)
    groups: dict[int, list[int]] = {}
    for review in reviews:
        groups.setdefault(find(review.review_ref), []).append(review.review_ref)
    return [sorted(g) for g in groups.values()]


def corrected_covered_area(reviews: list[ReviewCoverage]) -> float | None:
    """CCA = (N - r) / (r*c - r): N inclusions, r unique primary studies, c reviews (Pieper 2014)."""
    c = len(reviews)
    rows = {ref for review in reviews for ref in review.covered_refs}
    r = len(rows)
    if c < 2 or r == 0:
        return None
    n = sum(len(review.covered_refs) for review in reviews)
    return round((n - r) / (r * c - r), 3)


def build_review_map(reviews: list[ReviewCoverage]) -> ReviewMap:
    by_ref = {r.review_ref: r for r in reviews}
    groups = _overlap_groups(reviews)
    covered_by_chosen: dict[int, int] = {}
    notes: list[str] = []
    for group in groups:
        members = [by_ref[ref] for ref in group]
        pairwise = [r for r in members if not r.network]
        for review in members:
            if review.network:
                review.reason = "network meta-analysis (outside Core GRADE); its trials are assessed directly"
        if not pairwise:
            continue
        chosen = max(pairwise, key=lambda r: (r.year or 0, len(r.covered_refs)))
        chosen.chosen = True
        chosen.reason = (
            "only review in its group"
            if len(members) == 1
            else f"most recent / widest coverage among overlapping reviews {group}"
        )
        for other in members:
            if other is not chosen and not other.network:
                other.reason = f"overlaps with chosen review [{chosen.review_ref}]"
        for ref in chosen.covered_refs:
            covered_by_chosen.setdefault(ref, chosen.review_ref)
    if any(r.reference_count == 0 for r in reviews):
        notes.append("Some reviews had no retrievable reference list; their overlap could not be mapped.")
    return ReviewMap(
        reviews=sorted(reviews, key=lambda r: r.review_ref),
        groups=groups,
        covered_by_chosen=covered_by_chosen,
        corrected_covered_area=corrected_covered_area(reviews),
        notes=notes,
    )


async def map_reviews(
    studies: list[ScoredStudy],
    fetch_references: ReferenceFetcher,
    *,
    concurrency: int = 3,
) -> tuple[ReviewMap, dict[int, list[EvidenceStudy]]]:
    """Return the review map and the reference list fetched for each review (by its reference number)."""
    reviews_in_pool = [s for s in studies if is_systematic_review(s)]
    primaries = [s for s in studies if not is_systematic_review(s)]
    if not reviews_in_pool:
        return ReviewMap(notes=["No systematic reviews among the included studies."]), {}
    semaphore = asyncio.Semaphore(concurrency)
    references_by_review: dict[int, list[EvidenceStudy]] = {}

    async def coverage(review: ScoredStudy) -> ReviewCoverage:
        async with semaphore:
            try:
                result = await fetch_references(review)
                references = list(result.studies)
            except Exception:  # noqa: BLE001 - a missing reference list must not stop the graph
                references = []
        references_by_review[int(review.reference_number or 0)] = references
        return ReviewCoverage(
            review_ref=int(review.reference_number or 0),
            title=review.title or "",
            year=_year(review),
            covered_refs=match_references(references, primaries),
            reference_count=len(references),
            network=is_network_meta_analysis(review),
        )

    coverages = await asyncio.gather(*(coverage(r) for r in reviews_in_pool))
    return build_review_map(list(coverages)), references_by_review
