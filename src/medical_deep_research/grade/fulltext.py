"""Node B5 — open-access full text for the studies that will be extracted.

Risk-of-bias, indirectness and outcome data are rarely judgeable from abstracts. Texts already read by
the search agent are reused; for the rest, PMIDs are converted to PMCIDs and the Europe PMC JATS XML is
fetched. Publisher PDFs (uploaded by the user or found by Unpaywall) arrive through the agent's
fetch_fulltext/parse_pdf tools and the shared cache.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from ..research.fulltext import fetch_europe_pmc_fulltext_xml
from ..research.models import ScoredStudy
from ..research.search import POLITE_EMAIL, USER_AGENT

PMC_ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

XmlFetcher = Callable[[str], Awaitable[str | None]]


async def pmids_to_pmcids(pmids: list[str]) -> dict[str, str]:
    pmids = [p for p in dict.fromkeys(pmids) if p]
    if not pmids:
        return {}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
            response = await client.get(
                PMC_ID_CONVERTER_URL,
                params={"ids": ",".join(pmids), "format": "json", "tool": "medical-deep-research",
                        "email": POLITE_EMAIL},
            )
            response.raise_for_status()
            records = response.json().get("records") or []
    except Exception:  # noqa: BLE001 - optional enrichment
        return {}
    return {
        str(r.get("pmid")): str(r.get("pmcid"))
        for r in records
        if r.get("pmid") and r.get("pmcid")
    }


async def fetch_missing_fulltexts(
    studies: list[ScoredStudy],
    known: dict[int, str],
    *,
    fetch_xml: XmlFetcher = fetch_europe_pmc_fulltext_xml,
    resolve_pmcids: Callable[[list[str]], Awaitable[dict[str, str]]] = pmids_to_pmcids,
    concurrency: int = 4,
) -> dict[int, str]:
    """Return {reference_number: text} for studies without a known full text that have an open PMC copy."""
    missing = [s for s in studies if int(s.reference_number or 0) not in known]
    pmcids = await resolve_pmcids([s.pmid for s in missing if s.pmid and not s.pmcid])
    semaphore = asyncio.Semaphore(concurrency)
    found: dict[int, str] = {}

    async def one(study: ScoredStudy) -> None:
        pmcid = study.pmcid or (pmcids.get(study.pmid) if study.pmid else None)
        if not pmcid:
            return
        async with semaphore:
            try:
                text = await fetch_xml(pmcid)
            except Exception:  # noqa: BLE001
                text = None
        if text and len(text) > 1500:
            found[int(study.reference_number or 0)] = text

    await asyncio.gather(*(one(s) for s in missing))
    return found
