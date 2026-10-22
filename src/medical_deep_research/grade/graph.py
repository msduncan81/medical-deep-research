"""GRADE pipeline as an explicit LangGraph state graph — stage 1: outcomes, review map, extraction.

    propose_outcomes (A2) -> confirm_outcomes (A2, user) -> map_reviews (B4) -> seed_from_reviews (B4b)
        -> fetch_fulltext (B5) -> extract (B6) -> build_bodies (C0) -> assess (C1-C6) -> END

Later stages add the per-outcome assessment (bodies of evidence by design, the five domains, rating
up, certainty) and the outputs (evidence profile, Summary of Findings, report). Each node receives only
the data it needs; numbers and rules live in code, judgements are model calls with structured output.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from ..models import RunRequest
from ..research.models import ScoredStudy
from .extraction import extract_all, studies_to_extract
from .fulltext import fetch_missing_fulltexts
from .assessment import OutcomeProfile, assess_outcomes
from .models import (
    ComparisonSpec,
    GradeResult,
    GradeStage1Result,
    OutcomeProposal,
    OutcomeSpec,
    ReviewMap,
    StudyExtraction,
)
from .synthesis import EvidenceBody, build_bodies
from .outcomes import StructuredCall, confirm_outcomes, normalize_comparison, normalize_outcomes, propose_outcomes
from .reviews import ReferenceFetcher, map_reviews
from .seed import _key as seed_key
from .seed import fetch_records, is_trial, select_seed_candidates, to_scored


class GradeState(TypedDict, total=False):
    request: RunRequest
    studies: list[ScoredStudy]
    proposed_outcomes: list[OutcomeSpec]
    proposed_comparison: ComparisonSpec
    outcomes: list[OutcomeSpec]
    comparison: ComparisonSpec
    outcomes_confirmed_by_user: bool
    review_map: ReviewMap
    references_by_review: dict[int, list[Any]]
    seeded: list[ScoredStudy]
    covered_by: dict[int, set[int]]
    fulltexts: dict[int, str]
    extractions: list[StudyExtraction]
    bodies: list[EvidenceBody]
    profiles: list[OutcomeProfile]
    notes: list[str]


Progress = Callable[[str, str], Awaitable[None]]  # (node, message)


async def _noop(_node: str, _message: str) -> None:
    return None


FullTextFetcher = Callable[[list[ScoredStudy], dict[int, str]], Awaitable[dict[int, str]]]
SeedFetcher = Callable[[list[Any]], Awaitable[list[Any]]]


def build_stage1_graph(
    structured_call: StructuredCall,
    fetch_references: ReferenceFetcher,
    fulltext_for: Callable[[ScoredStudy], str | None],
    progress: Progress = _noop,
    fetch_full: FullTextFetcher | None = fetch_missing_fulltexts,
    with_assessment: bool = False,
    seed_records: SeedFetcher | None = fetch_records,
) -> Any:
    from langgraph.graph import END, START, StateGraph

    async def node_propose(state: GradeState) -> dict[str, Any]:
        await progress("propose_outcomes", "Proposing outcomes and certainty targets")
        request = state["request"]
        if request.grade_outcomes_override:
            proposal = OutcomeProposal(
                outcomes=[OutcomeSpec.model_validate(o) for o in request.grade_outcomes_override],
                comparison=ComparisonSpec.model_validate(request.grade_comparison_override or {}),
            )
            return {"proposed_outcomes": normalize_outcomes(proposal),
                    "proposed_comparison": normalize_comparison(proposal.comparison)}
        outcomes, comparison = await propose_outcomes(structured_call, request, state["studies"])
        return {"proposed_outcomes": outcomes, "proposed_comparison": comparison}

    async def node_confirm(state: GradeState) -> dict[str, Any]:
        await progress("confirm_outcomes", "Waiting for the user to confirm the outcomes")
        if state["request"].grade_outcomes_override:
            return {"outcomes": state["proposed_outcomes"], "comparison": state.get("proposed_comparison"),
                    "outcomes_confirmed_by_user": False,
                    "notes": [*state.get("notes", []), "Outcomes fixed in advance (reference set)."]}
        outcomes, comparison, confirmed, note = await confirm_outcomes(
            state["request"], state["proposed_outcomes"], state.get("proposed_comparison")
        )
        return {"outcomes": outcomes, "comparison": comparison, "outcomes_confirmed_by_user": confirmed,
                "notes": [*state.get("notes", []), note]}

    async def node_map_reviews(state: GradeState) -> dict[str, Any]:
        await progress("map_reviews", "Mapping systematic reviews to primary studies")
        review_map, references = await map_reviews(state["studies"], fetch_references)
        covered: dict[int, set[int]] = {}
        for review in review_map.reviews:
            if review.chosen:
                for ref in review.covered_refs:
                    covered.setdefault(ref, set()).add(review.review_ref)
        return {"review_map": review_map, "references_by_review": references, "covered_by": covered,
                "notes": [*state.get("notes", []), *review_map.notes]}

    async def node_seed(state: GradeState) -> dict[str, Any]:
        references = state.get("references_by_review") or {}
        if not references or seed_records is None:
            return {"seeded": []}
        await progress("seed_from_reviews", "Looking for trials cited by the included reviews")
        request = state["request"]
        from ..agentic_tools import is_excluded

        candidates, cited_by = select_seed_candidates(references, state["studies"],
                                                      lambda s: is_excluded(s, request))
        records = [r for r in await seed_records(candidates) if is_trial(r) and not is_excluded(r, request)]
        start = max((int(s.reference_number or 0) for s in state["studies"]), default=0) + 1
        seeded = to_scored(records, max(start, 101))
        covered = {k: set(v) for k, v in (state.get("covered_by") or {}).items()}
        chosen = {r.review_ref for r in state["review_map"].reviews if r.chosen}
        for study in seeded:
            reviews = cited_by.get(seed_key(study)) or set()
            if reviews & chosen:
                covered[int(study.reference_number or 0)] = reviews & chosen
        note = f"{len(seeded)} trials added from the reference lists of included reviews."
        return {"seeded": seeded, "studies": [*state["studies"], *seeded], "covered_by": covered,
                "notes": [*state.get("notes", []), note]}

    async def node_fetch_fulltext(state: GradeState) -> dict[str, Any]:
        selected, _ = studies_to_extract(state["studies"], state["review_map"])
        known = {int(st.reference_number or 0): text for st in selected if (text := fulltext_for(st))}
        await progress("fetch_fulltext", f"Looking for open full text ({len(known)} of {len(selected)} already read)")
        found = await fetch_full(selected, known) if fetch_full else {}
        notes = [*state.get("notes", []),
                 f"Full text available for {len(known) + len(found)} of {len(selected)} extracted studies."]
        return {"fulltexts": {**known, **found}, "notes": notes}

    async def node_extract(state: GradeState) -> dict[str, Any]:
        selected, skip_notes = studies_to_extract(state["studies"], state["review_map"])
        texts = state.get("fulltexts", {})

        async def on_progress(study: ScoredStudy) -> None:
            await progress("extract", f"Extracting outcome data from [{study.reference_number}]")

        extractions, fail_notes = await extract_all(
            structured_call, selected, state["outcomes"],
            lambda st: texts.get(int(st.reference_number or 0)), on_progress=on_progress,
            comparison=state.get("comparison"),
        )
        return {"extractions": extractions, "notes": [*state.get("notes", []), *skip_notes, *fail_notes]}

    async def node_build_bodies(state: GradeState) -> dict[str, Any]:
        await progress("build_bodies", "Assembling bodies of evidence per outcome and design")
        comparison = state.get("comparison")
        return {"bodies": build_bodies(state["outcomes"], state.get("extractions", []),
                                       comparison.subgroups if comparison else [],
                                       covered_by=state.get("covered_by"))}

    async def node_assess(state: GradeState) -> dict[str, Any]:
        bodies = state.get("bodies", [])
        await progress("assess", f"Rating certainty for {len(bodies)} bodies of evidence")
        profiles = await assess_outcomes(structured_call, state["request"], state["outcomes"], bodies,
                                         comparison=state.get("comparison"))
        return {"profiles": profiles}

    graph = StateGraph(GradeState)
    graph.add_node("propose_outcomes", node_propose)
    graph.add_node("confirm_outcomes", node_confirm)
    graph.add_node("map_reviews", node_map_reviews)
    graph.add_node("seed_from_reviews", node_seed)
    graph.add_node("fetch_fulltext", node_fetch_fulltext)
    graph.add_node("extract", node_extract)
    if with_assessment:
        graph.add_node("build_bodies", node_build_bodies)
        graph.add_node("assess", node_assess)
    graph.add_edge(START, "propose_outcomes")
    graph.add_edge("propose_outcomes", "confirm_outcomes")
    graph.add_edge("confirm_outcomes", "map_reviews")
    graph.add_edge("map_reviews", "seed_from_reviews")
    graph.add_edge("seed_from_reviews", "fetch_fulltext")
    graph.add_edge("fetch_fulltext", "extract")
    if with_assessment:
        graph.add_edge("extract", "build_bodies")
        graph.add_edge("build_bodies", "assess")
        graph.add_edge("assess", END)
    else:
        graph.add_edge("extract", END)
    return graph.compile()


async def run_stage1(
    request: RunRequest,
    studies: list[ScoredStudy],
    structured_call: StructuredCall,
    fetch_references: ReferenceFetcher,
    fulltext_for: Callable[[ScoredStudy], str | None],
    progress: Progress = _noop,
    fetch_full: FullTextFetcher | None = fetch_missing_fulltexts,
    seed_records: SeedFetcher | None = fetch_records,
) -> GradeStage1Result:
    graph = build_stage1_graph(structured_call, fetch_references, fulltext_for, progress, fetch_full,
                               seed_records=seed_records)
    final: GradeState = await graph.ainvoke({"request": request, "studies": studies, "notes": []})
    return GradeStage1Result(
        comparison=final.get("comparison") or ComparisonSpec(),
        seeded_studies=final.get("seeded", []),
        outcomes=final.get("outcomes", []),
        outcomes_confirmed_by_user=bool(final.get("outcomes_confirmed_by_user")),
        review_map=final.get("review_map") or ReviewMap(),
        extractions=final.get("extractions", []),
        notes=final.get("notes", []),
    )


async def run_grade(
    request: RunRequest,
    studies: list[ScoredStudy],
    structured_call: StructuredCall,
    fetch_references: ReferenceFetcher,
    fulltext_for: Callable[[ScoredStudy], str | None],
    progress: Progress = _noop,
    fetch_full: FullTextFetcher | None = fetch_missing_fulltexts,
    seed_records: SeedFetcher | None = fetch_records,
) -> GradeResult:
    """Stages 1 and 2: outcomes, review map, extraction, bodies of evidence and certainty per outcome."""
    graph = build_stage1_graph(structured_call, fetch_references, fulltext_for, progress, fetch_full,
                               with_assessment=True, seed_records=seed_records)
    final: GradeState = await graph.ainvoke({"request": request, "studies": studies, "notes": []})
    return GradeResult(
        comparison=final.get("comparison") or ComparisonSpec(),
        seeded_studies=final.get("seeded", []),
        outcomes=final.get("outcomes", []),
        outcomes_confirmed_by_user=bool(final.get("outcomes_confirmed_by_user")),
        review_map=final.get("review_map") or ReviewMap(),
        extractions=final.get("extractions", []),
        notes=final.get("notes", []),
        profiles=[
            {**p.model_dump(mode="json"), "label_a": p.label_a, "label_b": p.label_b}
            for p in final.get("profiles", [])
        ],
    )
