"""GRADE pipeline as an explicit LangGraph state graph — stage 1: outcomes, review map, extraction.

    propose_outcomes (A2) -> confirm_outcomes (A2, user) -> map_reviews (B4) -> extract (B6) -> END

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
from .models import GradeStage1Result, OutcomeSpec, ReviewMap, StudyExtraction
from .outcomes import StructuredCall, confirm_outcomes, propose_outcomes
from .reviews import ReferenceFetcher, map_reviews


class GradeState(TypedDict, total=False):
    request: RunRequest
    studies: list[ScoredStudy]
    proposed_outcomes: list[OutcomeSpec]
    outcomes: list[OutcomeSpec]
    outcomes_confirmed_by_user: bool
    review_map: ReviewMap
    extractions: list[StudyExtraction]
    notes: list[str]


Progress = Callable[[str, str], Awaitable[None]]  # (node, message)


async def _noop(_node: str, _message: str) -> None:
    return None


def build_stage1_graph(
    structured_call: StructuredCall,
    fetch_references: ReferenceFetcher,
    fulltext_for: Callable[[ScoredStudy], str | None],
    progress: Progress = _noop,
) -> Any:
    from langgraph.graph import END, START, StateGraph

    async def node_propose(state: GradeState) -> dict[str, Any]:
        await progress("propose_outcomes", "Proposing outcomes and certainty targets")
        outcomes = await propose_outcomes(structured_call, state["request"], state["studies"])
        return {"proposed_outcomes": outcomes}

    async def node_confirm(state: GradeState) -> dict[str, Any]:
        await progress("confirm_outcomes", "Waiting for the user to confirm the outcomes")
        outcomes, confirmed, note = await confirm_outcomes(state["request"], state["proposed_outcomes"])
        return {"outcomes": outcomes, "outcomes_confirmed_by_user": confirmed,
                "notes": [*state.get("notes", []), note]}

    async def node_map_reviews(state: GradeState) -> dict[str, Any]:
        await progress("map_reviews", "Mapping systematic reviews to primary studies")
        review_map = await map_reviews(state["studies"], fetch_references)
        return {"review_map": review_map, "notes": [*state.get("notes", []), *review_map.notes]}

    async def node_extract(state: GradeState) -> dict[str, Any]:
        selected, skip_notes = studies_to_extract(state["studies"], state["review_map"])

        async def on_progress(study: ScoredStudy) -> None:
            await progress("extract", f"Extracting outcome data from [{study.reference_number}]")

        extractions, fail_notes = await extract_all(
            structured_call, selected, state["outcomes"], fulltext_for, on_progress=on_progress
        )
        return {"extractions": extractions, "notes": [*state.get("notes", []), *skip_notes, *fail_notes]}

    graph = StateGraph(GradeState)
    graph.add_node("propose_outcomes", node_propose)
    graph.add_node("confirm_outcomes", node_confirm)
    graph.add_node("map_reviews", node_map_reviews)
    graph.add_node("extract", node_extract)
    graph.add_edge(START, "propose_outcomes")
    graph.add_edge("propose_outcomes", "confirm_outcomes")
    graph.add_edge("confirm_outcomes", "map_reviews")
    graph.add_edge("map_reviews", "extract")
    graph.add_edge("extract", END)
    return graph.compile()


async def run_stage1(
    request: RunRequest,
    studies: list[ScoredStudy],
    structured_call: StructuredCall,
    fetch_references: ReferenceFetcher,
    fulltext_for: Callable[[ScoredStudy], str | None],
    progress: Progress = _noop,
) -> GradeStage1Result:
    graph = build_stage1_graph(structured_call, fetch_references, fulltext_for, progress)
    final: GradeState = await graph.ainvoke({"request": request, "studies": studies, "notes": []})
    return GradeStage1Result(
        outcomes=final.get("outcomes", []),
        outcomes_confirmed_by_user=bool(final.get("outcomes_confirmed_by_user")),
        review_map=final.get("review_map") or ReviewMap(),
        extractions=final.get("extractions", []),
        notes=final.get("notes", []),
    )
