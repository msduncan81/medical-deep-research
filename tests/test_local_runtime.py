from __future__ import annotations

import asyncio
import sys
import types
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from medical_deep_research.models import EventType, RunRequest
from medical_deep_research.research.models import EvidenceStudy, SearchProviderResult, VerificationSummary
from medical_deep_research.runtime import LangChainLocalRuntime


def fake_lc_tool(arg: object = None) -> object:
    def decorate(func: Callable[..., object], name: str | None = None) -> Callable[..., object]:
        setattr(func, "name", name or func.__name__)
        return func

    if callable(arg):
        return decorate(arg)
    if isinstance(arg, str):
        return lambda func: decorate(func, arg)
    return lambda func: decorate(func)


def _tool_name(tool: object) -> str:
    return str(getattr(tool, "name", getattr(tool, "__name__", "tool")))


async def call_fake_tool(tools: list[object], tool_name: str, **kwargs: object) -> object:
    for tool in tools:
        if _tool_name(tool) == tool_name:
            return await tool(**kwargs)  # type: ignore[misc]
    raise AssertionError(f"Tool not found: {tool_name}")


class AlwaysAvailableLocalRuntime(LangChainLocalRuntime):
    @property
    def sdk_available(self) -> bool:
        return True


invocations: list[dict[str, object]] = []
chat_models: list[object] = []
agent_kwargs: list[dict[str, object]] = []


async def fake_context_tokens(_base_url: str, _api_key: str) -> int:
    return 65536


@contextmanager
def fake_langgraph_agent(agent_impl: Callable[[list[object], dict[str, object]], Any]) -> object:
    core_tools_module = types.ModuleType("langchain_core.tools")
    core_messages_module = types.ModuleType("langchain_core.messages")
    langchain_core_module = types.ModuleType("langchain_core")
    langchain_openai_module = types.ModuleType("langchain_openai")
    langgraph_module = types.ModuleType("langgraph")
    langgraph_prebuilt_module = types.ModuleType("langgraph.prebuilt")

    class FakeAgent:
        def __init__(self, tools: list[object]) -> None:
            self.tools = tools

        async def ainvoke(self, inputs: dict[str, object], config: dict[str, object] | None = None) -> object:
            invocations.append({"inputs": inputs, "config": config})
            result = agent_impl(self.tools, inputs)
            if hasattr(result, "__await__"):
                return await result
            return result

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            chat_models.append(self)

    class FakeMessage:
        def __init__(self, content: object = None, **_kwargs: object) -> None:
            self.content = content

    def create_react_agent(_llm: object, tools: list[object], prompt: object, **kwargs: object) -> FakeAgent:
        assert prompt
        agent_kwargs.append(kwargs)
        return FakeAgent(tools)

    core_tools_module.tool = fake_lc_tool
    core_messages_module.SystemMessage = FakeMessage
    core_messages_module.HumanMessage = FakeMessage
    langchain_openai_module.ChatOpenAI = FakeChatOpenAI
    langgraph_prebuilt_module.create_react_agent = create_react_agent

    previous = {
        name: sys.modules.get(name)
        for name in (
            "langchain_core",
            "langchain_core.messages",
            "langchain_core.tools",
            "langchain_openai",
            "langgraph",
            "langgraph.prebuilt",
        )
    }
    sys.modules["langchain_core"] = langchain_core_module
    sys.modules["langchain_core.messages"] = core_messages_module
    sys.modules["langchain_core.tools"] = core_tools_module
    sys.modules["langchain_openai"] = langchain_openai_module
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.prebuilt"] = langgraph_prebuilt_module
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _assert_progress_monotonic(events: list[Any]) -> None:
    """Progress must never decrease, and a completed run must end at 100."""
    previous = 0
    for event in events:
        assert event.progress >= previous, f"progress went backwards: {previous} -> {event.progress}"
        previous = event.progress
    completed = [e for e in events if e.event_type == EventType.RUN_COMPLETED]
    if completed:
        assert completed[-1].progress == 100, f"final progress {completed[-1].progress} != 100"


async def fake_search_source(source: str, query: str, **_kwargs: object) -> SearchProviderResult:
    studies = []
    if source == "PubMed":
        studies = [
            EvidenceStudy(
                source="PubMed",
                source_id="12345678",
                title="Erector spinae plane block after cardiac surgery randomized trial",
                abstract="A randomized trial evaluating ESPB after cardiac surgery reported reduced pain scores.",
                journal="Journal of Clinical Anesthesia",
                publication_year="2024",
                pmid="12345678",
                citation_count=12,
                evidence_level="Level II",
                publication_types=["Randomized Controlled Trial"],
                sources=["PubMed"],
            )
        ]
    return SearchProviderResult(source=source, query=query, studies=studies)


_LEVELS = ["Level I", "Level II", "Level III", "Level IV", "Level V"]


def _make_study(i: int, source: str, level: str) -> EvidenceStudy:
    return EvidenceStudy(
        source=source,
        source_id=f"{source}-{i}",
        title=f"{source} study {i} on ESPB after cardiac surgery",
        abstract=f"Abstract {i} reporting postoperative pain outcomes for ESPB versus PCA.",
        journal="Journal of Test Anesthesia",
        publication_year="2024",
        pmid=f"PM{i:06d}" if source == "PubMed" else None,
        doi=None if source == "PubMed" else f"10.1000/{source.lower()}-{i}",
        citation_count=50 - i,
        evidence_level=level,
        sources=[source],
    )


async def fake_multi_search_source(source: str, query: str, **_kwargs: object) -> SearchProviderResult:
    """Return a large, evidence-level-varied pool so paging/filtering can be exercised."""
    if source == "PubMed":
        studies = [_make_study(i, "PubMed", _LEVELS[i % 5]) for i in range(20)]
    elif source == "OpenAlex":
        studies = [_make_study(i, "OpenAlex", "Level III") for i in range(5)]
    else:
        studies = []
    return SearchProviderResult(source=source, query=query, studies=studies)


async def fake_verify_studies(studies: list[object], **_kwargs: object) -> VerificationSummary:
    return VerificationSummary(
        total_considered=len(studies),
        verified_pmids=len(studies),
        missing_pmids=0,
        missing_from_pubmed=0,
        notes=["fake verification"],
    )


def make_request() -> RunRequest:
    return RunRequest(
        run_id="test-local-run",
        query="Population: cardiac surgery; Intervention: ESPB; Comparison: PCA; Outcome: Pain score",
        query_type="pico",
        mode="detailed",
        provider="local",
        model="qwen3.6-27b",
        language="en",
        api_keys={"local_base_url": "http://127.0.0.1:11434"},
        offline_mode=False,
    )


def make_valid_report() -> str:
    synthesis = " ".join(
        "The ranked trial is directly relevant to the PICO question and reports postoperative pain outcomes [1]."
        for _ in range(90)
    )
    return f"""# Research Report

## 1. Executive Summary
ESPB after cardiac surgery was evaluated against PCA in the searched evidence [1].

## 2. Background
Postoperative pain after cardiac surgery affects mobilization and pulmonary recovery.

## 3. Methods
The agent searched PubMed and screened the retrieved study for population, intervention, comparator, and outcome alignment.

## 4. Results/Findings
{synthesis}

## 5. Discussion
The evidence should be interpreted cautiously because the available ranked set is small [1].
The overall GRADE certainty of evidence for the pain outcome is Low, rated down for imprecision [1].

## 6. Conclusions
ESPB may be a useful analgesic adjunct after cardiac surgery.

## 7. References

[1] Erector spinae plane block after cardiac surgery randomized trial. Journal of Clinical Anesthesia. 2024. PMID: 12345678.
"""


class LocalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def collect_events(self, agent_impl: Callable[[list[object], dict[str, object]], Any]) -> list[object]:
        with (
            fake_langgraph_agent(agent_impl),
            patch("medical_deep_research.runtime.search_source", fake_search_source),
            patch("medical_deep_research.runtime.verify_studies", fake_verify_studies),
            patch("medical_deep_research.agentic_tools.search_source", fake_search_source),
            patch("medical_deep_research.agentic_tools.verify_studies", fake_verify_studies),
            patch("medical_deep_research.runtime._local_server_context_tokens", fake_context_tokens),
        ):
            events = [event async for event in AlwaysAvailableLocalRuntime().stream_run(make_request())]
        _assert_progress_monotonic(events)
        return events

    async def test_local_runtime_stops_after_accepted_submit_report(self) -> None:
        report = make_valid_report().strip()

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            await call_fake_tool(tools, "submit_report", report_markdown=report)
            await asyncio.sleep(1)
            return {"messages": []}

        events = await self.collect_events(agent)
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertFalse(final.extra["had_error"])
        self.assertEqual(final.extra["report_source"], "submitted_report")
        self.assertEqual(final.report_markdown, report)

    def _rejected_report(self) -> str:
        return make_valid_report().replace(
            "available ranked set is small [1].",
            "available ranked set is small [1] [12].",
        )

    async def test_local_runtime_allows_rewrite_after_first_rejected_submit_report(self) -> None:
        rejected = self._rejected_report()
        report = make_valid_report().strip()

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            await call_fake_tool(tools, "submit_report", report_markdown=rejected)
            await call_fake_tool(tools, "submit_report", report_markdown=report)
            await asyncio.sleep(1)
            return {"messages": []}

        events = await self.collect_events(agent)
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertFalse(final.extra["had_error"])
        self.assertEqual(final.extra["report_source"], "submitted_report")
        self.assertEqual(final.report_markdown, report)

    async def test_local_runtime_recovers_after_repeated_rejected_submit_report(self) -> None:
        rejected = self._rejected_report()

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            await call_fake_tool(tools, "submit_report", report_markdown=rejected)
            await call_fake_tool(tools, "submit_report", report_markdown=rejected)
            await asyncio.sleep(1)
            return {"messages": []}

        events = await self.collect_events(agent)
        submit_results = [
            event for event in events
            if event.event_type == EventType.TOOL_RESULT and event.tool_name == "submit_report"
        ]
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertEqual(len(submit_results), 2)
        self.assertIn("error", submit_results[-1].extra["full_tool_output"])
        self.assertFalse(final.extra["had_error"])
        self.assertEqual(final.extra["report_source"], "recovered_agentic_state")
        self.assertIn("## 4. Results/Findings", final.report_markdown or "")
        self.assertIn("Erector spinae plane block", final.report_markdown or "")

    async def test_local_runtime_never_presents_uncited_answer_without_search(self) -> None:
        class FakeAIMessage:
            content = "ESPB clearly reduces pain after cardiac surgery. " * 60

        async def agent(_tools: list[object], _inputs: dict[str, object]) -> object:
            return {"messages": [FakeAIMessage()]}

        events = await self.collect_events(agent)
        fallback = [event for event in events if (event.extra or {}).get("fallback_reason")]
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertTrue(fallback)
        self.assertIn("without calling any research tools", fallback[0].extra["fallback_reason"])
        self.assertNotEqual(final.extra.get("report_source"), "agent_result")
        self.assertNotIn("ESPB clearly reduces pain", final.report_markdown or "")

    async def test_local_runtime_falls_back_when_model_never_starts_tool_use(self) -> None:
        async def agent(_tools: list[object], _inputs: dict[str, object]) -> object:
            await asyncio.sleep(5)
            return {"messages": []}

        with patch.object(AlwaysAvailableLocalRuntime, "first_tool_timeout_seconds", 0.05):
            events = await self.collect_events(agent)
        fallback = [event for event in events if (event.extra or {}).get("fallback_reason")]

        self.assertTrue(fallback)
        self.assertIn("did not begin tool use", fallback[0].extra["fallback_reason"])

    async def test_local_runtime_passes_recursion_limit_and_request_timeout(self) -> None:
        report = make_valid_report().strip()

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            await call_fake_tool(tools, "submit_report", report_markdown=report)
            return {"messages": []}

        await self.collect_events(agent)

        self.assertEqual(invocations[-1]["config"], {"recursion_limit": LangChainLocalRuntime.recursion_limit})
        self.assertIn("pre_model_hook", agent_kwargs[-1])
        self.assertEqual(chat_models[-1].kwargs["timeout"], LangChainLocalRuntime.request_timeout_seconds)  # type: ignore[attr-defined]
        self.assertEqual(chat_models[-1].kwargs["max_retries"], LangChainLocalRuntime.max_retries)  # type: ignore[attr-defined]

    async def test_submitted_report_has_reasoning_blocks_removed(self) -> None:
        report = make_valid_report().strip()

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            await call_fake_tool(
                tools, "submit_report", report_markdown=f"<think>Plan the sections first.</think>\n{report}"
            )
            return {"messages": []}

        events = await self.collect_events(agent)
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertEqual(final.extra["report_source"], "submitted_report")
        self.assertNotIn("<think>", final.report_markdown or "")
        self.assertNotIn("Plan the sections first", final.report_markdown or "")


class UnsubmittedReportTests(unittest.IsolatedAsyncioTestCase):
    collect_events = LocalRuntimeTests.collect_events

    async def test_report_written_as_final_message_goes_through_submit_report(self) -> None:
        report = make_valid_report().strip()

        class FakeAIMessage:
            content = report

        async def agent(tools: list[object], _inputs: dict[str, object]) -> object:
            await call_fake_tool(tools, "search_pubmed", query="cardiac surgery ESPB PCA pain", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")
            await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="Most relevant RCT.")
            return {"messages": [FakeAIMessage()]}

        async def fake_synthesize(_request: object, bridge: object) -> dict[str, object]:
            bridge._intermediate["synthesize_report"] = {}  # type: ignore[attr-defined]
            return {}

        with patch("medical_deep_research.runtime.tool_synthesize_report", fake_synthesize):
            events = await self.collect_events(agent)
        final = [event for event in events if event.event_type == EventType.RUN_COMPLETED][-1]

        self.assertEqual(final.extra["report_source"], "submitted_report")
        self.assertIn("## 4. Results/Findings", final.report_markdown or "")


class ContextTrimTests(unittest.TestCase):
    def _history(self, tool_outputs: int, size: int) -> list[object]:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        messages: list[object] = [HumanMessage(content="question")]
        for i in range(tool_outputs):
            call_id = f"call_{i}"
            messages.append(AIMessage(content="", tool_calls=[{"id": call_id, "name": "browse_studies", "args": {"page": i}}]))
            messages.append(ToolMessage(content="x" * size, tool_call_id=call_id, name="browse_studies"))
        return messages

    def test_history_under_budget_is_unchanged(self) -> None:
        from medical_deep_research.runtime import trim_tool_outputs_for_context

        messages = self._history(3, 300)
        self.assertIs(trim_tool_outputs_for_context(messages, budget_tokens=100_000), messages)

    def test_old_tool_outputs_are_stubbed_and_pairing_is_kept(self) -> None:
        from medical_deep_research.runtime import _estimate_tokens, trim_tool_outputs_for_context

        messages = self._history(20, 9000)
        trimmed = trim_tool_outputs_for_context(messages, budget_tokens=25_000, keep_recent=6)

        self.assertEqual(len(trimmed), len(messages))
        self.assertLessEqual(_estimate_tokens(trimmed), 25_000)
        tools = [m for m in trimmed if getattr(m, "type", None) == "tool"]
        self.assertIn("omitted to fit the context window", tools[0].content)
        self.assertTrue(all(len(m.content) == 9000 for m in tools[-6:]))
        self.assertEqual([m.tool_call_id for m in tools], [f"call_{i}" for i in range(20)])
        self.assertEqual(len(messages[2].content), 9000)  # the original list is not modified


class BlockTrimTests(unittest.TestCase):
    _history = ContextTrimTests._history

    def test_trimming_happens_in_blocks_and_keeps_earlier_stubs(self) -> None:
        from medical_deep_research.runtime import _estimate_tokens, trim_tool_outputs_for_context

        stubbed: set[str] = set()
        first = trim_tool_outputs_for_context(
            self._history(20, 9000), budget_tokens=40_000, keep_recent=6, stubbed=stubbed, target_fraction=0.6
        )
        self.assertLessEqual(_estimate_tokens(first), 24_000)
        first_stubbed = set(stubbed)
        self.assertTrue(first_stubbed)

        # One more turn: still under budget, so nothing new is stubbed and earlier stubs are re-applied.
        second = trim_tool_outputs_for_context(
            self._history(21, 9000), budget_tokens=40_000, keep_recent=6, stubbed=stubbed, target_fraction=0.6
        )
        self.assertEqual(stubbed, first_stubbed)
        for a, b in zip(first, second):
            self.assertEqual(a.content, b.content)


class BrowseLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_browse_studies_is_capped_per_pool_and_reset_by_get_studies(self) -> None:
        from medical_deep_research.agentic_tools import (
            AgenticEventBridge,
            max_browse_calls_per_pool,
            tool_browse_studies,
            tool_get_studies,
        )

        bridge = AgenticEventBridge()
        bridge.search_results.append(await fake_multi_search_source("PubMed", "q"))
        request = make_request()
        await tool_get_studies(request, bridge, context="clinical")
        for _ in range(max_browse_calls_per_pool()):
            self.assertNotIn("error", await tool_browse_studies(request, bridge, page=1))
        blocked = await tool_browse_studies(request, bridge, page=2)
        self.assertIn("limit reached", blocked["error"])

        await tool_get_studies(request, bridge, context="clinical")
        self.assertNotIn("error", await tool_browse_studies(request, bridge, page=1))


class ReasoningAndTranslationHelperTests(unittest.TestCase):
    def test_strip_reasoning_blocks(self) -> None:
        from medical_deep_research.agentic_tools import strip_reasoning_blocks

        self.assertEqual(strip_reasoning_blocks("<think>internal</think>\n# Report"), "# Report")
        self.assertEqual(strip_reasoning_blocks("dangling reasoning</think># Report"), "# Report")
        self.assertEqual(strip_reasoning_blocks("# Report without reasoning"), "# Report without reasoning")

    def test_translation_looks_complete_rejects_truncated_translation(self) -> None:
        from medical_deep_research.agentic_tools import translation_looks_complete

        original = make_valid_report()
        truncated = original.split("## 4.")[0] + "x" * 300
        self.assertFalse(translation_looks_complete(original, truncated))
        self.assertTrue(translation_looks_complete(original, original))


class NewToolBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def _codex_pcc_request(self) -> RunRequest:
        return RunRequest(
            run_id="codex-pcc-ai-communication",
            query=(
                "Population: health professions learners, trainees, clinicians, educators; "
                "Concept: AI-supported education, training, simulation, coaching, assessment, feedback; "
                "Context: shared decision making education, communication training, patient-centered decision conversations"
            ),
            query_type="pcc",
            mode="detailed",
            provider="codex",
            model="gpt-5.4-mini",
            language="en",
            api_keys={},
            offline_mode=False,
            query_payload={
                "population": "health professions learners, trainees, clinicians, educators",
                "concept": "AI-supported education, training, simulation, coaching, assessment, feedback",
                "context": "shared decision making education, communication training, patient-centered decision conversations",
            },
        )

    def _build_tools(self) -> list[object]:
        from medical_deep_research.agentic_tools import AgenticEventBridge
        from medical_deep_research.runtime import _build_langchain_tools

        with fake_langgraph_agent(lambda _t, _i: {"messages": []}):
            return _build_langchain_tools(make_request(), AgenticEventBridge())

    async def _seed_pool(self, tools: list[object]) -> None:
        with (
            patch("medical_deep_research.agentic_tools.search_source", fake_search_source),
            patch("medical_deep_research.agentic_tools.verify_studies", fake_verify_studies),
        ):
            await call_fake_tool(tools, "search_pubmed", query="q", max_results=3)
            await call_fake_tool(tools, "get_studies", context="clinical")

    def _build_tools_with_bridge(self) -> tuple[list[object], Any]:
        from medical_deep_research.agentic_tools import AgenticEventBridge
        from medical_deep_research.runtime import _build_langchain_tools

        bridge = AgenticEventBridge()
        with fake_langgraph_agent(lambda _t, _i: {"messages": []}):
            return _build_langchain_tools(make_request(), bridge), bridge

    async def _seed_multi(self, tools: list[object], *, sources: tuple[str, ...] = ("PubMed",)) -> None:
        names = {"PubMed": "search_pubmed", "OpenAlex": "search_openalex"}
        with (
            patch("medical_deep_research.agentic_tools.search_source", fake_multi_search_source),
            patch("medical_deep_research.agentic_tools.verify_studies", fake_verify_studies),
        ):
            for src in sources:
                await call_fake_tool(tools, names[src], query="q", max_results=25)
            await call_fake_tool(tools, "get_studies", context="clinical")

    async def test_get_studies_returns_top_tier_with_facets(self) -> None:
        import json

        from medical_deep_research.agentic_tools import STUDY_PAGE_SIZE

        tools = self._build_tools_with_bridge()[0]
        await self._seed_multi(tools)
        result = json.loads(await call_fake_tool(tools, "get_studies", context="clinical"))  # type: ignore[arg-type]
        self.assertEqual(result["total"], 20)
        self.assertEqual(result["shown"], STUDY_PAGE_SIZE)
        self.assertTrue(result["has_more"])
        self.assertEqual(len(result["studies"]), STUDY_PAGE_SIZE)
        self.assertEqual(sum(result["counts_by_evidence_level"].values()), 20)

    async def test_codex_pcc_requires_core_source_attempts_before_triage(self) -> None:
        from medical_deep_research.agentic_tools import AgenticEventBridge, tool_get_studies

        request = self._codex_pcc_request()
        bridge = AgenticEventBridge()
        bridge.search_results.append(SearchProviderResult(source="PubMed", query="q", error="HTTP 502"))

        result = await tool_get_studies(request, bridge, context="clinical")

        self.assertIn("error", result)
        self.assertIn("PMC", result["missing_sources"])
        self.assertIn("OpenAlex", result["missing_sources"])
        self.assertIn("PubMed", result["attempted_sources"])

    async def test_codex_pcc_source_attempt_errors_still_allow_triage(self) -> None:
        from medical_deep_research.agentic_tools import AgenticEventBridge, tool_get_studies

        request = self._codex_pcc_request()
        bridge = AgenticEventBridge()
        bridge.search_results.extend(
            [
                SearchProviderResult(source="PubMed", query="q", error="HTTP 502"),
                SearchProviderResult(source="PMC", query="q", error="HTTP 502"),
                SearchProviderResult(source="Europe PMC", query="q", studies=[]),
                SearchProviderResult(
                    source="OpenAlex",
                    query="q",
                    studies=[
                        EvidenceStudy(
                            source="openalex",
                            source_id="direct",
                            title="AI virtual patient communication skills training",
                            abstract="A simulated patient supported communication skills training for medical students.",
                            publication_year="2026",
                            evidence_level="Level III",
                            sources=["openalex"],
                        )
                    ],
                ),
                SearchProviderResult(source="Crossref", query="q", studies=[]),
                SearchProviderResult(source="ClinicalTrials.gov", query="q", studies=[]),
            ]
        )

        result = await tool_get_studies(request, bridge, context="clinical")

        self.assertNotIn("missing_sources", result)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["studies"][0]["title"], "AI virtual patient communication skills training")

    async def test_browse_before_get_studies_errors(self) -> None:
        import json

        tools = self._build_tools_with_bridge()[0]
        result = json.loads(await call_fake_tool(tools, "browse_studies", page=1))  # type: ignore[arg-type]
        self.assertIn("error", result)

    async def test_browse_pages_pool(self) -> None:
        import json

        from medical_deep_research.agentic_tools import STUDY_PAGE_SIZE

        tools = self._build_tools_with_bridge()[0]
        await self._seed_multi(tools)
        page1 = json.loads(await call_fake_tool(tools, "get_studies", context="clinical"))  # type: ignore[arg-type]
        page2 = json.loads(await call_fake_tool(tools, "browse_studies", page=2))  # type: ignore[arg-type]
        self.assertEqual(page2["shown"], 20 - STUDY_PAGE_SIZE)
        self.assertFalse(page2["has_more"])
        idx1 = {s["idx"] for s in page1["studies"]}
        idx2 = {s["idx"] for s in page2["studies"]}
        self.assertEqual(idx1 & idx2, set())  # stable, disjoint indices across pages
        self.assertEqual(len(idx1 | idx2), 20)

    async def test_browse_filters_by_level_and_source(self) -> None:
        import json

        tools = self._build_tools_with_bridge()[0]
        await self._seed_multi(tools, sources=("PubMed", "OpenAlex"))
        by_level = json.loads(await call_fake_tool(tools, "browse_studies", evidence_level="Level II"))  # type: ignore[arg-type]
        self.assertTrue(by_level["studies"])
        self.assertTrue(all(s["evidence_level"] == "Level II" for s in by_level["studies"]))
        self.assertEqual(by_level["filtered_total"], by_level["counts_by_evidence_level"]["Level II"])

        by_source = json.loads(await call_fake_tool(tools, "browse_studies", source="OpenAlex"))  # type: ignore[arg-type]
        self.assertEqual(by_source["filtered_total"], 5)
        self.assertTrue(all("OpenAlex" in s["sources"] for s in by_source["studies"]))

    async def test_browse_does_not_reset_screening(self) -> None:
        tools, bridge = self._build_tools_with_bridge()
        await self._seed_multi(tools)
        await call_fake_tool(tools, "screen_studies", included_indices=[1, 2])
        screening_before = dict(bridge.screening)
        pool_before = len(bridge._pre_scored)
        await call_fake_tool(tools, "browse_studies", page=1)
        self.assertEqual(bridge.screening, screening_before)
        self.assertEqual(len(bridge._pre_scored), pool_before)
        self.assertEqual(pool_before, 2)

    async def test_screen_whitelist_drops_unlisted(self) -> None:
        import json

        tools, bridge = self._build_tools_with_bridge()
        await self._seed_multi(tools)
        result = json.loads(await call_fake_tool(tools, "screen_studies", included_indices=[1]))  # type: ignore[arg-type]
        self.assertEqual(result["included"], 1)
        self.assertEqual(result["not_selected"], 19)
        self.assertEqual(len(bridge._pre_scored), 1)

    async def test_screen_empty_includes_errors(self) -> None:
        import json

        tools, bridge = self._build_tools_with_bridge()
        await self._seed_multi(tools)
        result = json.loads(await call_fake_tool(tools, "screen_studies", included_indices=[]))  # type: ignore[arg-type]
        self.assertIn("error", result)
        self.assertEqual(len(bridge._pre_scored), 20)  # pool untouched

    async def test_screen_before_get_studies_errors(self) -> None:
        import json

        tools = self._build_tools()
        result = json.loads(await call_fake_tool(tools, "screen_studies", included_indices=[1]))  # type: ignore[arg-type]
        self.assertIn("error", result)

    async def test_screen_filters_pool_before_ranking(self) -> None:
        import json

        tools = self._build_tools()
        await self._seed_pool(tools)
        result = json.loads(
            await call_fake_tool(
                tools, "screen_studies", included_indices=[1], excluded_indices=[1], exclusion_reasons=["wrong population"]
            )  # type: ignore[arg-type]
        )
        # Whitelist: index 1 is included so it survives (the excluded_indices entry is
        # ignored for an included study); the single-study pool yields no exclusions.
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["included"], 1)
        self.assertEqual(result["excluded"], 0)

    async def test_appraise_normalizes_certainty(self) -> None:
        import json

        tools = self._build_tools()
        await self._seed_pool(tools)
        await call_fake_tool(tools, "finalize_ranking", ranked_indices=[1], rationale="r")
        result = json.loads(
            await call_fake_tool(
                tools,
                "appraise_evidence",
                findings=["Pain reduced", "Mortality unchanged"],
                certainties=["MODERATE", "uncertain"],
                rationales=["RCT", "few events"],
                reference_numbers_csv=["1", "1"],
            )  # type: ignore[arg-type]
        )
        # "MODERATE" normalizes to the bucket; "uncertain" is unparseable -> Low default.
        self.assertEqual(result["certainties"], ["Moderate", "Low"])


if __name__ == "__main__":
    unittest.main()
