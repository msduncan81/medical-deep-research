"""Reference-set benchmark for the GRADE graph: certainty per outcome vs recent Cochrane reviews.

For each question x model the full pipeline runs with the Cochrane review's own outcomes fixed in advance
and the review itself excluded from every search (by CD number / PMID), so the certainty comes only from
the primary studies and other reviews. Certainty in mode A (computed) and mode B (holistic) is compared
outcome by outcome with the Cochrane Summary of Findings.

    uv run python scripts/benchmark_grade.py benchmarks/grade_reference.json OUT_DIR \
        [--models strata,deepseek] [--questions q1,q2]

Models (OpenAI-compatible endpoints):
    strata    http://127.0.0.1:8080/v1, model qwen3.8-flash-next-iq3_s (local)
    deepseek  OpenCode Go, model deepseek-v4.1-flash; key read from ~/.local/share/opencode/auth.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

LEVEL = {"high": 4, "moderate": 3, "low": 2, "very low": 1}
NAMES = {4: "high", 3: "moderate", 2: "low", 1: "very low", None: "no evidence"}

MODELS: dict[str, dict[str, Any]] = {
    "strata": {"base_url": "http://127.0.0.1:8080/v1", "model": "qwen3.8-flash-next-iq3_s", "context": None,
               "key": lambda: "local"},
    "deepseek": {"base_url": "https://opencode.ai/inference/go/openai/v1", "model": "deepseek-v4.1-flash",
                 "context": "128000", "key": lambda: json.loads(
                     (Path.home() / ".local/share/opencode/auth.json").read_text())["opencode-go"]["key"]},
}


def _query(question: dict[str, Any]) -> tuple[str, dict[str, str]]:
    pico = question["pico"]
    payload = {"population": pico["population"], "intervention": pico["intervention"],
               "comparison": pico["comparator"], "outcome": "; ".join(o["name"] for o in question["outcomes"])}
    return "; ".join(f"{k.capitalize()}: {v}" for k, v in payload.items()), payload


async def run_one(question: dict[str, Any], model_name: str, out: Path) -> dict[str, Any]:
    from medical_deep_research import runtime as rt
    from medical_deep_research.models import EventType, RunRequest

    cfg = MODELS[model_name]
    os.environ["MDR_GRADE_PIPELINE"] = "1"
    os.environ["MDR_LOCAL_BASE_URL"] = cfg["base_url"]
    os.environ["MDR_LOCAL_API_KEY"] = cfg["key"]()
    if cfg["context"]:
        os.environ["MDR_LOCAL_CONTEXT_TOKENS"] = cfg["context"]
    else:
        os.environ.pop("MDR_LOCAL_CONTEXT_TOKENS", None)

    async def skip_pdf_wait(_request, _bridge, ranks=None, timeout_seconds=900):  # noqa: ANN001
        return {"status": "skipped", "reason": "Benchmark run: no user PDF upload."}

    rt.tool_await_user_pdfs = skip_pdf_wait
    query, payload = _query(question)
    cochrane = question["cochrane"]
    request = RunRequest(
        run_id=f"bench-{question['id']}-{model_name}-{int(time.time())}", query=query, query_type="pico",
        mode="detailed", provider="local", model=cfg["model"], language="pt", query_payload=payload,
        recent_years_lookback=50,
        grade_outcomes_override=[{k: v for k, v in o.items() if not k.startswith("cochrane_")}
                                 for o in question["outcomes"]],
        grade_comparison_override=question.get("comparison", {}),
        exclude_ids=[x for x in (cochrane.get("cd"), cochrane.get("pmid"), cochrane.get("doi")) if x],
    )
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    profiles: list[dict[str, Any]] = []
    final = None
    with (out / "events.log").open("w", encoding="utf-8") as log:
        async for ev in rt.LangChainLocalRuntime().stream_run(request):
            extra = ev.extra or {}
            if ev.event_type == EventType.TOOL_CALLED:
                log.write(f"[{time.time() - started:7.1f}s] {ev.tool_name} {str(extra.get('tool_input') or '')[:160]}\n")
            elif ev.event_type == EventType.AGENT_STARTED:
                log.write(f"[{time.time() - started:7.1f}s] {ev.message[:200]}\n")
            if ev.event_type == EventType.ARTIFACT_CREATED and str(ev.artifact_type or "").startswith("grade_"):
                (out / f"{ev.artifact_type}.json").write_text(json.dumps(ev.artifact_json, ensure_ascii=False, indent=1))
                if ev.artifact_type == "grade_profile":
                    profiles = (ev.artifact_json or {}).get("profiles", [])
            if ev.event_type == EventType.RUN_COMPLETED:
                final = ev
            log.flush()
    if final:
        (out / "report.md").write_text(final.report_markdown or "", encoding="utf-8")
    by_id = {p["outcome"]["id"]: p for p in profiles}
    rows = []
    for outcome in question["outcomes"]:
        profile = by_id.get(outcome["id"], {})
        rows.append({"outcome": outcome["id"], "name": outcome["name"],
                     "cochrane": LEVEL.get(str(outcome.get("cochrane_certainty", "")).lower()),
                     "a": profile.get("certainty_a"), "b": profile.get("certainty_b")})
    summary = {"question": question["id"], "model": model_name, "seconds": round(time.time() - started),
               "report_source": (final.extra or {}).get("report_source") if final else None, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def score(summaries: list[dict[str, Any]]) -> str:
    lines = ["# Concordância com a Cochrane (certeza GRADE por desfecho)", "",
             "| Pergunta | Modelo | Desfecho | Cochrane | Modo A | Modo B |", "|---|---|---|---|---|---|"]
    stats: dict[tuple[str, str], list[int]] = {}
    for s in summaries:
        for r in s["rows"]:
            lines.append(f"| {s['question']} | {s['model']} | {r['name']} | {NAMES[r['cochrane']]} | "
                         f"{NAMES[r['a']]} | {NAMES[r['b']]} |")
            if r["cochrane"] is None:
                continue
            for mode in ("a", "b"):
                if r[mode] is not None:
                    stats.setdefault((s["model"], mode), []).append(r[mode] - r["cochrane"])
    lines += ["", "| Modelo | Modo | Desfechos comparados | Concordância exata | Até 1 nível | Diferença média (níveis) |",
              "|---|---|---|---|---|---|"]
    for (model, mode), diffs in sorted(stats.items()):
        n = len(diffs)
        exact = sum(d == 0 for d in diffs) / n
        within = sum(abs(d) <= 1 for d in diffs) / n
        lines.append(f"| {model} | {mode.upper()} | {n} | {exact:.0%} | {within:.0%} | {sum(diffs) / n:+.2f} |")
    lines += ["", "Diferença positiva: a avaliação automática é mais otimista que a Cochrane; negativa: mais conservadora.",
              "Desfechos sem evidência encontrada não entram nas médias."]
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("out")
    parser.add_argument("--models", default="strata,deepseek")
    parser.add_argument("--questions", default="")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    wanted = {q for q in args.questions.split(",") if q}
    out = Path(args.out)
    summaries = []
    for question in reference["questions"]:
        if wanted and question["id"] not in wanted:
            continue
        for model in [m for m in args.models.split(",") if m]:
            print(f"== {question['id']} / {model}", flush=True)
            try:
                summaries.append(await run_one(question, model, out / question["id"] / model))
            except Exception as exc:  # noqa: BLE001 - keep going with the other runs
                print(f"   failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            (out / "results.md").write_text(score(summaries), encoding="utf-8")
    print((out / "results.md").read_text())


if __name__ == "__main__":
    asyncio.run(main())
