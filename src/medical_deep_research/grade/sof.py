"""Nodes D2-D3 — Summary of Findings table and GRADE informative statements (computed, not written by the model).

SoF columns follow the Cochrane Handbook ch. 14 and Core GRADE 6: outcome, participants (studies),
relative effect (95% CI), risk with comparator and with intervention (absolute effect from the baseline
risk of the included studies), certainty, comments, and footnotes explaining each rating up or down.
Statements follow Santesso et al. 2020 (GRADE 26) and Core GRADE 2: the verb depends on certainty
("reduces" / "probably reduces" / "may reduce" / "very uncertain"); "little or no difference" when the
CI includes the null; "large" only with GRADE's defined large-effect thresholds (RR <= 0.5 or >= 2) or,
with an MID, an effect beyond it. No official Portuguese translation of the statement table was found;
the Portuguese wording mirrors the English.
"""

from __future__ import annotations

import re
from typing import Any

from .synthesis import RATIO_MEASURES

LEVEL_LABELS = {
    "pt": {4: "Alta", 3: "Moderada", 2: "Baixa", 1: "Muito baixa", None: "Sem evidência"},
    "en": {4: "High", 3: "Moderate", 2: "Low", 1: "Very low", None: "No evidence"},
}
DOMAIN_LABELS = {
    "pt": {"risk_of_bias": "risco de viés", "inconsistency": "inconsistência", "indirectness": "evidência indireta",
           "imprecision": "imprecisão", "publication_bias": "viés de publicação", "rating_up": "elevação"},
    "en": {"risk_of_bias": "risk of bias", "inconsistency": "inconsistency", "indirectness": "indirectness",
           "imprecision": "imprecision", "publication_bias": "publication bias", "rating_up": "rating up"},
}
CERTAINTY_SYMBOLS = {4: "⊕⊕⊕⊕", 3: "⊕⊕⊕◯", 2: "⊕⊕◯◯", 1: "⊕◯◯◯"}

_TEMPLATES = {
    "pt": {
        "effect": {4: "{i} {verb} {o}", 3: "{i} provavelmente {verb} {o}", 2: "{i} pode {verb_inf} {o}"},
        "none": {4: "{i} resulta em pouca ou nenhuma diferença em {o}",
                 3: "{i} provavelmente resulta em pouca ou nenhuma diferença em {o}",
                 2: "{i} pode resultar em pouca ou nenhuma diferença em {o}"},
        "very_low": "A evidência é muito incerta sobre o efeito de {i} em {o}",
        "no_evidence": "Nenhum estudo incluído relatou {o}",
        "verbs": {"down": ("reduz", "reduzir"), "up": ("aumenta", "aumentar")},
        "large": {"down": ("reduz muito", "reduzir muito"), "up": ("aumenta muito", "aumentar muito")},
    },
    "en": {
        "effect": {4: "{i} {verb} {o}", 3: "{i} probably {verb} {o}", 2: "{i} may {verb_inf} {o}"},
        "none": {4: "{i} results in little to no difference in {o}",
                 3: "{i} probably results in little to no difference in {o}",
                 2: "{i} may result in little to no difference in {o}"},
        "very_low": "The evidence is very uncertain about the effect of {i} on {o}",
        "no_evidence": "No included study reported {o}",
        "verbs": {"down": ("reduces", "reduce"), "up": ("increases", "increase")},
        "large": {"down": ("results in a large reduction in", "result in a large reduction in"),
                  "up": ("results in a large increase in", "result in a large increase in")},
    },
}


def _in_sentence(name: str) -> str:
    """Outcome name inside a sentence: lower-case the first letter unless it starts an acronym (PA, HbA1c)."""
    name = name.strip()
    if len(name) > 1 and name[0].isupper() and name[1].islower():
        return name[0].lower() + name[1:]
    return name


_PT_RATIONALES: list[tuple[str, str]] = [
    (r"(\d+)% of the weight comes from high risk-of-bias studies -> serious limitations",
     r"\1% do peso vem de estudos com alto risco de viés → limitações sérias"),
    (r"(\d+)% of the weight comes from high risk-of-bias studies; most evidence is not at high risk",
     r"\1% do peso vem de estudos com alto risco de viés; a maior parte da evidência não tem alto risco"),
    (r"; (\d+)% could not be judged \(limited appraisal\)", r"; \1% não pôde ser julgado (avaliação limitada)"),
    (r"; judged from abstracts only", r"; julgado apenas pelos resumos"),
    (r"single estimate: inconsistency not rated", r"estimativa única: inconsistência não avaliada"),
    (r"I² (\d+)%; estimates on both sides of the null", r"I² \1%; estimativas dos dois lados do nulo"),
    (r"I² (\d+)%; estimates on one side of the null", r"I² \1%; estimativas de um só lado do nulo"),
    (r"; some non-overlapping CIs", r"; há intervalos sem sobreposição"),
    (r"; no non-overlapping CIs", r"; intervalos sobrepostos"),
    (r"no quantitative estimate could be combined \(narrative evidence only\)",
     r"nenhuma estimativa quantitativa pôde ser combinada (evidência apenas narrativa)"),
    (r"CI includes the null and is very wide \(upper/lower = ([\d.]+)\)",
     r"IC inclui o nulo e é muito amplo (razão superior/inferior = \1)"),
    (r"CI includes the null with only (\d+) participants", r"IC inclui o nulo, com apenas \1 participantes"),
    (r"CI includes the null", r"IC inclui o nulo"),
    (r"CI crosses both the important-benefit and important-harm thresholds \(MID ([\d.]+)\)",
     r"IC cruza os limiares de benefício e de dano importantes (DMI \1)"),
    (r"CI crosses the MID \(([\d.]+)\)", r"IC cruza a DMI (\1)"),
    (r"very wide CI \(upper/lower = ([\d.]+)\)", r"IC muito amplo (razão superior/inferior = \1)"),
    (r"large relative effect from (\d+) participants, below the OIS \((\d+)\)",
     r"efeito relativo grande com \1 participantes, abaixo do tamanho ótimo de informação (\2)"),
    (r"(\d+) participants, under 30% of the OIS \((\d+)\)",
     r"\1 participantes, menos de 30% do tamanho ótimo de informação (\2)"),
    (r"(\d+) participants, below the OIS \((\d+)\)",
     r"\1 participantes, abaixo do tamanho ótimo de informação (\2)"),
    (r"Egger test P = ([\d.]+) \(small-study effects suspected\)",
     r"teste de Egger P = \1 (suspeita de efeito de estudos pequenos)"),
    (r"very large effect", r"efeito muito grande"),
    (r"large effect", r"efeito grande"),
]


def localize_rationale(text: str, language: str | None) -> str:
    """Code-generated domain rationales are written in English; translate the known ones for PT reports."""
    if _lang(language) != "pt":
        return text
    for pattern, replacement in _PT_RATIONALES:
        text = re.sub(pattern, replacement, text)
    return text


def _lang(language: str | None) -> str:
    return "pt" if (language or "").lower().startswith("pt") else "en"


def _main_body(profile: dict[str, Any]) -> dict[str, Any] | None:
    bodies = profile.get("bodies") or []
    rct = next((b for b in bodies if b["design"] == "rct" and b.get("estimate")), None)
    nrs = next((b for b in bodies if b["design"] == "nrs" and b.get("estimate")), None)
    if rct and nrs and (nrs.get("certainty_a") or 0) > (rct.get("certainty_a") or 0):
        return nrs
    return rct or nrs or (bodies[0] if bodies else None)


def informative_statement(profile: dict[str, Any], intervention: str, language: str | None) -> str:
    text = _statement(profile, intervention, language)
    return text[:1].upper() + text[1:]


def _statement(profile: dict[str, Any], intervention: str, language: str | None) -> str:
    lang = _lang(language)
    t = _TEMPLATES[lang]
    outcome = profile["outcome"]
    name = _in_sentence(outcome["name"])
    certainty = profile.get("certainty_a")
    if certainty is None:
        return t["no_evidence"].format(o=name)
    if certainty == 1:
        return t["very_low"].format(i=_in_sentence(intervention), o=name)
    body = _main_body(profile)
    estimate = (body or {}).get("estimate") or {}
    if not estimate:
        return t["very_low"].format(i=_in_sentence(intervention), o=name)
    measure, est, lo, hi = estimate["measure"], estimate["estimate"], estimate["ci_low"], estimate["ci_high"]
    null = 1.0 if measure in RATIO_MEASURES else 0.0
    if lo < null < hi:
        return t["none"][certainty].format(i=intervention, o=name)
    direction = "down" if est < null else "up"
    large = False
    if measure in RATIO_MEASURES:
        large = est <= 0.5 or est >= 2
    elif outcome.get("target") == "mid" and outcome.get("mid_value"):
        large = abs(est) >= 2 * abs(float(outcome["mid_value"]))
    verb, verb_inf = (t["large"] if large else t["verbs"])[direction]
    return t["effect"][certainty].format(i=intervention, o=name, verb=verb, verb_inf=verb_inf)


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def _absolute(body: dict[str, Any], pt: bool = True) -> tuple[str, str]:
    """(risk with comparator, risk with intervention + difference) per 1000, for ratio measures."""
    estimate = body.get("estimate") or {}
    cer = body.get("control_event_rate")
    if not estimate or estimate.get("measure") not in RATIO_MEASURES or not cer:
        return "—", "—"
    def risk(ratio: float) -> float:
        if estimate["measure"] == "OR":
            odds = cer / (1 - cer) * ratio
            return odds / (1 + odds)
        return min(1.0, cer * ratio)
    base = cer * 1000
    point = risk(estimate["estimate"]) * 1000
    lo, hi = sorted((risk(estimate["ci_low"]) * 1000, risk(estimate["ci_high"]) * 1000))
    diff = point - base
    if pt:
        sign = "a mais" if diff >= 0 else "a menos"
        return (f"{base:.0f} por 1.000",
                f"{point:.0f} por 1.000 ({abs(diff):.0f} {sign}; IC 95% {lo - base:+.0f} a {hi - base:+.0f})")
    sign = "more" if diff >= 0 else "fewer"
    return (f"{base:.0f} per 1,000",
            f"{point:.0f} per 1,000 ({abs(diff):.0f} {sign}; 95% CI {lo - base:+.0f} to {hi - base:+.0f})")


def summary_of_findings(profiles: list[dict[str, Any]], intervention: str, language: str | None) -> str:
    """Markdown SoF table with footnotes."""
    lang = _lang(language)
    labels, dlabels = LEVEL_LABELS[lang], DOMAIN_LABELS[lang]
    pt = lang == "pt"
    header = (
        "| Desfecho (importância) | Participantes (estudos) | Efeito relativo (IC 95%) | Risco com comparador | "
        "Risco com intervenção | Certeza (GRADE) | Comentário |" if pt else
        "| Outcome (importance) | Participants (studies) | Relative effect (95% CI) | Risk with comparator | "
        "Risk with intervention | Certainty (GRADE) | Comment |"
    )
    rows = [header, "|" + "---|" * 7]
    notes: list[str] = []
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for profile in profiles:
        outcome = profile["outcome"]
        importance = ("crítico" if outcome.get("importance") == "critical" else "importante") if pt else outcome.get("importance")
        statement = informative_statement(profile, intervention, language)
        body = _main_body(profile)
        certainty = profile.get("certainty_a")
        if body is None:
            rows.append(f"| {outcome['name']} ({importance}) | — | — | — | — | {labels[None]} | {statement} |")
            continue
        estimate = body.get("estimate") or {}
        k = estimate.get("k") or len(body.get("refs") or [])
        participants = f"{body['participants']:,}".replace(",", ".") if body.get("participants") else "?"
        comparator_risk, intervention_risk = _absolute(body, pt)
        if estimate and estimate.get("measure") in RATIO_MEASURES:
            relative = (f"{estimate['measure']} {_fmt(estimate['estimate'])} ({_fmt(estimate['ci_low'])} a "
                        f"{_fmt(estimate['ci_high'])})")
        else:
            # Differences (MD, SMD, RD) are absolute effects: shown once, in the intervention column.
            relative = "—"
            if estimate:
                names = {"MD": "DM" if pt else "MD", "SMD": "DMP" if pt else "SMD", "RD": "DR" if pt else "RD"}
                unit = f" {outcome['mid_unit']}" if outcome.get("mid_unit") and estimate["measure"] == "MD" else ""
                intervention_risk = (f"{names.get(estimate['measure'], estimate['measure'])} "
                                     f"{_fmt(estimate['estimate'])}{unit} ({_fmt(estimate['ci_low'])} a "
                                     f"{_fmt(estimate['ci_high'])})")
        marks = []
        for domain in body.get("domains") or []:
            if domain["rating"]:
                letter = next(letters)
                marks.append(letter)
                sign = "rebaixada" if domain["rating"] < 0 else "elevada"
                if not pt:
                    sign = "rated down" if domain["rating"] < 0 else "rated up"
                levels = abs(domain["rating"])
                rationale = localize_rationale(domain["rationale"], language)
                notes.append(f"{letter}. {outcome['name']}: {sign} {levels} {'nível' if levels == 1 else 'níveis'} "
                             f"por {dlabels.get(domain['domain'], domain['domain'])} — {rationale}" if pt else
                             f"{letter}. {outcome['name']}: {sign} {levels} level(s) for "
                             f"{dlabels.get(domain['domain'], domain['domain'])} — {domain['rationale']}")
        adjustment = (body.get("adjustment_a") or {})
        if adjustment.get("adjustment"):
            letter = next(letters)
            marks.append(letter)
            notes.append(f"{letter}. {outcome['name']}: ajuste global de {adjustment['adjustment']:+d} — "
                         f"{adjustment.get('rationale', '')}" if pt else
                         f"{letter}. {outcome['name']}: overall adjustment {adjustment['adjustment']:+d} — "
                         f"{adjustment.get('rationale', '')}")
        cert = f"{CERTAINTY_SYMBOLS.get(certainty, '')} {labels[certainty]}" + (
            f" ({', '.join(marks)})" if marks else "")
        rows.append(f"| {outcome['name']} ({importance}) | {participants} ({k}) | {relative} | {comparator_risk} | "
                    f"{intervention_risk} | {cert} | {statement} |")
        for group in body.get("subgroups") or []:
            if group.get("credible"):
                for level in group["levels"]:
                    rows.append(f"| ↳ {group['variable']}: {level['level']} | ({level['k']}) | "
                                f"{estimate.get('measure', '')} {_fmt(level['estimate'])} ({_fmt(level['ci_low'])} a "
                                f"{_fmt(level['ci_high'])}) | | | | |")
    table = "\n".join(rows)
    if notes:
        table += "\n\n" + ("**Notas da tabela**" if pt else "**Footnotes**") + "\n\n" + "\n".join(f"- {n}" for n in notes)
    return table


def modes_table(profiles: list[dict[str, Any]], language: str | None) -> str:
    """Appendix: certainty in mode A (computed) and mode B (holistic), and agreement with reviews."""
    lang = _lang(language)
    labels = LEVEL_LABELS[lang]
    pt = lang == "pt"
    rows = ["| Desfecho | Modo A (calculado) | Modo B (holístico) | Revisões |" if pt else
            "| Outcome | Mode A (computed) | Mode B (holistic) | Reviews |", "|---|---|---|---|"]
    for profile in profiles:
        reviews = "; ".join(profile.get("comparison_notes") or []) or "—"
        rows.append(f"| {profile['outcome']['name']} | {labels[profile.get('certainty_a')]} | "
                    f"{labels[profile.get('certainty_b')]} | {reviews} |")
    return "\n".join(rows)
