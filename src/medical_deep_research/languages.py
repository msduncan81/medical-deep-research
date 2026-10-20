"""Language codes, display names and light-weight language detection for reports and queries."""

from __future__ import annotations

import re

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ko": "Korean (한국어)",
    "pt": "Brazilian Portuguese (português do Brasil)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
}

# Extra writing guidance appended to report prompts for a target language.
LANGUAGE_STYLE_NOTES: dict[str, str] = {
    "pt": (
        "Use Brazilian Portuguese medical terminology as used in Brazilian guidelines and journals. "
        "Write 'atenção primária à saúde (APS)' for primary care, never 'atenção básica'. "
        "Keep evidence-level labels as 'Level I' to 'Level V' and keep GRADE certainty in the form "
        "'certeza da evidência: alta/moderada/baixa/muito baixa'."
    ),
}

_PT_MARKERS = re.compile(
    r"\b(de|da|do|das|dos|que|não|com|para|uma|um|em|no|na|nos|nas|os|as|é|são|pelo|pela|entre|sobre)\b",
    re.IGNORECASE,
)
_EN_MARKERS = re.compile(
    r"\b(the|and|of|with|for|in|on|is|are|was|were|this|that|from|by|between|among)\b",
    re.IGNORECASE,
)
_PT_DIACRITICS = re.compile(r"[ãõçáéíóúâêôà]", re.IGNORECASE)
_HANGUL = re.compile(r"[가-힣]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def language_display_name(code: str | None) -> str:
    """Human-readable language name for prompts; unknown codes are returned as given."""
    if not code:
        return LANGUAGE_NAMES["en"]
    return LANGUAGE_NAMES.get(code.lower(), code)


def language_style_note(code: str | None) -> str:
    return LANGUAGE_STYLE_NOTES.get((code or "").lower(), "")


def report_language_instruction(code: str | None) -> str:
    """Sentence telling the model which language to write the report in, plus style guidance."""
    note = language_style_note(code)
    sentence = f"Write the report in {language_display_name(code)}."
    return f"{sentence} {note}" if note else sentence


def is_english(code: str | None) -> bool:
    return (code or "").lower() in {"", "en", "english"}


def looks_like_portuguese(text: str) -> bool:
    sample = text[:6000]
    pt = len(_PT_MARKERS.findall(sample)) + 2 * len(_PT_DIACRITICS.findall(sample))
    en = len(_EN_MARKERS.findall(sample))
    return pt >= 3 and pt > 1.5 * en


def looks_like_language(text: str, code: str | None) -> bool | None:
    """True/False when the language can be detected cheaply, None when it cannot."""
    code = (code or "").lower()
    if not text.strip():
        return None
    if code == "pt":
        return looks_like_portuguese(text)
    if code == "ko":
        letters = len(_LETTER.findall(text[:6000])) or 1
        return len(_HANGUL.findall(text[:6000])) / letters > 0.2
    if is_english(code):
        return not looks_like_portuguese(text) and not _HANGUL.search(text[:6000])
    return None
