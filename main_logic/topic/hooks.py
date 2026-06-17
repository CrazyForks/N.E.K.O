"""Topic-hook prompt helpers for proactive chat.

This module intentionally does not schedule, persist, or deliver anything.
It only turns already-approved proactive candidates into a compact prompt
section that the existing /api/proactive_chat Phase 2 path can consume.

Prompt placement contract:
* reflection follow-up topics render here and are appended to memory_context,
  next to the long-term conversation history they extend.
* open_threads stay in the activity-state section, close to live state/tone and
  the decision rules. Do not merge them back into this memory-cue section: they
  are recent unfinished semantic threads, not old reminiscence.
* background deep-topic hooks are delivered through build_topic_hook_callback
  in delivery.py, not through this helper.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from main_logic.topic.common import clean_text


# Deliberately encouraging, not discouraging: paired with Phase 2's repeated
# anti-repeat warnings, weak/negative wording made the model treat callbacks as
# "high repeat risk" and skip them. Frame old topics as welcome, natural memory
# cues. Keep the rendering intentionally quieter than major ====== sections:
# memory cues should be available near conversation history, not compete with
# recent-chat dedup or activity-state decision blocks.
_INTRO_ZH = "回忆线索：以下旧话题距今较久，可顺手接、但没必要主动提出。"
_INTRO_ZH_TW = "回憶線索：以下舊話題距今較久，可順手接、但沒必要主動提出。"
_INTRO_EN = "Memory cues: older topics from prior conversations; okay to pick up naturally, but no need to raise proactively."
_INTRO_JA = "記憶の手がかり：以前の古い話題です。自然に拾ってもよいですが、無理に持ち出す必要はありません。"
_INTRO_KO = "기억 단서: 이전 대화의 오래된 화제입니다. 자연스럽게 이어도 되지만, 먼저 꺼낼 필요는 없습니다."
_INTRO_ES = "Pistas de memoria: temas antiguos de conversaciones previas; puedes retomarlos con naturalidad, pero no hace falta sacarlos activamente."
_INTRO_PT = "Pistas de memória: temas antigos de conversas anteriores; pode retomá-los naturalmente, mas não precisa puxá-los ativamente."
_INTRO_RU = "Подсказки памяти: старые темы из прошлых разговоров; можно естественно вернуться к ним, но не нужно поднимать их специально."

_INTROS = {
    "zh": _INTRO_ZH,
    "zh-CN": _INTRO_ZH,
    "zh-TW": _INTRO_ZH_TW,
    "en": _INTRO_EN,
    "ja": _INTRO_JA,
    "ko": _INTRO_KO,
    "es": _INTRO_ES,
    "pt": _INTRO_PT,
    "ru": _INTRO_RU,
}

_LABELS = {
    "zh": "较久前的回忆线索",
    "zh-CN": "较久前的回忆线索",
    "zh-TW": "較久前的回憶線索",
    "en": "Older memory cue",
    "ja": "古い記憶の手がかり",
    "ko": "오래된 기억 단서",
    "es": "Pista de memoria antigua",
    "pt": "Pista de memória antiga",
    "ru": "Давняя подсказка памяти",
}

def _lang_key(lang: str) -> str:
    raw = (lang or "").strip()
    if raw in _LABELS:
        return raw
    if raw.lower().startswith("zh"):
        return "zh"
    short = raw.split("-", 1)[0].lower()
    if short in _LABELS:
        return short
    return "en"


def _iter_followup_texts(followup_topics: Iterable[Mapping[str, Any]] | None) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for topic in followup_topics or []:
        if not isinstance(topic, Mapping):
            continue
        text = clean_text(topic.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def build_topic_hook_prompt(
    lang: str,
    *,
    followup_topics: Iterable[Mapping[str, Any]] | None = None,
    max_items: int = 3,
) -> str:
    """Render old reflection follow-up topics for the proactive prompt.

    The output is deliberately a prompt section, not final copy. Phase 2 still
    owns character voice, timing, and whether to pass. This helper renders only
    old reflection follow-ups; open_threads stay in the activity-state section,
    and the background topic pool delivers through build_topic_hook_callback.
    """
    key = _lang_key(lang)
    label = _LABELS.get(key, _LABELS["en"])
    intro = _INTROS.get(key, _INTROS["en"])

    memory_texts = _iter_followup_texts(followup_topics)[:max_items]
    if not memory_texts:
        return ""

    lines = [intro]
    for text in memory_texts:
        lines.append(f"- {label}: {text}")
    return "\n".join(lines) + "\n"
