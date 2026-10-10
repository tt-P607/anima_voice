"""剥离通话语音中的等待标记并按句切分。"""

from __future__ import annotations

import re

_WAIT = re.compile(r"\[wait\s*:\s*([0-9]+(?:\.[0-9]+)?)\]", re.IGNORECASE)
_SENTENCE = re.compile(r".+?(?:……|[。！？!?\n]+)|.+$", re.DOTALL)


def parse_speech_segments(
    text: str, split_sentences: bool = True
) -> list[tuple[str, float]]:
    """将文本解析为正文和播放前等待秒数。"""

    segments: list[tuple[str, float]] = []
    pending = 0.0
    cursor = 0
    for match in _WAIT.finditer(text):
        segments.extend(_split(text[cursor : match.start()], pending, split_sentences))
        pending = float(match.group(1))
        cursor = match.end()
    segments.extend(_split(text[cursor:], pending, split_sentences))
    return [(body, wait) for body, wait in segments if body.strip()]


def _split(text: str, wait: float, split_sentences: bool) -> list[tuple[str, float]]:
    """切分文本并把等待只绑定至第一个有效片段。"""

    content = text.strip()
    if not content:
        return []
    parts = (
        [match.group(0).strip() for match in _SENTENCE.finditer(content)]
        if split_sentences
        else [content]
    )
    return [
        (part, wait if index == 0 else 0.0) for index, part in enumerate(parts) if part
    ]
