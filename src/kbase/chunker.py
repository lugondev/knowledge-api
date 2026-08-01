"""Text -> retrievable pieces, cut on headings first and length second.

Sizes are in characters, not tokens. Every embedding provider tokenises
differently, and Vietnamese diacritics make token estimates swing hard enough
that a "400 token" chunk can be half the length one provider to the next.
Characters behave identically everywhere, and the only thing the limit really
has to do is keep a chunk small enough to be about one idea.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BOUNDARY = re.compile(r"[.!?…]\s|\n")


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    heading: str
    ordinal: int


def _sections(text: str) -> list[tuple[str, str]]:
    """[(heading path, body)] in document order.

    The stack holds (level, title), not just titles. Indexing a plain title list
    by `level - 1` assumes a document whose headings start at `#` and never skip
    a level; a document that opens at `##` puts its first heading at index 0, and
    then every later `##` nests under the previous one instead of replacing it.
    """
    stack: list[tuple[int, str]] = []
    out: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        joined = "\n".join(body).strip()
        if joined:
            out.append((" > ".join(title for _level, title in stack), joined))
        body.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            body.append(line)
    flush()
    return out


def _split_body(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    # Overlap must leave room to advance; otherwise every window starts where the
    # previous one did and the loop never ends.
    step_back = min(overlap, max_chars // 2)
    out: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + max_chars, len(body))
        if end < len(body):
            window = body[start:end]
            cut = -1
            for m in _BOUNDARY.finditer(window):
                if m.end() > max_chars - step_back:
                    cut = m.end()
                    break
            if cut > 0:
                end = start + cut
        piece = body[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(body):
            break
        start = max(end - step_back, start + 1)
    return out


def chunk_markdown(
    text: str, *, max_chars: int = 800, overlap: int = 100
) -> list[ChunkPiece]:
    pieces: list[ChunkPiece] = []
    for heading, body in _sections(text):
        for piece in _split_body(body, max_chars, overlap):
            pieces.append(ChunkPiece(text=piece, heading=heading, ordinal=len(pieces)))
    return pieces
