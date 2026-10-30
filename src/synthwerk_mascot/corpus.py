"""Split the public Markdown docs into chunks. Each chunk keeps the source it came from.

A chunk is the text under one heading (or a part of it, when the section is long). Its
source is the file path plus the GitHub anchor of the heading, so an answer can cite it.
"""

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1200
MIN_LETTERS = 40

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)")
_HTML_BLOCK = re.compile(r"<(picture|svg|details)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>\n]+>")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    path: str  # repo-relative file path, for example docs/infrastructure.md
    anchor: str  # GitHub anchor of the nearest heading; "" for text above the first heading
    title: str  # heading path, for example "Infrastructure > Environments"
    text: str


def inline_text(markdown: str) -> str:
    """Markdown inline syntax to plain text: links keep their label, code and emphasis lose
    their marks."""
    text = _LINK.sub(r"\1", markdown)
    text = _TAG.sub("", text)
    return text.replace("`", "").replace("**", "").replace("__", "").strip()


def slugify(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to '-'."""
    text = inline_text(heading).lower()
    kept = "".join(
        ch
        for ch in text
        if ch in " -_" or unicodedata.category(ch)[0] in {"L", "N"} or ch.isalnum()
    )
    return kept.replace(" ", "-")


def _clean(lines: Iterable[str]) -> str:
    out: list[str] = []
    for line in lines:
        if _TABLE_RULE.match(line):
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [inline_text(cell) for cell in stripped.strip("|").split("|")]
            line = " | ".join(cell for cell in cells if cell)
        else:
            line = inline_text(line)
        out.append(line.rstrip())
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _pack(text: str, max_chars: int) -> Iterator[str]:
    """Paragraphs packed into parts of at most max_chars; a longer paragraph is cut at
    sentence ends, then hard."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_chars:
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            pieces.append(sentence)
    part = ""
    for piece in pieces:
        if not piece:
            continue
        joined = f"{part}\n\n{piece}" if part else piece
        if len(joined) <= max_chars:
            part = joined
            continue
        if part:
            yield part
        part = piece
    if part:
        yield part


def split_markdown(path: str, markdown: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Chunks of one Markdown file. Pictures, HTML and ```text diagrams are left out: they
    carry no words a reader asks about."""
    markdown = _HTML_COMMENT.sub("", markdown)
    markdown = _HTML_BLOCK.sub("", markdown)

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    headings: list[tuple[int, str]] = []
    anchor = ""
    body: list[str] = []
    fence: str | None = None
    skip_fence = False

    def flush() -> None:
        text = _clean(body)
        body.clear()
        if _letters(text) < MIN_LETTERS:
            return
        title = " > ".join(h for _, h in headings) or Path(path).stem
        chunks.extend(Chunk(path, anchor, title, part) for part in _pack(text, max_chars))

    for line in markdown.splitlines():
        fence_match = _FENCE.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1) == fence:
                fence = None
            elif not skip_fence:
                body.append(line)
            continue
        if fence_match:
            fence = fence_match.group(1)
            skip_fence = fence_match.group(2) in {"text", "mermaid", "svg"}
            continue
        heading = _HEADING.match(line)
        if heading is None:
            body.append(line)
            continue
        flush()
        level, raw = len(heading.group(1)), heading.group(2)
        name = inline_text(raw)
        headings = [h for h in headings if h[0] < level] + [(level, name)]
        slug = slugify(raw)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count}"
    flush()
    return chunks


def load_corpus(root: Path, pattern: str = "docs/*.md") -> list[Chunk]:
    """Chunks of every file under root that matches pattern, in a stable order."""
    chunks: list[Chunk] = []
    for file in sorted(root.glob(pattern)):
        relative = file.relative_to(root).as_posix()
        chunks.extend(split_markdown(relative, file.read_text(encoding="utf-8")))
    return chunks
