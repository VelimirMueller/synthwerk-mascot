"""The flat vector index: one normalised float32 row per chunk, plus the chunk sources.

Files in the index directory (written by the image build, read-only at run time):

- `vectors.f32`: `count x dim` float32, little endian, row-major. No header.
- `chunks.json`: one object per row: path, anchor, title, text.
- `manifest.json`: model, corpus reference, dim, count and the SHA-256 of `vectors.f32`.

Search is exact: the vectors are normalised, so cosine similarity is one matrix-vector product.
No index library, no database.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthwerk_mascot.corpus import Chunk
from synthwerk_mascot.embedder import Embedder, Vectors, passage_text

VECTORS_FILE = "vectors.f32"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"
FORMAT_VERSION = 1


class IndexLoadError(RuntimeError):
    """The index files are missing or do not match their manifest."""


@dataclass(frozen=True)
class Manifest:
    model: str
    corpus_ref: str
    source_url_base: str  # a chunk URL is source_url_base + path + "#" + anchor
    dim: int
    count: int
    vectors_sha256: str
    format: int = FORMAT_VERSION


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, vectors: Vectors, chunks: Sequence[Chunk], manifest: Manifest) -> None:
        if vectors.shape != (len(chunks), manifest.dim):
            raise IndexLoadError(f"vectors {vectors.shape} do not match {len(chunks)} chunks")
        self.vectors = vectors
        self.chunks = list(chunks)
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self, query: Vectors, k: int, min_score: float = -1.0, min_margin: float = 0.0
    ) -> list[Hit]:
        """The k best chunks for one normalised query vector, best first. At most one chunk
        per section (path + anchor), so the citations differ. Hits below min_score drop.

        No hit at all when the best score beats the median score by less than min_margin:
        an off-topic question scores about the same on every chunk."""
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ query.reshape(-1).astype(np.float32)
        if float(scores.max() - np.median(scores)) < min_margin:
            return []
        hits: list[Hit] = []
        sections: set[tuple[str, str]] = set()
        for row in np.argsort(-scores, kind="stable"):
            score = float(scores[row])
            if score < min_score or len(hits) == k:
                break
            chunk = self.chunks[int(row)]
            if (chunk.path, chunk.anchor) in sections:
                continue
            sections.add((chunk.path, chunk.anchor))
            hits.append(Hit(chunk, score))
        return hits

    def url(self, chunk: Chunk) -> str:
        base = self.manifest.source_url_base + chunk.path
        return f"{base}#{chunk.anchor}" if chunk.anchor else base

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        data = np.ascontiguousarray(self.vectors, dtype="<f4").tobytes()
        (directory / VECTORS_FILE).write_bytes(data)
        chunks = [asdict(c) for c in self.chunks]
        (directory / CHUNKS_FILE).write_text(
            json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        (directory / MANIFEST_FILE).write_text(
            json.dumps(asdict(self.manifest), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, expected_dim: int | None = None) -> "VectorIndex":
        try:
            raw: dict[str, Any] = json.loads((directory / MANIFEST_FILE).read_text("utf-8"))
            manifest = Manifest(**raw)
            data = (directory / VECTORS_FILE).read_bytes()
            rows = json.loads((directory / CHUNKS_FILE).read_text("utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise IndexLoadError(f"cannot read the index in {directory}: {exc}") from exc
        if manifest.format != FORMAT_VERSION:
            raise IndexLoadError(f"index format {manifest.format}, expected {FORMAT_VERSION}")
        if hashlib.sha256(data).hexdigest() != manifest.vectors_sha256:
            raise IndexLoadError("vectors.f32 does not match its SHA-256 in manifest.json")
        if expected_dim is not None and manifest.dim != expected_dim:
            raise IndexLoadError(f"index dim {manifest.dim}, model dim {expected_dim}")
        if len(data) != manifest.count * manifest.dim * 4:
            raise IndexLoadError("vectors.f32 size does not match count x dim")
        vectors = np.frombuffer(data, dtype="<f4").reshape(manifest.count, manifest.dim)
        chunks = [Chunk(**row) for row in rows]
        return cls(vectors.astype(np.float32), chunks, manifest)


def build_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    model: str,
    corpus_ref: str,
    source_url_base: str,
    batch_size: int = 16,
) -> VectorIndex:
    """Embed every chunk with the `passage: ` prefix and wrap the rows in an index."""
    parts = [
        embedder.embed([passage_text(c.title, c.text) for c in chunks[i : i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ]
    vectors = (
        np.concatenate(parts).astype(np.float32)
        if parts
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    digest = hashlib.sha256(np.ascontiguousarray(vectors, dtype="<f4").tobytes()).hexdigest()
    manifest = Manifest(
        model=model,
        corpus_ref=corpus_ref,
        source_url_base=source_url_base,
        dim=embedder.dim,
        count=len(chunks),
        vectors_sha256=digest,
    )
    return VectorIndex(vectors, chunks, manifest)
