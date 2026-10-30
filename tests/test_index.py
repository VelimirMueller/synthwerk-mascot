import json
from pathlib import Path

import numpy as np
import pytest

from synthwerk_mascot.corpus import Chunk
from synthwerk_mascot.embedder import Embedder, query_text
from synthwerk_mascot.index import (
    CHUNKS_FILE,
    MANIFEST_FILE,
    VECTORS_FILE,
    IndexLoadError,
    Manifest,
    VectorIndex,
    build_index,
)
from tests.conftest import FAKE_DIM, REF


def _copy(src: Path, dst: Path) -> Path:
    dst.mkdir()
    for name in (VECTORS_FILE, CHUNKS_FILE, MANIFEST_FILE):
        (dst / name).write_bytes((src / name).read_bytes())
    return dst


def test_index_build_writes_flat_float32_file_and_sources(index_dir: Path) -> None:
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    chunks = json.loads((index_dir / CHUNKS_FILE).read_text())
    size = (index_dir / VECTORS_FILE).stat().st_size
    assert manifest["dim"] == FAKE_DIM
    assert manifest["count"] == len(chunks) > 0
    assert size == manifest["count"] * FAKE_DIM * 4
    assert manifest["corpus_ref"] == REF
    assert set(chunks[0]) == {"path", "anchor", "title", "text"}


def test_index_load_roundtrip_rows_normalised(index_dir: Path) -> None:
    index = VectorIndex.load(index_dir, expected_dim=FAKE_DIM)
    assert index.vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), 1.0, rtol=1e-5)


def test_index_load_changed_vectors_refused(index_dir: Path, tmp_path: Path) -> None:
    copy = _copy(index_dir, tmp_path / "i")
    data = bytearray((copy / VECTORS_FILE).read_bytes())
    data[0] ^= 0xFF
    (copy / VECTORS_FILE).write_bytes(bytes(data))
    with pytest.raises(IndexLoadError, match="SHA-256"):
        VectorIndex.load(copy)


def test_index_load_other_model_dim_refused(index_dir: Path) -> None:
    with pytest.raises(IndexLoadError, match="dim"):
        VectorIndex.load(index_dir, expected_dim=384)


def test_index_load_missing_files_refused(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError, match="cannot read"):
        VectorIndex.load(tmp_path)


def test_index_load_other_format_refused(index_dir: Path, tmp_path: Path) -> None:
    copy = _copy(index_dir, tmp_path / "i")
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["format"] = 99
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(IndexLoadError, match="format"):
        VectorIndex.load(copy)


def test_index_load_count_not_matching_size_refused(index_dir: Path, tmp_path: Path) -> None:
    copy = _copy(index_dir, tmp_path / "i")
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["count"] += 1
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(IndexLoadError, match="size"):
        VectorIndex.load(copy)


def test_index_vectors_not_matching_chunks_refused() -> None:
    manifest = Manifest("m", REF, "u/", 4, 2, "x")
    with pytest.raises(IndexLoadError, match="do not match"):
        VectorIndex(np.zeros((1, 4), dtype=np.float32), [], manifest)


def test_search_question_words_best_source_first(
    loaded_index: VectorIndex, embedder: Embedder
) -> None:
    query = embedder.embed([query_text("which colour is the accent of the palette")])[0]
    hits = loaded_index.search(query, k=2)
    assert hits[0].chunk.path == "docs/design-system.md"
    assert hits[0].chunk.anchor == "colour"
    assert hits[0].score >= hits[1].score


def test_search_one_hit_per_section() -> None:
    chunks = [Chunk("d.md", "a", "A", "one"), Chunk("d.md", "a", "A", "two")]
    chunks.append(Chunk("d.md", "b", "B", "three"))
    vectors = np.array([[1, 0], [0.99, 0.141], [0.6, 0.8]], dtype=np.float32)
    index = VectorIndex(vectors, chunks, Manifest("m", REF, "u/", 2, 3, "x"))
    hits = index.search(np.array([1, 0], dtype=np.float32), k=3)
    assert [h.chunk.text for h in hits] == ["one", "three"]


def test_search_below_min_score_dropped(loaded_index: VectorIndex, embedder: Embedder) -> None:
    query = embedder.embed([query_text("palette accent colour")])[0]
    assert loaded_index.search(query, k=3, min_score=1.01) == []


def test_search_flat_scores_below_margin_no_hit(
    loaded_index: VectorIndex, embedder: Embedder
) -> None:
    query = embedder.embed([query_text("palette accent colour")])[0]
    assert loaded_index.search(query, k=3, min_margin=0.0)
    assert loaded_index.search(query, k=3, min_margin=2.0) == []


def test_search_empty_index_no_hit() -> None:
    index = VectorIndex(np.zeros((0, 2), dtype=np.float32), [], Manifest("m", REF, "u/", 2, 0, "x"))
    assert index.search(np.array([1, 0], dtype=np.float32), k=3) == []


def test_url_with_and_without_anchor(loaded_index: VectorIndex) -> None:
    base = f"https://example.test/blob/{REF}/"
    assert loaded_index.url(Chunk("docs/a.md", "x-y", "t", "b")) == f"{base}docs/a.md#x-y"
    assert loaded_index.url(Chunk("docs/a.md", "", "t", "b")) == f"{base}docs/a.md"


def test_build_index_no_chunks_empty_index(embedder: Embedder) -> None:
    index = build_index([], embedder, model="m", corpus_ref=REF, source_url_base="u/")
    assert len(index) == 0
    assert index.vectors.shape == (0, FAKE_DIM)
