"""Search quality smoke with the real model and the real index (marker `model`).

Skipped when models/ or index/ is missing (the unit stage has no weights). The image build
runs the same questions in scripts/smoke_check.py, inside a container with no network.
Build the index locally: scripts/fetch_model.py, scripts/fetch_corpus.py, then
`python -m synthwerk_mascot.build` (see README).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from synthwerk_mascot.embedder import OnnxEmbedder, query_text
from synthwerk_mascot.index import VectorIndex
from synthwerk_mascot.platform.config import Settings

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "multilingual-e5-small"
INDEX_DIR = ROOT / "index"
QUESTIONS: dict[str, Any] = json.loads((ROOT / "scripts" / "smoke_questions.json").read_text())

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not (MODEL_DIR / "model.onnx").is_file() or not (INDEX_DIR / "manifest.json").is_file(),
        reason="real model or index missing",
    ),
]


@pytest.fixture(scope="module")
def real() -> tuple[OnnxEmbedder, VectorIndex]:
    embedder = OnnxEmbedder(MODEL_DIR)
    return embedder, VectorIndex.load(INDEX_DIR, expected_dim=embedder.dim)


def _search(real: tuple[OnnxEmbedder, VectorIndex], question: str) -> list[str]:
    embedder, index = real
    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    hits = index.search(
        embedder.embed([query_text(question)])[0],
        defaults.top_k,
        defaults.min_score,
        defaults.min_margin,
    )
    return [hit.chunk.path for hit in hits]


@pytest.mark.parametrize("case", QUESTIONS["answered"], ids=lambda c: c["question"])
def test_quality_question_best_passage_from_expected_doc(
    real: tuple[OnnxEmbedder, VectorIndex], case: dict[str, Any]
) -> None:
    paths = _search(real, case["question"])

    assert paths, "no passage found"
    assert paths[0] in case["paths"]


@pytest.mark.parametrize("question", QUESTIONS["unanswered"])
def test_quality_off_topic_question_nothing_found(
    real: tuple[OnnxEmbedder, VectorIndex], question: str
) -> None:
    assert _search(real, question) == []
