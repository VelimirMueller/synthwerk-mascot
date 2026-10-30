from pathlib import Path

import numpy as np
import pytest

from synthwerk_mascot.embedder import (
    ModelNotAvailableError,
    OnnxEmbedder,
    normalise,
    passage_text,
    query_text,
)
from tests.conftest import FAKE_DIM


def test_prefixes_e5_query_and_passage() -> None:
    assert query_text("Wo läuft prd?") == "query: Wo läuft prd?"
    assert passage_text("Infra > Hosts", "One VPS.") == "passage: Infra > Hosts\nOne VPS."


def test_embed_batch_one_normalised_row_per_text(model_dir: Path) -> None:
    vectors = OnnxEmbedder(model_dir).embed(["resend sends mails", "indigo accent colour"])
    assert vectors.shape == (2, FAKE_DIM)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_embed_padding_in_a_batch_does_not_change_the_vector(model_dir: Path) -> None:
    embedder = OnnxEmbedder(model_dir)
    alone = embedder.embed(["resend mails"])[0]
    padded = embedder.embed(["resend mails", "the public demo milestone ships the mascot"])[0]
    np.testing.assert_allclose(alone, padded, rtol=1e-5, atol=1e-6)


def test_embed_no_texts_empty_matrix(model_dir: Path) -> None:
    assert OnnxEmbedder(model_dir).embed([]).shape == (0, FAKE_DIM)


def test_embed_long_text_truncated_to_max_tokens(model_dir: Path) -> None:
    vectors = OnnxEmbedder(model_dir, max_tokens=4).embed(["mails " * 50])
    assert vectors.shape == (1, FAKE_DIM)


def test_model_files_missing_not_available(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match=r"model\.onnx, tokenizer\.json"):
        OnnxEmbedder(tmp_path)


def test_normalise_zero_row_stays_zero() -> None:
    out = normalise(np.array([[0.0, 0.0], [3.0, 4.0]]))
    expected = np.array([[0.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
