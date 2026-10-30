"""Shared fixtures. The fake model is a real ONNX graph and a real tokenizer.json, so the
tests run the same ONNX Runtime and tokenizer code as the image, without the 470 MB weights.

The fake encoder looks up one fixed random vector per word; the mean of those vectors is a
bag of words, so texts that share words score high. Good enough to test the plumbing.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from synthwerk_mascot.corpus import load_corpus
from synthwerk_mascot.embedder import Embedder, OnnxEmbedder
from synthwerk_mascot.index import VectorIndex, build_index
from synthwerk_mascot.main import create_app
from synthwerk_mascot.platform.config import Settings

FAKE_DIM = 32
REF = "0123456789abcdef0123456789abcdef01234567"

DOCS = {
    "infrastructure.md": """# Infrastructure

The demo runs on one server.

## Email: Resend

Resend sends the invite mails and the double opt-in mails from the EU region.

## Object storage

Uploads go to Supabase Storage, keyed per org, with a quota per org.
""",
    "design-system.md": """# Design system

<picture><img alt="banner" src="x.svg"></picture>

## Colour

The palette uses indigo as the accent colour and a neon magenta for highlights.

```text
+-----+  ascii art that the index leaves out
```
""",
    "roadmap.md": """# Roadmap

## Milestones

The public demo milestone ships the mascot, the public site and the live demo.
""",
}

WORDS_EXTRA = "query passage which who what where how does do the a is mails mail colour"


def write_corpus(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir(parents=True)
    for name, text in DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    return root


def _vocab(texts: list[str]) -> dict[str, int]:
    words = sorted({w for t in texts for w in re.findall(r"\w+", t.lower())})
    vocab = {"<pad>": 0, "[UNK]": 1}
    for word in words:
        vocab.setdefault(word, len(vocab))
    return vocab


def write_fake_model(directory: Path, texts: list[str], dim: int = FAKE_DIM) -> Path:
    """model.onnx (Gather from a seeded table) and a word-level tokenizer.json."""
    directory.mkdir(parents=True, exist_ok=True)
    vocab = _vocab([*texts, WORDS_EXTRA])
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))

    table = np.random.default_rng(7).normal(size=(len(vocab), dim)).astype(np.float32)
    table[0] = 0.0  # <pad>
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"])],
        "fake-e5",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"]),
            helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, ["b", "s"]),
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", dim])],
        initializer=[numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(directory / "model.onnx"))
    return directory


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fake_model(tmp_path_factory.mktemp("model"), list(DOCS.values()))


@pytest.fixture(scope="session")
def embedder(model_dir: Path) -> Embedder:
    return OnnxEmbedder(model_dir)


@pytest.fixture(scope="session")
def index_dir(
    tmp_path_factory: pytest.TempPathFactory, corpus_dir: Path, embedder: Embedder
) -> Path:
    index = build_index(
        load_corpus(corpus_dir),
        embedder,
        model="fake",
        corpus_ref=REF,
        source_url_base=f"https://example.test/blob/{REF}/",
    )
    out = tmp_path_factory.mktemp("index")
    index.save(out)
    return out


@pytest.fixture
def settings(model_dir: Path, index_dir: Path) -> Settings:
    # The fake bag-of-words scores are not e5 scores: no threshold, no margin.
    return Settings(
        model_dir=model_dir,
        index_dir=index_dir,
        min_score=0.0,
        min_margin=0.0,
        rate_limit_per_client=50,
        rate_limit_global=500,
        _env_file=None,  # type: ignore[call-arg]
    )


def wait_loaded(client: TestClient) -> None:
    client.app.state.loading.join(timeout=10)  # type: ignore[attr-defined]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        wait_loaded(test_client)
        yield test_client


@pytest.fixture
def loaded_index(index_dir: Path) -> VectorIndex:
    return VectorIndex.load(index_dir, expected_dim=FAKE_DIM)
