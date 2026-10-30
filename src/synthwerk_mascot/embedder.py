"""Text embeddings with `intfloat/multilingual-e5-small` on ONNX Runtime (CPU).

e5 wants a prefix on every input: `query: ` for a question, `passage: ` for a document part.
Without it the results get worse. The vector is the mean of the token states (attention mask
applied), normalised to length 1, so cosine similarity is a dot product.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MAX_TOKENS = 512

Vectors = npt.NDArray[np.float32]


class ModelNotAvailableError(RuntimeError):
    """The model files are missing. The image ships them; a plain checkout must fetch them."""


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Vectors:
        """One normalised float32 row per text. The caller adds the e5 prefix."""
        ...


def query_text(question: str) -> str:
    return QUERY_PREFIX + question


def passage_text(title: str, text: str) -> str:
    return f"{PASSAGE_PREFIX}{title}\n{text}"


def normalise(vectors: npt.NDArray[np.floating]) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out: Vectors = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
    return out


class OnnxEmbedder:
    """Mean-pooled, normalised sentence vectors from an ONNX encoder and its tokenizer.json."""

    def __init__(self, model_dir: Path, threads: int = 0, max_tokens: int = MAX_TOKENS) -> None:
        model, tokenizer_file = model_dir / "model.onnx", model_dir / "tokenizer.json"
        missing = [p.name for p in (model, tokenizer_file) if not p.is_file()]
        if missing:
            raise ModelNotAvailableError(f"missing in {model_dir}: {', '.join(missing)}")
        # ONNX Runtime has platform telemetry. The container has no network anyway; this also
        # stops the attempt (and its warning about a read-only file system).
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._dim = int(self._session.get_outputs()[0].shape[-1])
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_tokens)
        pad_id = self._tokenizer.token_to_id("<pad>")
        self._tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        encodings = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        (states,) = self._session.run(None, feeds)[:1]
        weights = mask[:, :, None].astype(np.float32)
        pooled = (states * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        return normalise(pooled)
