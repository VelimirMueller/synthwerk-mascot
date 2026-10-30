"""Build the flat vector index from a corpus directory. Runs in the image build, offline.

Usage: python -m synthwerk_mascot.build --corpus corpus --ref <commit> --out index
"""

import argparse
import sys
from pathlib import Path

from synthwerk_mascot.corpus import load_corpus
from synthwerk_mascot.embedder import OnnxEmbedder
from synthwerk_mascot.index import build_index

MODEL = (
    "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
    "/onnx/model_qint8_avx512_vnni.onnx"
)
SOURCE_REPO = "https://github.com/VelimirMueller/synthwerk"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="holds docs/*.md")
    parser.add_argument("--ref", required=True, help="commit of the corpus, for the citations")
    parser.add_argument("--out", type=Path, default=Path("index"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/multilingual-e5-small"))
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    chunks = load_corpus(args.corpus)
    if not chunks:
        print(f"no chunks under {args.corpus}/docs/*.md", file=sys.stderr)
        return 1
    embedder = OnnxEmbedder(args.model_dir, threads=args.threads)
    index = build_index(
        chunks,
        embedder,
        model=MODEL,
        corpus_ref=args.ref,
        source_url_base=f"{SOURCE_REPO}/blob/{args.ref}/",
    )
    index.save(args.out)
    files = len({c.path for c in chunks})
    print(f"index: {len(index)} chunks from {files} files, dim {index.manifest.dim} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
