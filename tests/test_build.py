import json
from pathlib import Path

from synthwerk_mascot.build import main
from synthwerk_mascot.index import MANIFEST_FILE, VectorIndex
from tests.conftest import FAKE_DIM, REF


def test_build_cli_corpus_to_index_files(corpus_dir: Path, model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "index"

    code = main(
        [
            "--corpus",
            str(corpus_dir),
            "--ref",
            REF,
            "--out",
            str(out),
            "--model-dir",
            str(model_dir),
        ]
    )

    assert code == 0
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["source_url_base"] == f"https://github.com/VelimirMueller/synthwerk/blob/{REF}/"
    assert manifest["model"].startswith("intfloat/multilingual-e5-small@")
    assert len(VectorIndex.load(out, expected_dim=FAKE_DIM)) == manifest["count"]


def test_build_cli_empty_corpus_fails(model_dir: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--corpus",
            str(tmp_path),
            "--ref",
            REF,
            "--out",
            str(tmp_path / "i"),
            "--model-dir",
            str(model_dir),
        ]
    )

    assert code == 1
