"""Download the embedding model into models/, verified by SHA-256.

Weights are not committed to git. The URLs are pinned to one revision and every file is checked.
The image build runs this script; the running mascot never downloads anything.

Usage: uv run python scripts/fetch_model.py
"""

import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}"

# path under models/ -> (url, sha256)
# model.onnx is the int8 copy the model authors publish (118 MB instead of 470 MB). It runs on
# any CPU (the file name only names the CPU it was tuned for) and needs about half the memory
# of the float32 file; the smoke questions get the same sources with both.
ARTIFACTS = {
    "multilingual-e5-small/model.onnx": (
        f"{BASE}/onnx/model_qint8_avx512_vnni.onnx",
        "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
    ),
    "multilingual-e5-small/tokenizer.json": (
        f"{BASE}/onnx/tokenizer.json",
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as fh:  # noqa: S310 - fixed https URLs
                shutil.copyfileobj(response, fh)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                sys.exit(f"download failed after {attempts} attempts: {url}: {exc}")
            time.sleep(2**attempt)


def fetch(name: str, url: str, expected: str) -> None:
    target = MODELS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected:
        print(f"ok        {name}")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    download(url, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {name}: expected {expected}, got {actual}")
    tmp.replace(target)
    print(f"fetched   {name}")


def main() -> None:
    for name, (url, digest) in ARTIFACTS.items():
        fetch(name, url, digest)


if __name__ == "__main__":
    main()
