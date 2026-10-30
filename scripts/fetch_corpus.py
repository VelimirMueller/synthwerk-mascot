"""Download the public corpus: docs/*.md of VelimirMueller/synthwerk at one pinned commit.

The image build runs this script; the running mascot never downloads anything. A commit SHA
names the content exactly, so the same ref gives the same corpus.

Usage: uv run python scripts/fetch_corpus.py <full commit sha> [--out corpus]
"""

import argparse
import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "VelimirMueller/synthwerk"
_SHA = re.compile(r"[0-9a-f]{40}")
_DOC = re.compile(r"^[^/]+/docs/[^/]+\.md$")  # top-level docs only, like docs/*.md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="full 40-character commit SHA of synthwerk")
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    if not _SHA.fullmatch(args.ref):
        sys.exit("ref must be a full 40-character commit SHA")

    url = f"https://codeload.github.com/{REPO}/tar.gz/{args.ref}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    docs = args.out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _DOC.match(member.name):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            (docs / Path(member.name).name).write_bytes(source.read())
            count += 1
    if count == 0:
        sys.exit(f"no docs/*.md in {REPO}@{args.ref}")
    print(f"corpus: {count} files from {REPO}@{args.ref} -> {docs}")


if __name__ == "__main__":
    main()
