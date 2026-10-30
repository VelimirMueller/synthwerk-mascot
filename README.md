# synthwerk-mascot

The public helper of [Synthwerk](https://github.com/VelimirMueller/synthwerk). It answers questions about Synthwerk without sign-in. Each answer is a set of passages from the public docs, and each passage cites its source.

The mascot is isolated from the rest of Synthwerk: no database, no tools, no event bus, no outbound network. The model and the index are in the image.

## Status

- State: working, v0.1.0. Not deployed: the mascot has no host yet ([synthwerk#48](https://github.com/VelimirMueller/synthwerk/issues/48)).
- Answer mode: retrieval only. The mascot returns the best passages of the docs. No language model writes text.
- Corpus: `docs/*.md` of `VelimirMueller/synthwerk` at a pinned commit (`CORPUS_REF` in the `Dockerfile`).
- Model: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (MIT), the int8 ONNX file of the model authors, on ONNX Runtime (CPU). 384 dimensions.
- Image: about 490 MB unpacked (130 MB model files). Memory: about 470 MiB.
- Name: "Strobe" is a proposal, not final.

## What it does

- `POST /v1/answers` with `{"question": "..."}` answers with up to 3 passages: title, text, score and source (`path`, `anchor`, `url`).
- `found: false` and no passages: nothing in the docs is close enough to the question.
- German and English questions work. The docs are in English.
- Strict rate limit: 6 questions per client per minute and 60 for all clients together (both can be set). Over the limit: `429` with `Retry-After`.
- No user data is stored. The question is not logged. The rate limit keeps a keyed hash of the client address in memory only.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/answers \
  -H 'Content-Type: application/json' \
  -d '{"question": "Welcher Anbieter verschickt E-Mails?"}'
```

```json
{
  "found": true,
  "passages": [
    {
      "title": "Infrastructure > Decisions of 2026-10-29 > Email: Resend",
      "text": "- Sends invites (#37) and double opt-in mails (#46). ...",
      "score": 0.796,
      "source": {
        "path": "docs/infrastructure.md",
        "anchor": "email-resend",
        "url": "https://github.com/VelimirMueller/synthwerk/blob/<commit>/docs/infrastructure.md#email-resend"
      }
    }
  ]
}
```

## How it works

1. The image build downloads the model (pinned revision, SHA-256 checked) and the docs (pinned commit).
2. The build splits each doc at its headings into chunks of at most 1,200 characters. Pictures, HTML and `text` diagrams are left out.
3. The build embeds each chunk with the prefix `passage: ` and writes the index: `index/vectors.f32` (one normalised float32 row per chunk), `index/chunks.json` (the source of each row) and `index/manifest.json` (model, corpus commit, SHA-256 of the vectors).
4. At start, the service loads the model and the index from the image. `/readyz` answers `503` until both are loaded.
5. For a question, the service embeds `query: <question>` and does an exact cosine search in memory (one matrix-vector product). It keeps one passage per section.
6. Two gates stop answers to off-topic questions: the best score must be at least `MIN_SCORE` (0.78), and it must beat the median score of all chunks by `MIN_MARGIN` (0.045).

A new corpus needs a new image: change `CORPUS_REF` and build again.

## Isolation

- The running container needs no network. `scripts/docker-smoke.sh` starts it with `--network none`, a read-only root file system and no capabilities, and checks inside it that a TCP connection and a DNS lookup fail.
- `tests/test_no_egress.py` blocks every socket call in the test process and proves that loading and answering make none.
- ONNX Runtime telemetry is switched off. `HF_HUB_OFFLINE=1` is set.
- No OpenTelemetry exporter: it would be outbound traffic. Logs are JSON on stdout only.
- The edge reaches the mascot over an internal network with no route out. That wiring is in `synthwerk-infra` (not done yet).

## Run locally

Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/fetch_model.py                      # 135 MB into models/
uv run python scripts/fetch_corpus.py <synthwerk commit>   # docs/*.md into corpus/
uv run python -m synthwerk_mascot.build --corpus corpus --ref <synthwerk commit>
uv run uvicorn synthwerk_mascot.main:create_app --factory --port 8000
```

With Docker:

```bash
docker build -t synthwerk-mascot .
docker run --rm -p 127.0.0.1:8000:8000 synthwerk-mascot
scripts/docker-smoke.sh   # build, run with --network none, check inside the container
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `SYNTHWERK_PLATFORM_ENV` | `local` | `local`, `ci`, `dev`, `stg`, `prd` |
| `SYNTHWERK_MASCOT_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `SYNTHWERK_MASCOT_LOG_FORMAT` | `json` | `json` or `console` (local only) |
| `SYNTHWERK_MASCOT_CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8080"]` | Browser origins that may call `POST /v1/answers` |
| `SYNTHWERK_MASCOT_MODEL_DIR` | `models/multilingual-e5-small` | `model.onnx` and `tokenizer.json` |
| `SYNTHWERK_MASCOT_INDEX_DIR` | `index` | The index files |
| `SYNTHWERK_MASCOT_ORT_THREADS` | `2` | ONNX Runtime threads; `0` = default |
| `SYNTHWERK_MASCOT_TOP_K` | `3` | Most passages in one answer |
| `SYNTHWERK_MASCOT_MIN_SCORE` | `0.78` | Lowest cosine score of a passage |
| `SYNTHWERK_MASCOT_MIN_MARGIN` | `0.045` | Best score minus median score, at least |
| `SYNTHWERK_MASCOT_MAX_QUESTION_CHARS` | `300` | Longer questions get `422` |
| `SYNTHWERK_MASCOT_RATE_LIMIT_PER_CLIENT` | `6` | Questions per client per window |
| `SYNTHWERK_MASCOT_RATE_LIMIT_GLOBAL` | `60` | Questions of all clients per window |
| `SYNTHWERK_MASCOT_RATE_LIMIT_WINDOW_S` | `60` | Window in seconds |
| `SYNTHWERK_MASCOT_RATE_LIMIT_MAX_CLIENTS` | `10000` | Clients tracked at once; past that, new clients get `429` |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | uvicorn: the proxy whose `X-Forwarded-For` names the client. Set it to the edge address |

`GET /healthz` is the liveness probe. `GET /readyz` is the readiness probe, with the checks `model` and `index`.

## Tests

| Command | What it covers |
| --- | --- |
| `uv run pytest` | Chunking, index build and load, search, rate limit, API, health contract, logs, no egress in process. A small real ONNX graph stands in for the model |
| `uv run pytest -m model` | Search quality with the real model and index (German and English questions, `scripts/smoke_questions.json`). Skipped without `models/` and `index/` |
| `scripts/docker-smoke.sh` | The image with no network: health, search quality, no egress, rate limit |

## CI and release

- `ci.yml` on every PR: lint (ruff, mypy), unit (pytest, coverage), build (the Docker smoke).
- `deploy.yml`, by hand on `develop` only, publishes `ghcr.io/velimirmueller/synthwerk-mascot:<sha>`. No host yet, so it deploys nothing.
- `promote.yml` moves a verified SHA to `stage` and `main`.

## License

[MIT](LICENSE). The model `multilingual-e5-small` is MIT as well.
