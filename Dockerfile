# syntax=docker/dockerfile:1.7
# The image holds everything the mascot needs: the code, the model files and the vector index.
# Only the build stage reaches the network (model from Hugging Face, docs from GitHub, both
# pinned). The running container needs no network at all.

FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
# Model: multilingual-e5-small ONNX at a pinned revision, checked by SHA-256.
RUN .venv/bin/python scripts/fetch_model.py
# Corpus: docs/*.md of VelimirMueller/synthwerk at a pinned commit. Bump it for new docs.
ARG CORPUS_REF=6d85efc565cae51492607b2c74b01305c7ba8f76
RUN .venv/bin/python scripts/fetch_corpus.py "$CORPUS_REF" --out /tmp/corpus \
 && .venv/bin/python -m synthwerk_mascot.build --corpus /tmp/corpus --ref "$CORPUS_REF" --out index

FROM python:3.14-slim
RUN useradd --system --uid 10001 --home /app app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY --from=build /app/models /app/models
COPY --from=build /app/index /app/index
COPY --from=build /app/README.md /app/LICENSE /app/
# Git commit shown by /readyz; the build passes --build-arg COMMIT=<sha>.
ARG COMMIT=0000000
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    SYNTHWERK_MASCOT_COMMIT=$COMMIT FORWARDED_ALLOW_IPS=127.0.0.1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
# One worker: the rate limit lives in process memory.
CMD ["uvicorn", "synthwerk_mascot.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--no-access-log"]
