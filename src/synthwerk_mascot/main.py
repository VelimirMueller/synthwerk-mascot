import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from synthwerk_mascot import __version__, api
from synthwerk_mascot.embedder import Embedder, OnnxEmbedder
from synthwerk_mascot.index import VectorIndex
from synthwerk_mascot.platform import health
from synthwerk_mascot.platform.config import Settings, get_settings
from synthwerk_mascot.platform.health import CheckResult, ReadinessCheck
from synthwerk_mascot.platform.log import configure_logging, request_id_bound
from synthwerk_mascot.ratelimit import RateLimiter

logger = logging.getLogger("synthwerk_mascot")

HEALTH_PATHS = frozenset({"/healthz", "/readyz"})
# Accept a caller's X-Request-ID only when it is short and log-safe; otherwise mint one.
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")

Loader = Callable[[Settings], tuple[Embedder, VectorIndex]]


def load_model_and_index(settings: Settings) -> tuple[Embedder, VectorIndex]:
    """Both from local files that ship in the image. Nothing is downloaded."""
    embedder = OnnxEmbedder(settings.model_dir, threads=settings.ort_threads)
    index = VectorIndex.load(settings.index_dir, expected_dim=embedder.dim)
    return embedder, index


def _start_loading(app: FastAPI, settings: Settings, loader: Loader) -> threading.Thread:
    """Load in a thread, so /healthz answers and /readyz says `down` until the load ends."""

    def run() -> None:
        try:
            embedder, index = loader(settings)
        except Exception as exc:
            app.state.load_failed = True
            logger.error(
                "model or index failed to load",
                extra={"event.name": "mascot.load.failed", "error.type": type(exc).__name__},
            )
            return
        app.state.embedder, app.state.index = embedder, index
        logger.info(
            "model and index loaded",
            extra={
                "event.name": "mascot.load.done",
                "mascot.index.chunks": len(index),
                "mascot.corpus.ref": index.manifest.corpus_ref,
            },
        )

    thread = threading.Thread(target=run, name="mascot-load", daemon=True)
    thread.start()
    return thread


async def _model_ready(request: Request) -> CheckResult:
    if request.app.state.embedder is not None:
        return CheckResult("ok")
    return CheckResult("down", "load failed" if request.app.state.load_failed else "loading")


async def _index_ready(request: Request) -> CheckResult:
    index: VectorIndex | None = request.app.state.index
    if index is None:
        return CheckResult("down", "load failed" if request.app.state.load_failed else "loading")
    return CheckResult("ok") if len(index) else CheckResult("down", "index is empty")


def create_app(settings: Settings | None = None, *, loader: Loader | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        settings,
        {
            "service.name": "synthwerk-mascot",
            "service.version": __version__,
            "deployment.environment.name": settings.platform_env,
        },
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.loading = _start_loading(app, settings, loader or load_model_and_index)
        yield

    app = FastAPI(title="Synthwerk Mascot API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.commit = settings.commit
    app.state.embedder = None
    app.state.index = None
    app.state.load_failed = False
    app.state.limiter = RateLimiter(
        settings.rate_limit_per_client,
        settings.rate_limit_global,
        settings.rate_limit_window_s,
        max_clients=settings.rate_limit_max_clients,
    )
    app.state.readiness_checks = [
        ReadinessCheck("model", _model_ready),
        ReadinessCheck("index", _index_ready),
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        with request_id_bound(request_id):
            error: Exception | None = None
            try:
                response = await call_next(request)
            except Exception as exc:
                # Answer the 500 here, so it carries the request id and gets its log line.
                error = exc
                response = PlainTextResponse("Internal Server Error", status_code=500)
            response.headers["X-Request-ID"] = request_id
            route = request.scope.get("route")
            fields: dict[str, object] = {
                "event.name": "http.server.request",
                "http.request.method": request.method,
                "http.response.status_code": response.status_code,
                "synthwerk.duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            if route is not None:  # unmatched paths (404) get no route: never the raw path
                fields["http.route"] = route.path
            if error is not None:
                fields["error.type"] = type(error).__qualname__
                logger.error("request completed", extra=fields)
            else:
                level = logging.DEBUG if request.url.path in HEALTH_PATHS else logging.INFO
                logger.log(level, "request completed", extra=fields)
        return response

    app.include_router(health.router)
    app.include_router(api.router)
    return app
