"""POST /v1/answers, /healthz, /readyz, CORS and the request log line (blueprint
docs/conventions/python-service.md §2, §5, §7)."""

import asyncio
import io
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from starlette.requests import Request

from synthwerk_mascot.embedder import Embedder
from synthwerk_mascot.index import VectorIndex
from synthwerk_mascot.main import create_app
from synthwerk_mascot.platform.config import Settings
from synthwerk_mascot.platform.health import CheckResult, ReadinessCheck, _run
from synthwerk_mascot.platform.log import FORBIDDEN_KEYS, configure_logging
from tests.conftest import REF, wait_loaded

CONTRACT = yaml.safe_load((Path(__file__).parent / "contracts" / "health.yaml").read_text())
REQUIRED_LOG_FIELDS = {
    "timestamp",
    "level",
    "msg",
    "event.name",
    "service.name",
    "service.version",
    "deployment.environment.name",
}


def contract_errors(schema: str, body: object) -> list[str]:
    validator = Draft202012Validator(
        {"$ref": f"#/components/schemas/{schema}", "components": CONTRACT["components"]}
    )
    return [error.message for error in validator.iter_errors(body)]


def ask(client: TestClient, question: str, **headers: str) -> Any:
    return client.post("/v1/answers", json={"question": question}, headers=headers)


def blocked_loader(release: threading.Event) -> Any:
    def load(_: Settings) -> tuple[Embedder, VectorIndex]:
        release.wait(5)
        raise RuntimeError("stop")

    return load


# --- answers -----------------------------------------------------------------------------


def test_answer_question_passages_cite_source_with_link(client: TestClient) -> None:
    response = ask(client, "which colour is the accent of the palette")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["found"] is True
    best = body["passages"][0]
    assert best["source"] == {
        "path": "docs/design-system.md",
        "anchor": "colour",
        "url": f"https://example.test/blob/{REF}/docs/design-system.md#colour",
    }
    assert "indigo" in best["text"]
    assert best["title"] == "Design system > Colour"
    assert len(body["passages"]) <= 3


def test_answer_every_passage_has_a_source(client: TestClient) -> None:
    body = ask(client, "uploads storage quota").json()

    assert body["passages"]
    assert all(p["source"]["url"].startswith("https://") for p in body["passages"])


def test_answer_nothing_close_enough_found_false(settings: Settings) -> None:
    strict = settings.model_copy(update={"min_score": 0.999})
    with TestClient(create_app(strict)) as client:
        wait_loaded(client)
        body = ask(client, "chocolate cake recipe").json()

    assert body == {"found": False, "passages": []}


@pytest.mark.parametrize("question", ["", "   ", "x" * 301])
def test_answer_empty_or_too_long_question_422(client: TestClient, question: str) -> None:
    assert ask(client, question).status_code == 422


def test_answer_over_client_limit_429_with_retry_after(settings: Settings) -> None:
    limited = settings.model_copy(update={"rate_limit_per_client": 2})
    with TestClient(create_app(limited)) as client:
        wait_loaded(client)
        codes = [ask(client, "resend mails").status_code for _ in range(3)]
        refused = ask(client, "resend mails")

    assert codes == [200, 200, 429]
    assert int(refused.headers["retry-after"]) > 0


def test_answer_model_still_loading_503(settings: Settings) -> None:
    release = threading.Event()
    with TestClient(create_app(settings, loader=blocked_loader(release))) as client:
        response = ask(client, "resend mails")
        release.set()

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"


# --- health (§2) -------------------------------------------------------------------------


def test_healthz_process_runs_200_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


def test_readyz_model_and_index_loaded_200_ok(client: TestClient) -> None:
    body = client.get("/readyz").json()

    assert body["status"] == "ok"
    assert {c["name"]: c["status"] for c in body["checks"]} == {"model": "ok", "index": "ok"}
    assert contract_errors("Readiness", body) == []


def test_readyz_startup_not_complete_503_down(settings: Settings) -> None:
    release = threading.Event()
    with TestClient(create_app(settings, loader=blocked_loader(release))) as client:
        response = client.get("/readyz")
        release.set()

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "down"
    assert {c["detail"] for c in body["checks"]} == {"loading"}
    assert contract_errors("Readiness", body) == []


def test_readyz_load_failed_503_down_with_detail(settings: Settings, tmp_path: Path) -> None:
    broken = settings.model_copy(update={"index_dir": tmp_path})
    with TestClient(create_app(broken)) as client:
        wait_loaded(client)
        response = client.get("/readyz")
        answer = ask(client, "resend mails")

    assert response.status_code == 503
    assert {c["detail"] for c in response.json()["checks"]} == {"load failed"}
    assert answer.status_code == 503


def test_readyz_empty_index_down(
    settings: Settings, embedder: Embedder, loaded_index: VectorIndex
) -> None:
    empty = VectorIndex(loaded_index.vectors[:0], [], replace(loaded_index.manifest, count=0))
    with TestClient(create_app(settings, loader=lambda _: (embedder, empty))) as client:
        wait_loaded(client)
        body = client.get("/readyz").json()

    assert body["status"] == "down"
    assert {c["name"]: c.get("detail") for c in body["checks"]}["index"] == "index is empty"


@pytest.mark.parametrize(("behaviour", "detail"), [("slow", "timeout"), ("raises", "check failed")])
def test_readiness_check_slow_or_failing_down(behaviour: str, detail: str) -> None:
    async def run(_: Request) -> CheckResult:
        if behaviour == "raises":
            raise OSError("disk gone")
        await asyncio.sleep(1)
        return CheckResult("ok")

    check = asyncio.run(
        _run(ReadinessCheck("disk", run, timeout_s=0.01), Request({"type": "http"}))
    )

    assert (check.status, check.detail) == ("down", detail)


def test_openapi_has_health_and_answers_only(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/healthz", "/readyz", "/v1/answers"}


# --- CORS --------------------------------------------------------------------------------


def test_cors_allowed_origin_preflight_ok(client: TestClient) -> None:
    response = client.options(
        "/v1/answers",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_other_origin_preflight_refused(client: TestClient) -> None:
    response = client.options(
        "/v1/answers",
        headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 400


# --- logs (§5) ---------------------------------------------------------------------------


def _stdout_lines(settings: Settings, question: str) -> list[dict[str, Any]]:
    buffer = io.StringIO()
    with TestClient(create_app(settings)) as client:
        configure_logging(
            settings,
            {
                "service.name": "synthwerk-mascot",
                "service.version": "0.1.0",
                "deployment.environment.name": "local",
            },
            stream=buffer,
        )
        wait_loaded(client)
        ask(client, question, **{"X-Request-ID": "req-1"})
        client.get("/healthz")
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_log_line_answer_required_fields_no_question(settings: Settings) -> None:
    lines = _stdout_lines(settings, "where do uploads go secretword")

    line = next(entry for entry in lines if entry["msg"] == "request completed")
    assert line.keys() >= REQUIRED_LOG_FIELDS
    assert line["http.route"] == "/v1/answers"
    assert line["synthwerk.request_id"] == "req-1"
    assert not FORBIDDEN_KEYS & line.keys()
    assert all("secretword" not in json.dumps(entry) for entry in lines)


def test_log_line_health_requests_not_at_info(settings: Settings) -> None:
    lines = _stdout_lines(settings, "resend")

    assert not [e for e in lines if e.get("http.route") == "/healthz"]


def test_unhandled_error_500_with_request_id(settings: Settings) -> None:
    class Broken:
        dim = 32

        def embed(self, texts: Any) -> Any:
            raise RuntimeError("boom")

    def load(_: Settings) -> tuple[Embedder, VectorIndex]:
        return Broken(), VectorIndex.load(settings.index_dir)

    with TestClient(create_app(settings, loader=load), raise_server_exceptions=False) as client:
        wait_loaded(client)
        response = ask(client, "resend")

    assert response.status_code == 500
    assert response.headers["x-request-id"]


def test_settings_variable_without_service_prefix_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("SYNTHWERK_MASCOT_RATE_LIMIT_PER_CLIENT", "3")
    monkeypatch.setenv("SYNTHWERK_PLATFORM_ENV", "ci")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert (settings.top_k, settings.rate_limit_per_client, settings.platform_env) == (3, 3, "ci")


def test_console_format_only_in_local(settings: Settings) -> None:
    buffer = io.StringIO()
    prd = settings.model_copy(update={"platform_env": "prd", "log_format": "console"})
    configure_logging(prd, {"service.name": "s"}, stream=buffer)
    logging.getLogger("synthwerk_mascot").warning("x")

    assert json.loads(buffer.getvalue().splitlines()[-1])["msg"] == "x"
