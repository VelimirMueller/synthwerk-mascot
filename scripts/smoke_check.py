"""Checks inside the mascot container (stdlib only). scripts/docker-smoke.sh runs it with
`docker exec` in a container started with `--network none`.

1. /healthz answers 200 and /readyz answers 200 (model and index loaded from the image).
2. Search quality: every question in smoke_questions.json gets its expected source first;
   every off-topic question finds nothing. Each passage cites a source URL.
3. No egress: a TCP connection to a public address and a DNS lookup both fail.
4. Rate limit: the client gets 429 with Retry-After once its window is full.
"""

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "http://127.0.0.1:8000"
QUESTIONS: dict[str, Any] = json.loads(
    (Path(__file__).with_name("smoke_questions.json")).read_text(encoding="utf-8")
)
failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("ok    " if ok else "FAIL  ") + message)
    if not ok:
        failures.append(message)


def get(path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as response:  # noqa: S310 - loopback
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def ask(question: str) -> tuple[int, Any, dict[str, str]]:
    request = urllib.request.Request(  # noqa: S310 - loopback
        BASE + "/v1/answers",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, None, dict(exc.headers)


def wait_ready(seconds: int = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if get("/readyz")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)


def main() -> int:
    wait_ready()
    code, body = get("/healthz")
    check(code == 200 and body == {"status": "ok"}, "/healthz 200 ok")
    code, body = get("/readyz")
    if code != 200 or not isinstance(body, dict):
        check(False, f"/readyz answered {code} within 60 s: {body}")
        return 1
    check(body["status"] == "ok", f"/readyz 200 ok (commit {body['commit']})")

    for case in QUESTIONS["answered"]:
        code, body, _ = ask(case["question"])
        paths = [p["source"]["path"] for p in body["passages"]] if code == 200 else []
        cited = code == 200 and all(p["source"]["url"] for p in body["passages"])
        check(
            bool(paths) and paths[0] in case["paths"] and cited,
            f"quality: {case['question']!r} -> {paths[:1]} (want one of {case['paths']})",
        )
    for question in QUESTIONS["unanswered"]:
        code, body, _ = ask(question)
        check(code == 200 and body["found"] is False, f"quality: {question!r} -> nothing found")

    for host, port in (("1.1.1.1", 443), ("140.82.112.3", 443), ("8.8.8.8", 53)):
        try:
            socket.create_connection((host, port), timeout=3).close()
            check(False, f"no egress: connection to {host}:{port} must fail")
        except OSError as exc:
            check(True, f"no egress: {host}:{port} refused ({type(exc).__name__})")
    try:
        socket.getaddrinfo("huggingface.co", 443)
        check(False, "no egress: DNS lookup must fail")
    except OSError as exc:
        check(True, f"no egress: DNS lookup refused ({type(exc).__name__})")

    refused = None
    for _ in range(200):
        code, _, headers = ask("rate limit probe")
        if code == 429:
            refused = headers
            break
    retry = {k.lower(): v for k, v in (refused or {}).items()}.get("retry-after", "")
    check(refused is not None and retry.isdigit(), f"rate limit: 429 with Retry-After {retry}")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
