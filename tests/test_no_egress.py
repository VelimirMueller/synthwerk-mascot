"""No egress, in process: the full path (load the model and the index, answer a question)
makes no outbound connection and no DNS lookup. scripts/docker-smoke.sh proves the same for
the image, in a container with no network at all."""

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synthwerk_mascot.main import create_app
from synthwerk_mascot.platform.config import Settings
from tests.conftest import wait_loaded


class EgressAttempt(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    attempts: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(args[:2]))
        raise EgressAttempt("network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield attempts


def test_no_egress_guard_blocks_a_connection(no_network: list[str]) -> None:
    with pytest.raises(EgressAttempt):
        socket.create_connection(("example.com", 443), timeout=1)
    assert no_network


def test_no_egress_load_and_answer_make_no_network_call(
    settings: Settings, no_network: list[str]
) -> None:
    with TestClient(create_app(settings)) as client:
        wait_loaded(client)
        ready = client.get("/readyz")
        answer = client.post("/v1/answers", json={"question": "where do uploads go"})

    assert ready.status_code == 200
    assert answer.status_code == 200
    assert answer.json()["found"] is True
    assert no_network == []
