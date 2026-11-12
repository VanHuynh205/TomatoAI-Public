"""All API routes enforce the same local session boundary before body parsing."""

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor


def app_for(predictor: FakePredictor, catalog: CareCatalog, **kwargs: Any) -> FastAPI:
    from backend.app import create_app

    return create_app(
        predictor=predictor,
        care_catalog=catalog,
        session_token=kwargs.get("token", "test-session"),
        allowed_origin=kwargs.get("origin", "http://127.0.0.1:8765"),
    )


def test_health_care_and_tooling_schema(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    from backend.app import build_openapi

    app = app_for(predictor, care_catalog)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/v1/health").status_code == 401
        health = client.get("/api/v1/health", headers=auth)
        assert health.json() == {
            "schema_version": "1",
            "ready": True,
            "model_version": "fixture-model-v1",
            "knowledge_version": "fixture-care-v1",
            "reason": None,
        }
        assert (
            client.get(
                "/api/v1/health", headers={"X-Session-Token": "test-session"}
            ).status_code
            == 200
        )
        card = client.get("/api/v1/care/healthy", headers=auth)
        assert card.json()["class_id"] == "healthy"
        assert card.json()["knowledge_version"] == "fixture-care-v1"
        absent = client.get("/api/v1/care/no_such_class", headers=auth)
        assert (
            absent.status_code == 404
            and absent.json()["error"]["code"] == "care_not_found"
        )
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        assert client.get("/api/v1/unknown", headers=auth).status_code == 404
    assert build_openapi() == app.openapi()
    assert predictor.calls == 0


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/v1/health", "GET"),
        ("/api/v1/care/healthy", "GET"),
        ("/api/v1/predict", "POST"),
        ("/api/v1/unknown", "GET"),
    ],
)
@pytest.mark.parametrize(
    "mutation,status,code",
    [
        ("token", 401, "unauthorized"),
        ("missing", 401, "unauthorized"),
        ("origin", 403, "forbidden_origin"),
        ("host", 403, "forbidden_host"),
        ("duplicate", 401, "unauthorized"),
    ],
)
def test_api_boundary(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    path: str,
    method: str,
    mutation: str,
    status: int,
    code: str,
) -> None:
    headers = dict(auth)
    if mutation == "token":
        headers["X-Session-Token"] = "old-session"
    elif mutation == "missing":
        del headers["X-Session-Token"]
    elif mutation == "origin":
        headers["Origin"] = "https://evil.test"
    elif mutation == "host":
        headers["Host"] = "evil.test"
    pairs = list(headers.items())
    if mutation == "duplicate":
        pairs.append(("X-Session-Token", "another"))
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.request(method, path, headers=pairs)
    assert response.status_code == status
    assert response.json()["schema_version"] == "1"
    assert response.json()["error"]["code"] == code
    assert "test-session" not in response.text and "old-session" not in response.text
    assert predictor.calls == 0


def test_post_requires_origin(
    predictor: FakePredictor, care_catalog: CareCatalog
) -> None:
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict", headers={"X-Session-Token": "test-session"}
        )
    assert (
        response.status_code == 403
        and response.json()["error"]["code"] == "forbidden_origin"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token": ""},
        {"token": " "},
        {"origin": "https://evil.test:8765"},
        {"origin": "http://127.0.0.1"},
        {"origin": "http://127.0.0.1:0"},
        {"origin": "http://user:pass@127.0.0.1:8765"},
        {"origin": "http://127.0.0.1:8765/path"},
    ],
)
def test_invalid_factory_boundary(
    predictor: FakePredictor, care_catalog: CareCatalog, kwargs: dict[str, str]
) -> None:
    with pytest.raises(ValueError):
        app_for(predictor, care_catalog, **kwargs)


def test_missing_card_prevents_startup(
    predictor: FakePredictor, care_catalog: CareCatalog
) -> None:
    predictor.class_ids = ("healthy", "new_class")
    with pytest.raises(ValueError, match="missing care"):
        app_for(predictor, care_catalog)


def test_openapi_cli_does_not_import_torch_or_model(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    script = "import sys; from backend.openapi import main; assert main(['--output',sys.argv[1]])==0; assert 'torch' not in sys.modules; assert 'ai.inference' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/api/v1/health" in json.loads(output.read_text())["paths"]


@pytest.mark.parametrize(
    "headers",
    [
        [
            ("X-Session-Token", "test-session"),
            ("Origin", "http://127.0.0.1:8765"),
            ("Origin", "http://127.0.0.1:8765"),
        ],
        [
            ("X-Session-Token", "test-session"),
            ("Host", "127.0.0.1:8765"),
            ("Host", "evil.test"),
        ],
    ],
)
def test_duplicate_origin_or_host_is_rejected(
    predictor: FakePredictor, care_catalog: CareCatalog, headers: list[tuple[str, str]]
) -> None:
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.get("/api/v1/health", headers=headers)
    assert response.status_code == 403


def test_rejected_session_never_reads_upload(
    predictor: FakePredictor, care_catalog: CareCatalog
) -> None:
    app = app_for(predictor, care_catalog)

    async def check() -> None:
        async def forbidden_body() -> AsyncIterator[bytes]:
            pytest.fail("unauthorized request body was consumed")
            yield b""

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            response = await client.post("/api/v1/predict", content=forbidden_body())
            assert response.status_code == 401

    asyncio.run(check())


def test_internal_error_and_validation_never_echo_input(
    predictor: FakePredictor, care_catalog: CareCatalog
) -> None:
    app = app_for(predictor, care_catalog)

    @app.get("/test-validation")
    async def validation(count: int) -> int:
        return count

    @app.get("/test-error")
    async def fail() -> None:
        raise RuntimeError("C:/private/path secret-token")

    with TestClient(
        app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False
    ) as client:
        for path, status, code in [
            ("/test-error", 500, "internal_error"),
            ("/test-validation?count=secret-token", 422, "invalid_input"),
        ]:
            response = client.get(path)
            assert response.status_code == status
            assert response.json()["error"]["code"] == code
            assert (
                "secret-token" not in response.text and "private" not in response.text
            )
