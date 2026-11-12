"""Composition uses the existing API and serves only the frontend directory."""

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    directory = tmp_path / "frontend"
    directory.mkdir()
    (directory / "index.html").write_text("<h1>Tomato fixture</h1>", encoding="utf-8")
    (directory / "app.js").write_text("console.log('fixture')", encoding="utf-8")
    (tmp_path / "model.pth").write_bytes(b"private model")
    return directory


def compose(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    catalog: CareCatalog,
) -> tuple[Any, Mock]:
    server = importlib.import_module("desktop.server")
    model = Mock(return_value=predictor)
    monkeypatch.setattr(server, "Predictor", model)
    monkeypatch.setattr(server.CareCatalog, "load", Mock(return_value=catalog))
    return (
        server.create_desktop_app(
            assets,
            assets.parent / "model",
            assets.parent / "care.json",
            "test-session",
            "http://127.0.0.1:8765",
        ),
        model,
    )


def test_static_does_not_shadow_api(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
) -> None:
    hidden_api = assets / "api" / "v1"
    hidden_api.mkdir(parents=True)
    (hidden_api / "unknown").write_text("must not serve", encoding="utf-8")
    app, model = compose(monkeypatch, assets, predictor, care_catalog)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/").text == "<h1>Tomato fixture</h1>"
        assert client.get("/app.js").status_code == 200
        assert client.get("/api/v1/health").status_code == 401
        health = client.get("/api/v1/health", headers=auth)
        assert health.status_code == 200
        assert health.json()["ready"] is True
        assert health.json()["model_version"] == "fixture-model-v1"
        for method in ("GET", "POST", "PROPFIND"):
            response = client.request(method, "/api/v1/unknown", headers=auth)
            assert response.status_code == 404
            assert response.headers["content-type"] == "application/json"
        assert client.get("/api/v1/api/v1/health", headers=auth).status_code == 404
        assert (
            client.get(
                "/api/v1/health", headers={**auth, "X-Session-Token": "old"}
            ).status_code
            == 401
        )
        for path in ("/model.pth", "/care.json", "/../model.pth", "/%2e%2e/model.pth"):
            assert client.get(path).status_code == 404
    model.assert_called_once_with(assets.parent / "model", device="cpu")


def test_missing_assets_fail_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
) -> None:
    (assets / "index.html").unlink()
    with pytest.raises(ValueError, match="index"):
        compose(monkeypatch, assets, predictor, care_catalog)


def test_explicit_dinov3_export_uses_candidate_predictor(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
) -> None:
    server = importlib.import_module("desktop.server")
    module = importlib.import_module("ai.dinov3_inference")
    model_dir = assets.parent / "candidate"
    model_dir.mkdir()
    (model_dir / "leaf-classifier.json").write_text("{}", encoding="utf-8")
    candidate = Mock(return_value=predictor)
    legacy = Mock()
    monkeypatch.setattr(module, "DinoV3Predictor", candidate)
    monkeypatch.setattr(server, "Predictor", legacy)
    monkeypatch.setattr(server.CareCatalog, "load", Mock(return_value=care_catalog))

    origin = "http://127.0.0.1:8765"
    app = server.create_desktop_app(
        assets, model_dir, assets.parent / "care.json", "test-session", origin
    )
    with TestClient(app, base_url=origin) as client:
        health = client.get("/api/v1/health", headers=auth)
        assert health.status_code == 200
        assert health.json()["model_version"] == predictor.model_version
    candidate.assert_called_once_with(model_dir)
    legacy.assert_not_called()


def test_missing_care_stops_composition(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
) -> None:
    predictor.class_ids = (*predictor.class_ids, "missing")
    with pytest.raises(ValueError, match="missing care"):
        compose(monkeypatch, assets, predictor, care_catalog)
