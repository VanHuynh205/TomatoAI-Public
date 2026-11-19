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


def test_create_desktop_app_routes_cnn_gru_before_dinov3(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
) -> None:
    server = importlib.import_module("desktop.server")
    dinov3_mod = importlib.import_module("ai.dinov3_inference")
    model_dir = assets.parent / "cnn-gru-bundle"
    model_dir.mkdir()
    (model_dir / "leaf-classifier.json").write_text("{}", encoding="utf-8")
    (model_dir / "cnn-gru-release.json").write_text("{}", encoding="utf-8")
    cnn_gru = Mock(return_value=predictor)
    dinov3 = Mock()
    legacy = Mock()
    monkeypatch.setattr(server, "CnnGruPredictor", cnn_gru)
    monkeypatch.setattr(dinov3_mod, "DinoV3Predictor", dinov3)
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
    cnn_gru.assert_called_once_with(model_dir)
    dinov3.assert_not_called()
    legacy.assert_not_called()


def test_browser_server_real_mode_serves_cnn_gru_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import io
    import json

    from PIL import Image
    from safetensors.torch import save_file

    from ai.cnn_gru_inference import CNN_GRU_MANIFEST, EXPORT_LIMITS
    from ai.cnn_gru_model import IMAGENET_MEAN, IMAGENET_STD, build_cnn_gru_model
    from ai.dinov3_field import CLASSES
    from desktop.tests import browser_server

    model_dir = tmp_path / "cnn-gru-real"
    model_dir.mkdir()
    model = build_cnn_gru_model(
        "resnet50", len(CLASSES), use_gru_head=False, hidden_dim=16, proj_dim=32
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        model_dir / "model.safetensors",
    )
    (model_dir / "leaf-classifier.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "resnet50",
                "use_gru_head": False,
                "hidden_dim": 16,
                "proj_dim": 32,
                "model_version": "cnn-gru-e2e-v1",
                "class_ids": list(CLASSES),
                "size": 288,
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "test_time_horizontal_flip": True,
                "temperature": 1.0,
                "threshold": 0.0,
                "artifact_kind": "trained",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / CNN_GRU_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "cnn-gru-e2e-v1",
                "architecture": "resnet50",
                "evaluation_size": 288,
                "temperature": 1.0,
                "threshold": 0.0,
                "artifact_kind": "trained",
                "sha256": {
                    name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                    for name in EXPORT_LIMITS
                },
            }
        ),
        encoding="utf-8",
    )

    care_path = browser_server.ROOT / "infra/knowledge/care-v1.json"
    monkeypatch.setenv("TOMATO_E2E_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("TOMATO_E2E_CARE_PATH", str(care_path))
    origin = "http://127.0.0.1:8769"
    token = "browser-test-token-123"
    app = browser_server.create_browser_app("real", token, origin)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (30, 120, 40)).save(buf, format="PNG")
    headers = {"X-Session-Token": token, "Origin": origin}
    with TestClient(app, base_url=origin) as client:
        health = client.get("/api/v1/health", headers=headers)
        assert health.status_code == 200
        assert health.json()["model_version"] == "cnn-gru-e2e-v1"
        predict = client.post(
            "/api/v1/predict",
            headers=headers,
            files={"image": ("leaf.png", buf.getvalue(), "image/png")},
        )
        assert predict.status_code == 200
        assert predict.json()["model_version"] == "cnn-gru-e2e-v1"

    # Explicit cnn-gru mode enforces CNN_GRU_MANIFEST and TOMATO_E2E_REQUIRE_TRAINED=1
    monkeypatch.setenv("TOMATO_E2E_REQUIRE_TRAINED", "1")
    app_cnn_gru = browser_server.create_browser_app("cnn-gru", token, origin)
    assert app_cnn_gru.state.runtime.predictor.model_version == "cnn-gru-e2e-v1"
    with pytest.raises(ValueError, match="requires fixture weights"):
        browser_server.create_browser_app("fixture-bundle", token, origin)

    non_cnn_gru_dir = tmp_path / "empty-bundle"
    non_cnn_gru_dir.mkdir()
    monkeypatch.setenv("TOMATO_E2E_MODEL_DIR", str(non_cnn_gru_dir))
    with pytest.raises(ValueError, match="CNN-GRU"):
        browser_server.create_browser_app("cnn-gru", token, origin)


@pytest.mark.parametrize("manifest", ["dinov2-release.json", "cnn-gru-release.json"])
def test_dinov3_harness_rejects_other_leaf_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: str
) -> None:
    from desktop.tests import browser_server

    (tmp_path / "leaf-classifier.json").write_text("{}", encoding="utf-8")
    (tmp_path / manifest).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TOMATO_E2E_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TOMATO_E2E_CARE_PATH", "unused-care.json")
    composition = Mock(side_effect=AssertionError("unexpected predictor load"))
    monkeypatch.setattr(browser_server, "create_desktop_app", composition)

    with pytest.raises(ValueError, match="requires an explicit DINOv3 leaf export"):
        browser_server.create_browser_app(
            "dinov3", "test-session", "http://127.0.0.1:8765"
        )
    composition.assert_not_called()


def test_missing_care_stops_composition(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
) -> None:
    predictor.class_ids = (*predictor.class_ids, "missing")
    with pytest.raises(ValueError, match="missing care"):
        compose(monkeypatch, assets, predictor, care_catalog)
