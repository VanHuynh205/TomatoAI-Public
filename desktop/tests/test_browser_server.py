"""The real smoke label must never accept fixture weights."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from desktop.tests import browser_server
from tests.ai.test_yolo import yolo_bundle as yolo_bundle  # noqa: PLC0414


def test_fixture_bundle_mode_uses_yolo_runtime(
    yolo_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOMATO_E2E_MODEL_DIR", str(yolo_bundle))
    monkeypatch.setenv(
        "TOMATO_E2E_CARE_PATH",
        str(browser_server.ROOT / "infra/knowledge/care-v1.json"),
    )
    app = browser_server.create_browser_app(
        "fixture-bundle", "test-session", "http://127.0.0.1:8765"
    )
    assert app.state.runtime.ready
    assert app.state.runtime.predictor.model_version == "yolo-fixture-v1"


def test_real_mode_rejects_fixture_model(
    yolo_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOMATO_E2E_MODEL_DIR", str(yolo_bundle))
    monkeypatch.setenv("TOMATO_E2E_CARE_PATH", "unused-care.json")
    composition = Mock(
        return_value=SimpleNamespace(
            state=SimpleNamespace(
                runtime=SimpleNamespace(
                    catalog=SimpleNamespace(artifact_kind="curated")
                )
            )
        )
    )
    monkeypatch.setattr(browser_server, "create_desktop_app", composition)
    with pytest.raises(ValueError, match="fixture bundle forbidden"):
        browser_server.create_browser_app(
            "real", "test-session", "http://127.0.0.1:8765"
        )
    composition.assert_not_called()
