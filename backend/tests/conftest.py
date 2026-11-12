"""Application fixtures contain synthetic predictions and synthetic care only."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai.contracts import Prediction, ScoredClass
from backend.care import CareCatalog
from backend.tests.test_contract import care_payload as care_payload  # noqa: PLC0414


class FakePredictor:
    class_ids: tuple[str, ...] = ("healthy", "early_blight")
    model_version = "fixture-model-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.result = Prediction(
            [ScoredClass("healthy", 0.9), ScoredClass("early_blight", 0.1)],
            "ok",
            "healthy",
            self.model_version,
        )

    def predict(self, image: Image.Image) -> Prediction:
        self.calls += 1
        assert image.mode == "RGB"
        return self.result


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def care_catalog(tmp_path: Path, care_payload: dict[str, Any]) -> CareCatalog:
    path = tmp_path / "care.json"
    cards = [
        {**care_payload, "class_id": label, "name_vi": name}
        for label, name in [
            ("healthy", "Lá khỏe theo nhãn dữ liệu"),
            ("early_blight", "Đốm vòng"),
        ]
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "knowledge_version": "fixture-care-v1",
                "artifact_kind": "fixture",
                "cards": cards,
            }
        ),
        encoding="utf-8",
    )
    return CareCatalog.load(path)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-Session-Token": "test-session", "Origin": "http://127.0.0.1:8765"}
