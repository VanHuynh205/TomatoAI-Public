"""Inference accepts a single still image, including at the HTTP boundary."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor
from backend.tests.test_security import app_for


def test_predict_rejects_animated_png(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    with (
        Image.new("RGB", (16, 16), "red") as first,
        Image.new("RGB", (16, 16), "blue") as second,
        io.BytesIO() as stream,
    ):
        first.save(stream, "PNG", save_all=True, append_images=[second], duration=100)
        payload = stream.getvalue()

    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("animated.png", payload, "image/png")},
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image"
    assert predictor.calls == 0
