"""Slot ownership survives client cancellation and model labels never use indexes."""

import asyncio
import io
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai.contracts import Prediction, ScoredClass
from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor
from backend.tests.test_predict import encoded, multipart
from backend.tests.test_security import app_for
from tests.ai.test_yolo import yolo_bundle as yolo_bundle  # noqa: PLC0414


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "EXIF"])
def test_yolo_bundle_predictor_api_tracer(
    yolo_bundle: Path, mode: str, auth: dict[str, str]
) -> None:
    from ai.inference import Predictor
    from backend.app import create_app
    from backend.images import decode_image

    predictor = Predictor(yolo_bundle)
    root = Path(__file__).resolve().parents[2]
    catalog = CareCatalog.load(root / "infra/knowledge/care-v1.json")
    app = create_app(
        predictor=predictor,
        care_catalog=catalog,
        session_token="test-session",
        allowed_origin="http://127.0.0.1:8765",
    )
    image = Image.new("RGB" if mode == "EXIF" else mode, (997, 43))
    buffer = io.BytesIO()
    if mode == "EXIF":
        exif = image.getexif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)
    else:
        image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    with decode_image(payload) as decoded:
        expected = predictor.predict(decoded)
    mime, suffix = ("image/jpeg", "jpg") if mode == "EXIF" else ("image/png", "png")
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        health = client.get("/api/v1/health", headers=auth).json()
        assert health["ready"] and health["model_version"] == "yolo-fixture-v1"
        response = client.post(
            "/api/v1/predict",
            files={"image": (f"leaf.{suffix}", payload, mime)},
            headers=auth,
        )
        assert response.status_code == 200, response.text
        actual = response.json()
        assert actual["model_version"] == expected.model_version
        assert actual["status"] == expected.status == "ok"
        assert (
            actual["class_id"]
            == actual["care_card_id"]
            == expected.class_id
            == "bacterial_spot"
        )
        assert [row["class_id"] for row in actual["top_k"]] == [
            row.class_id for row in expected.top_k
        ]
        assert [row["score"] for row in actual["top_k"]] == pytest.approx(
            [row.score for row in expected.top_k], abs=1e-6
        )
        assert "logits" not in actual
        care = client.get("/api/v1/care/bacterial_spot", headers=auth)
        assert care.status_code == 200 and care.json()["class_id"] == "bacterial_spot"


class BlockingPredictor(FakePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.concurrent = 0
        self.max_concurrent = 0
        self.image: Image.Image | None = None

    def predict(self, image: Image.Image) -> Prediction:
        self.image = image
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.set()
        try:
            if not self.release.wait(10):
                raise TimeoutError("test barrier timed out")
            return super().predict(image)
        finally:
            self.concurrent -= 1


@pytest.mark.parametrize("cancel", [False, True])
def test_busy_and_cancelled_client_keep_one_slot(
    care_catalog: CareCatalog, auth: dict[str, str], cancel: bool
) -> None:
    predictor = BlockingPredictor()
    app = app_for(predictor, care_catalog)
    files = {"image": ("leaf.png", encoded(), "image/png")}

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8765",
            headers=auth,
        ) as client:
            first = asyncio.create_task(client.post("/api/v1/predict", files=files))
            try:
                started = await asyncio.to_thread(predictor.started.wait, 3)
                assert started, "inference never reached the barrier"
                if cancel:
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                read_body = False

                async def body() -> bytes:
                    nonlocal read_body
                    read_body = True
                    return b"bad body must not be parsed while busy"

                async def stream() -> AsyncIterator[bytes]:
                    yield await body()

                second = await client.post("/api/v1/predict", content=stream())
                assert second.status_code == 429, second.text
                assert second.json()["error"]["code"] == "busy"
                assert not read_body
                assert predictor.max_concurrent == 1
            finally:
                predictor.release.set()
                if not first.done():
                    await first
            if not cancel:
                assert first.result().status_code == 200
            deadline = asyncio.get_running_loop().time() + 3
            while True:
                final = await client.post("/api/v1/predict", files=files)
                if final.status_code != 429:
                    break
                assert asyncio.get_running_loop().time() < deadline, (
                    "slot was never released"
                )
                await asyncio.sleep(0.01)
            assert final.status_code == 200, final.text
            assert predictor.max_concurrent == 1

    asyncio.run(check())
    assert predictor.image is not None
    with pytest.raises(ValueError, match="closed"):
        predictor.image.getpixel((0, 0))


def test_reordered_model_ids_map_to_same_care(
    care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    predictor = FakePredictor()
    predictor.class_ids = ("early_blight", "healthy")
    predictor.result = Prediction(
        [ScoredClass("healthy", 0.9), ScoredClass("early_blight", 0.1)],
        "ok",
        "healthy",
        predictor.model_version,
    )
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            files={"image": ("leaf.png", encoded(), "image/png")},
            headers=auth,
        )
        card = client.get(
            "/api/v1/care/" + response.json()["care_card_id"], headers=auth
        )
    assert response.status_code == 200
    assert response.json()["class_id"] == response.json()["care_card_id"] == "healthy"
    assert card.json()["name_vi"] == "Lá khỏe theo nhãn dữ liệu"


def test_extra_care_cards_do_not_prevent_readiness(
    care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    predictor = FakePredictor()
    predictor.class_ids = ("healthy",)
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        assert client.get("/api/v1/health", headers=auth).json()["ready"] is True


def test_shutdown_during_upload_never_starts_inference(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    app = app_for(predictor, care_catalog)
    data = multipart(encoded())

    async def check() -> None:
        async def body() -> AsyncIterator[bytes]:
            yield data[:100]
            app.state.runtime.stopping.set()
            yield data[100:]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8765",
            headers=auth,
        ) as client:
            response = await client.post(
                "/api/v1/predict",
                content=body(),
                headers={"Content-Type": "multipart/form-data; boundary=test-boundary"},
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert predictor.calls == 0
        assert not app.state.runtime.slot.locked()
        assert not app.state.runtime.workers

    asyncio.run(check())
