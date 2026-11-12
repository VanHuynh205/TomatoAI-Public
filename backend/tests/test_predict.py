"""Synthetic images exercise the real HTTP/parser/decode boundary."""

import asyncio
import io
import struct
import tempfile
import zlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import ClientDisconnect

from ai.contracts import Prediction, ScoredClass
from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor
from backend.tests.test_security import app_for


def encoded(fmt: str = "PNG", size: tuple[int, int] = (16, 12)) -> bytes:
    with Image.new("RGB", size, (45, 170, 60)) as image, io.BytesIO() as stream:
        image.save(stream, fmt)
        return stream.getvalue()


def multipart(data: bytes, *, mime: str = "image/png", name: str = "image") -> bytes:
    return (
        (
            f'--test-boundary\r\nContent-Disposition: form-data; name="{name}"; filename="leaf.png"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        + data
        + b"\r\n--test-boundary--\r\n"
    )


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_maps_ids_and_versions(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    fmt: str,
    mime: str,
) -> None:
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf", encoded(fmt), mime)},
        )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": "1",
        "status": "ok",
        "class_id": "healthy",
        "top_k": [
            {"class_id": "healthy", "score": 0.9},
            {"class_id": "early_blight", "score": 0.1},
        ],
        "model_version": "fixture-model-v1",
        "knowledge_version": "fixture-care-v1",
        "care_card_id": "healthy",
        "warnings": [],
    }
    assert predictor.calls == 1


def test_uncertain_never_selects_care(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    predictor.result = Prediction(
        [ScoredClass("healthy", 0.49)], "uncertain", None, "fixture-model-v1"
    )
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", encoded(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "uncertain"
    assert response.json()["class_id"] is None
    assert response.json()["care_card_id"] is None


@pytest.mark.parametrize(
    "case,status,code",
    [
        ("fake", 422, "invalid_input"),
        ("empty", 422, "invalid_input"),
        ("gif", 415, "unsupported_image"),
        ("mime", 415, "unsupported_image"),
        ("wrong_mime", 415, "unsupported_image"),
        ("truncated", 422, "invalid_input"),
        ("missing", 422, "invalid_input"),
        ("duplicate", 422, "invalid_input"),
        ("wrong_field", 422, "invalid_input"),
        ("text_field", 422, "invalid_input"),
        ("incomplete", 422, "invalid_input"),
        ("json", 422, "invalid_input"),
    ],
)
def test_rejects_input_before_inference(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    case: str,
    status: int,
    code: str,
) -> None:
    data, mime = encoded(), "image/png"
    if case == "fake":
        data = b"this is not a PNG"
    elif case == "empty":
        data = b""
    elif case == "gif":
        data, mime = encoded("GIF"), "image/gif"
    elif case == "mime":
        mime = "text/plain"
    elif case == "wrong_mime":
        mime = "image/jpeg"
    elif case == "truncated":
        data, mime = encoded("JPEG")[:40], "image/jpeg"
    body = multipart(
        data, mime=mime, name="other" if case == "wrong_field" else "image"
    )
    if case == "duplicate":
        body = body.removesuffix(b"--test-boundary--\r\n") + multipart(data)
    elif case == "missing":
        body = b"--test-boundary--\r\n"
    elif case == "text_field":
        body = body.replace(b'; filename="leaf.png"', b"")
    elif case == "incomplete":
        body = body.removesuffix(b"\r\n--test-boundary--\r\n")
    elif case == "json":
        body = b'{"image":"x"}'
    headers = {
        **auth,
        "Content-Type": "application/json"
        if case == "json"
        else "multipart/form-data; boundary=test-boundary",
    }
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post("/api/v1/predict", headers=headers, content=body)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert predictor.calls == 0


@pytest.mark.parametrize("length,status", [(10_000_000, 200), (10_000_001, 413)])
def test_exact_file_byte_limit(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    length: int,
    status: int,
) -> None:
    data = encoded().ljust(length, b"\0")
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", data, "image/png")},
        )
    assert response.status_code == status, response.text
    assert predictor.calls == (1 if status == 200 else 0)
    if status == 413:
        assert response.json()["error"]["code"] == "payload_too_large"


@pytest.mark.parametrize("size,status", [((5000, 4000), 200), ((20000001, 1), 413)])
def test_exact_pixel_limit(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    size: tuple[int, int],
    status: int,
) -> None:
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", encoded(size=size), "image/png")},
        )
    assert response.status_code == status, response.text
    assert predictor.calls == (1 if status == 200 else 0)
    if status == 413:
        assert response.json()["error"]["code"] == "image_too_large"


@pytest.mark.parametrize("side", [10_000, 20_000])
def test_pillow_bomb_headers_are_rejected_before_inference(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    side: int,
) -> None:
    data = bytearray(encoded())
    # Change only IHDR dimensions and its CRC; never allocate the declared pixels.
    data[16:24] = struct.pack(">II", side, side)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", bytes(data), "image/png")},
        )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["code"] == "image_too_large"
    assert predictor.calls == 0


def test_invalid_png_crc_is_rejected_before_inference(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    data = bytearray(encoded())
    chunk = data.index(b"IDAT")
    length = struct.unpack(">I", data[chunk - 4 : chunk])[0]
    data[chunk + 4 + length] ^= 1
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", bytes(data), "image/png")},
        )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_input"
    assert predictor.calls == 0


@pytest.mark.parametrize("extra,status", [(0, 200), (1, 413)])
@pytest.mark.parametrize("content_length", [None, "1", "99999999"])
def test_stream_limit_ignores_content_length(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    extra: int,
    status: int,
    content_length: str | None,
) -> None:
    body = multipart(encoded().ljust(10_000_000, b"\0"))
    # A legal MIME epilogue still counts toward the whole request byte limit.
    body = body.ljust(10_065_536 + extra, b"\r")
    headers = {**auth, "Content-Type": "multipart/form-data; boundary=test-boundary"}
    if content_length is not None:
        headers["Content-Length"] = content_length
    app = app_for(predictor, care_catalog)

    async def check() -> httpx.Response:
        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(body), 65536):
                yield body[start : start + 65536]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            return await client.post(
                "/api/v1/predict", headers=headers, content=chunks()
            )

    response = asyncio.run(check())
    assert response.status_code == status, response.text
    assert predictor.calls == (1 if status == 200 else 0)


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "duplicate",
        "version",
        "nan",
        "inf",
        "negative",
        "over_one",
        "empty",
        "uncertain_id",
        "not_selected",
    ],
)
def test_invalid_model_output_degrades_health(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    invalid: str,
) -> None:
    result = predictor.result
    if invalid == "unknown":
        result.top_k[1].class_id = "not_in_model"
    elif invalid == "duplicate":
        result.top_k[1].class_id = "healthy"
    elif invalid == "version":
        result.model_version = "wrong-version"
    elif invalid in ("nan", "inf", "negative", "over_one"):
        result.top_k[0].score = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -0.1,
            "over_one": 1.1,
        }[invalid]
    elif invalid == "empty":
        result.top_k = []
    elif invalid == "uncertain_id":
        result.status = "uncertain"
    else:
        result.class_id = "not_in_model"
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", encoded(), "image/png")},
        )
        health = client.get("/api/v1/health", headers=auth)
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "model_unavailable"
    assert health.status_code == 503 and health.json()["ready"] is False


def test_model_failure_is_sanitized(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image: Image.Image) -> Prediction:
        raise RuntimeError("C:/private/model-path secret-token")

    monkeypatch.setattr(predictor, "predict", fail)
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            headers=auth,
            files={"image": ("leaf.png", encoded(), "image/png")},
        )
        health = client.get("/api/v1/health", headers=auth)
    assert response.status_code == health.status_code == 503
    assert "private" not in response.text + health.text
    assert "secret-token" not in response.text + health.text


def test_multipart_header_limit_is_preserved(
    predictor: FakePredictor, care_catalog: CareCatalog, auth: dict[str, str]
) -> None:
    payload = multipart(encoded()).replace(
        b"Content-Type:", b"X-Too-Long: " + b"a" * 65536 + b"\r\nContent-Type:", 1
    )
    with TestClient(
        app_for(predictor, care_catalog), base_url="http://127.0.0.1:8765"
    ) as client:
        response = client.post(
            "/api/v1/predict",
            content=payload,
            headers={
                **auth,
                "Content-Type": "multipart/form-data; boundary=test-boundary",
            },
        )
    assert response.status_code == 422
    assert predictor.calls == 0


def test_decode_applies_exif_and_detaches_rgb() -> None:
    from backend.images import decode_image

    with Image.new("RGB", (7, 5), (45, 170, 60)) as original, io.BytesIO() as stream:
        exif = original.getexif()
        exif[274] = 6
        original.save(stream, "JPEG", exif=exif)
        payload = stream.getvalue()
    with decode_image(payload) as decoded:
        assert decoded.size == (5, 7)
        assert decoded.mode == "RGB"
        assert decoded.getexif().get(274) is None


def test_schema_documents_predict_binary_and_errors() -> None:
    from backend.app import build_openapi

    schema = build_openapi()
    operation = schema["paths"]["/api/v1/predict"]["post"]
    body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    if "$ref" in body:
        body = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[1]]
    assert body["required"] == ["image"]
    assert body["properties"]["image"] == {"type": "string", "format": "binary"}
    assert {"200", "401", "403", "413", "415", "422", "429", "500", "503"} <= operation[
        "responses"
    ].keys()


@pytest.mark.parametrize(
    "case",
    ["success", "decode_error", "incomplete", "disconnect", "cancel", "oversize"],
)
def test_upload_files_close_on_all_exit_paths(
    predictor: FakePredictor,
    care_catalog: CareCatalog,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    import starlette.formparsers

    opened: list[Any] = []
    original = tempfile.SpooledTemporaryFile

    def tracked(*args: Any, **kwargs: Any) -> Any:
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked)
    data = b"fake" if case == "decode_error" else encoded().ljust(1_100_000, b"\0")
    payload = multipart(data)
    if case == "incomplete":
        payload = payload.removesuffix(b"\r\n--test-boundary--\r\n")
    app = app_for(predictor, care_catalog)
    headers = {**auth, "Content-Type": "multipart/form-data; boundary=test-boundary"}

    async def check() -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield payload[:500]
            if case == "disconnect":
                raise ClientDisconnect()
            if case == "cancel":
                raise asyncio.CancelledError()
            if case == "oversize":
                yield b"x" * 10_065_537
            else:
                yield payload[500:]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            if case in ("disconnect", "cancel"):
                with pytest.raises(
                    ClientDisconnect if case == "disconnect" else asyncio.CancelledError
                ):
                    await client.post(
                        "/api/v1/predict", headers=headers, content=chunks()
                    )
            else:
                response = await client.post(
                    "/api/v1/predict", headers=headers, content=chunks()
                )
                assert (
                    response.status_code
                    == {
                        "success": 200,
                        "decode_error": 422,
                        "incomplete": 422,
                        "oversize": 413,
                    }[case]
                ), response.text
            assert opened and all(file.closed for file in opened)
            assert all(not file._rolled for file in opened)
            # A rejected/cancelled parse must release the slot for a valid request.
            response = await client.post(
                "/api/v1/predict",
                headers=auth,
                files={"image": ("leaf.png", encoded(), "image/png")},
            )
            assert response.status_code == 200, response.text

    asyncio.run(check())
