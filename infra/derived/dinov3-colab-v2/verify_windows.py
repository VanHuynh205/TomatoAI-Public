"""Verify the frozen Colab v2 export through the Windows desktop API."""

import csv
import hashlib
import json
import math
import platform
import secrets
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V1 = HERE.with_name("dinov3-colab-v1")
MODEL_VERSION = "dinov3-vits16-field-224-v2"
CARE_VERSION = "care-v1"
NON_LEAF = "not_tomato_leaf"
CONFIDENCE_TOLERANCE = 0.01
REPEATS = 12


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_fixture(predictions, previous, class_id, status, output_name):
    with predictions.open(encoding="utf-8", newline="") as stream:
        candidates = [
            row
            for row in csv.DictReader(stream)
            if row["class_id"] == row["prediction"] == class_id
            and row["status"] == status
            and (status != "ok" or row["domain"] == "field")
            and 0 <= float(row["confidence"]) <= 1
        ]
    assert candidates, f"No qualifying {class_id} fixture in {predictions.name}"
    reused = next(
        (
            row
            for row in candidates
            if (row["path"], row["sha256"]) == (previous["path"], previous["sha256"])
        ),
        None,
    )
    row = reused or max(candidates, key=lambda item: float(item["confidence"]))
    return {
        **row,
        "benchmark": previous["benchmark"],
        "file": f"smoke/{output_name}",
        "selection": "reused_v1" if reused else "v2_integration_fixture",
    }


def verify_prediction(payload, fixture, tomato_classes):
    expected = fixture["prediction"] if fixture["status"] == "ok" else None
    assert payload["schema_version"] == "1", payload
    assert payload["model_version"] == MODEL_VERSION, payload
    assert payload["knowledge_version"] == CARE_VERSION, payload
    assert payload["status"] == fixture["status"], payload
    assert payload["class_id"] == payload["care_card_id"] == expected, payload
    top = payload["top_k"]
    ids = {item["class_id"] for item in top}
    scores = [item["score"] for item in top]
    assert len(top) == len(ids) == 3 and ids <= tomato_classes, payload
    assert scores == sorted(scores, reverse=True), payload
    confidence = float(fixture["confidence"])
    if expected is not None:
        assert top[0]["class_id"] == expected, payload
        assert math.isclose(scores[0], confidence, abs_tol=CONFIDENCE_TOLERANCE), (
            payload
        )
        return {"api_top1_matches_colab": True, "api_top1_score": scores[0]}

    # The API omits the non-leaf score. Every omitted tomato score is at most top-3.
    upper = max(0.0, 1.0 - sum(scores))
    lower = max(0.0, upper - (len(tomato_classes) - len(top)) * scores[-1])
    assert lower - CONFIDENCE_TOLERANCE <= confidence <= upper + CONFIDENCE_TOLERANCE, (
        payload
    )
    return {
        "api_non_leaf_score_bounds": [lower, upper],
        "api_top1_matches_colab": True if lower > scores[0] else None,
    }


def main():
    if not __debug__:
        raise RuntimeError("Run with Python assertions enabled")
    assert platform.system() == "Windows", "Run this smoke test on Windows"
    model = HERE / "model"
    frozen_path = HERE / "frozen-model.json"
    frozen = read_json(frozen_path)
    assert frozen["status"] == "frozen_before_holdout_prediction", frozen["status"]
    assert frozen["model_version"] == MODEL_VERSION, frozen["model_version"]
    model_hashes = {path.name: sha256(path) for path in sorted(model.iterdir())}
    assert model_hashes == frozen["sha256"], (
        "Model files differ from the frozen receipt"
    )
    classes = read_json(model / "leaf-classifier.json")["class_ids"]
    assert classes[-1] == NON_LEAF and len(classes) == len(set(classes)), classes
    tomato_classes = set(classes[:-1])
    lock_path = V1 / "holdout-lock.json"
    assert sha256(lock_path) == frozen["holdout_lock_sha256"], "Holdout lock mismatch"

    previous = {row["benchmark"]: row for row in read_json(V1 / "smoke/fixtures.json")}
    external = HERE / "dinov3-v2-external-predictions.csv"
    legacy = HERE / "dinov3-v2-legacy-predictions.csv"
    fixtures = [
        select_fixture(
            external, previous["external"], "bacterial_spot", "ok", "accepted-leaf.jpg"
        ),
        select_fixture(
            legacy, previous["legacy"], NON_LEAF, "uncertain", "rejected-non-leaf.jpg"
        ),
    ]
    holdout = V1 / "dinov3-colab-holdout.zip"
    with ZipFile(holdout) as archive:
        images = [archive.read(f"{row['benchmark']}/{row['path']}") for row in fixtures]
    for row, image in zip(fixtures, images, strict=True):
        assert hashlib.sha256(image).hexdigest() == row["sha256"], row["path"]

    sys.path.insert(0, str(ROOT))
    import torch
    from fastapi.testclient import TestClient

    from backend.schemas import CareResponse
    from desktop.server import create_desktop_app

    assets = ROOT / "frontend/dist"
    care_path = ROOT / "infra/knowledge/care-v1.json"
    origin = "http://127.0.0.1:8765"
    token = secrets.token_urlsafe(32)
    started = perf_counter()
    app = create_desktop_app(assets, model, care_path, token, origin)
    load_seconds = perf_counter() - started
    runtime = app.state.runtime
    assert runtime.catalog.artifact_kind == "curated"
    assert set(runtime.predictor.class_ids) == tomato_classes
    devices = {tensor.device.type for tensor in runtime.predictor._model.parameters()}
    assert devices == {"cpu"}, devices

    cases = []
    headers = {"X-Session-Token": token, "Origin": origin}
    with TestClient(app, base_url=origin, headers=headers) as client:
        health = client.get("/api/v1/health").raise_for_status().json()
        assert health == {
            "schema_version": "1",
            "ready": True,
            "model_version": MODEL_VERSION,
            "knowledge_version": CARE_VERSION,
            "reason": None,
        }, health
        for row, image in zip(fixtures, images, strict=True):
            upload = {"image": (Path(row["file"]).name, image, "image/jpeg")}
            warmup = client.post("/api/v1/predict", files=upload)
            payload = warmup.raise_for_status().json()
            parity = verify_prediction(payload, row, tomato_classes)
            latency = []
            for _ in range(REPEATS):
                started = perf_counter()
                response = client.post("/api/v1/predict", files=upload)
                latency.append((perf_counter() - started) * 1000)
                assert response.raise_for_status().json() == payload, row["path"]

            card = client.get(f"/api/v1/care/{row['prediction']}")
            if payload["care_card_id"] is None:
                assert card.status_code == 404, card.text
                assert card.json()["error"]["code"] == "care_not_found", card.text
            else:
                expected = runtime.catalog.get(row["prediction"])
                assert expected is not None
                assert card.raise_for_status().json() == CareResponse(
                    **expected.model_dump(), knowledge_version=CARE_VERSION
                ).model_dump(mode="json")
            cases.append(
                {
                    "fixture": row,
                    "response": payload,
                    "colab_parity": parity,
                    "care_status_code": card.status_code,
                    "batch_size": 1,
                    "repeats": REPEATS,
                    "latency_ms": latency,
                    "median_ms": median(latency),
                    "p95_nearest_rank_ms": sorted(latency)[
                        math.ceil(REPEATS * 0.95) - 1
                    ],
                }
            )

    smoke = HERE / "smoke"
    smoke.mkdir(exist_ok=True)
    for row, image in zip(fixtures, images, strict=True):
        (HERE / row["file"]).write_bytes(image)
    write_json(smoke / "fixtures.json", fixtures)
    report = smoke / "windows-cpu.json"
    write_json(
        report,
        {
            "status": "passed",
            "scope": "Windows CPU using the production desktop API composition via TestClient",
            "fixture_scope": "Integration cases selected for known outcomes",
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "model_load_seconds": load_seconds,
            "model_version": MODEL_VERSION,
            "selected_epoch": frozen["selected_epoch"],
            "model_sha256": model_hashes["model.safetensors"],
            "model_file_sha256": model_hashes,
            "artifact_sha256": {
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in (
                    frozen_path,
                    external,
                    legacy,
                    lock_path,
                    holdout,
                    care_path,
                    assets / "index.html",
                )
            },
            "confidence_absolute_tolerance": CONFIDENCE_TOLERANCE,
            "health": health,
            "cases": cases,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
