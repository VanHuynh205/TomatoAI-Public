"""Opt-in D4 smoke: trained CPU predictor versus the real HTTP composition."""

import argparse
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from ai.cnn_gru_inference import CNN_GRU_MANIFEST, CnnGruPredictor
from ai.contracts import Prediction
from ai.dinov2_inference import DINO_V2_MANIFEST, DinoV2Predictor
from ai.inference import Predictor
from backend.app import PredictorContract
from backend.care import CareCatalog
from backend.schemas import CareResponse, HealthResponse, PredictResponse
from desktop.server import DEFAULT_MODEL_BUNDLE
from desktop.tests.browser_server import ROOT

# Locked before real execution; float32 CPU kernels may use different thread counts.
ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5


def assert_parity(
    expected: Prediction, payload: dict[str, Any], knowledge: str
) -> None:
    actual = PredictResponse.model_validate(payload)
    assert actual.status == expected.status
    assert actual.class_id == expected.class_id
    assert actual.model_version == expected.model_version
    assert actual.knowledge_version == knowledge
    assert actual.care_card_id == (
        expected.class_id if expected.status == "ok" else None
    )
    assert len(actual.top_k) == len(expected.top_k)
    for observed, direct in zip(actual.top_k, expected.top_k, strict=True):
        assert observed.class_id == direct.class_id
        assert math.isclose(
            observed.score, direct.score, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE
        )


def run(report_path: Path, port: int, bundle: Path | None = None) -> None:
    model_dir = (
        (bundle if bundle.is_absolute() else ROOT / bundle)
        if bundle is not None
        else ROOT / DEFAULT_MODEL_BUNDLE
    )
    care_path = ROOT / "infra/knowledge/care-v1.json"
    catalog = CareCatalog.load(care_path)
    if catalog.artifact_kind != "curated":
        raise ValueError("Real smoke requires curated care; pending drafts cannot run.")
    images = [
        ROOT / "desktop/tests/fixtures" / name
        for name in ("healthy.png", "healthy.jpg", "uncertain.png")
    ]
    is_dinov2 = (model_dir / DINO_V2_MANIFEST).is_file()
    is_cnn_gru = (model_dir / CNN_GRU_MANIFEST).is_file()
    model_inputs = (
        sorted(p for p in model_dir.iterdir() if p.is_file())
        if (is_dinov2 or is_cnn_gru)
        else [model_dir / "metadata.json", model_dir / "model.pth"]
    )
    inputs = [*model_inputs, care_path, *images]

    def _label(path: Path) -> str:
        return (
            path.relative_to(ROOT).as_posix()
            if path.is_relative_to(ROOT)
            else str(path)
        )

    before = {
        _label(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    predictor: PredictorContract = (
        DinoV2Predictor(model_dir)
        if is_dinov2
        else CnnGruPredictor(model_dir)
        if is_cnn_gru
        else Predictor(model_dir, device="cpu")
    )
    catalog.validate_model_classes(predictor.class_ids)
    token = secrets.token_urlsafe(32)
    origin = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "TOMATO_E2E_SESSION_TOKEN": token,
        "TOMATO_E2E_MODEL_DIR": str(model_dir),
        "TOMATO_E2E_CARE_PATH": str(care_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with report_path.with_suffix(".server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "desktop.tests.browser_server",
                "--port",
                str(port),
                "--mode",
                "real",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=origin,
                timeout=30,
                trust_env=False,
                follow_redirects=False,
                headers={"X-Session-Token": token, "Origin": origin},
            ) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Real harness exited before readiness; inspect server log."
                        )
                    try:
                        response = client.get("/api/v1/health", timeout=1)
                    except httpx.TransportError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Real harness readiness exceeded 30 seconds."
                            ) from None
                        time.sleep(0.05)
                        continue
                    response.raise_for_status()
                    health = HealthResponse.model_validate(response.json())
                    assert health.ready and health.reason is None
                    assert health.model_version == predictor.model_version
                    assert health.knowledge_version == catalog.knowledge_version
                    break
                for path in images:
                    with (
                        Image.open(path) as original,
                        closing(original.convert("RGB")) as rgb,
                    ):
                        expected = predictor.predict(rgb)
                    mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
                    response = client.post(
                        "/api/v1/predict",
                        files={"image": (path.name, path.read_bytes(), mime)},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    assert_parity(expected, payload, catalog.knowledge_version)
                    if payload["care_card_id"] is not None:
                        card_response = client.get(
                            "/api/v1/care/" + payload["care_card_id"]
                        )
                        card_response.raise_for_status()
                        card = catalog.get(payload["care_card_id"])
                        assert card is not None
                        assert card_response.json() == CareResponse(
                            **card.model_dump(),
                            knowledge_version=catalog.knowledge_version,
                        ).model_dump(mode="json")
                    rows.append(
                        {
                            "image": path.relative_to(ROOT).as_posix(),
                            "direct": asdict(expected),
                            "api": payload,
                        }
                    )
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    after = {
        _label(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    assert before == after, "Smoke inputs changed during execution"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "mode": "real",
                "model_version": predictor.model_version,
                "knowledge_version": catalog.knowledge_version,
                "device": "cpu",
                "absolute_tolerance": ABS_TOLERANCE,
                "relative_tolerance": REL_TOLERANCE,
                "input_sha256": before,
                "inputs_unchanged": True,
                "cases": rows,
                "scope": "Synthetic fixture parity only; no accuracy, TEST or native offline claim.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bundle", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    run(args.report, args.port, args.bundle)
    print("Real CPU/API parity passed; report:", args.report)


if __name__ == "__main__":
    main()
