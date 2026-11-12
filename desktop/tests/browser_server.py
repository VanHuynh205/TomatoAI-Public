"""Browser-only fixture/real harness. Never imported by the production launcher."""

import argparse
import os
import socket
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI
from PIL import Image

from ai.artifacts import load_bundle
from ai.contracts import Prediction, ScoredClass
from ai.dinov2_inference import DINO_V2_MANIFEST
from backend.app import create_app
from backend.care import CareCatalog
from backend.security import validate_boundary
from desktop.server import DEFAULT_MODEL_BUNDLE, add_frontend, create_desktop_app

ROOT = Path(__file__).resolve().parents[2]


class FixturePredictor:
    class_ids = ("healthy", "fixture_other")
    model_version = "fixture-model-v1"

    def predict(self, image: Image.Image) -> Prediction:
        assert image.mode == "RGB"
        _, green, blue = cast(tuple[int, int, int], image.getpixel((0, 0)))
        uncertain = blue > green
        return Prediction(
            top_k=[ScoredClass("healthy", 0.8), ScoredClass("fixture_other", 0.2)],
            status="uncertain" if uncertain else "ok",
            class_id=None if uncertain else "healthy",
            model_version=self.model_version,
        )


def create_browser_app(mode: str, token: str, origin: str) -> FastAPI:
    validate_boundary(token, origin)
    if mode == "fixture":
        app = create_app(
            predictor=FixturePredictor(),
            care_catalog=CareCatalog.load(ROOT / "desktop/tests/fixtures/care.json"),
            session_token=token,
            allowed_origin=origin,
        )
        add_frontend(app, ROOT / "frontend/dist")
        return app
    if mode not in ("real", "fixture-bundle", "dinov3"):
        raise ValueError("unknown browser harness mode")
    model_dir = (
        Path(os.getenv("TOMATO_E2E_MODEL_DIR") or ROOT / DEFAULT_MODEL_BUNDLE)
        if mode == "real"
        else Path(os.environ["TOMATO_E2E_MODEL_DIR"])
    )
    if mode == "dinov3":
        if not (model_dir / "leaf-classifier.json").is_file():
            raise ValueError("dinov3 mode requires an explicit DINOv3 leaf export")
    elif mode == "fixture-bundle" or not (model_dir / DINO_V2_MANIFEST).is_file():
        bundle = load_bundle(model_dir, require_trained=mode == "real")
        if mode == "fixture-bundle" and bundle.metadata["artifact_kind"] != "fixture":
            raise ValueError("fixture-bundle mode requires fixture weights")
    app = create_desktop_app(
        ROOT / "frontend/dist",
        model_dir,
        Path(os.environ["TOMATO_E2E_CARE_PATH"]),
        token,
        origin,
    )
    if app.state.runtime.catalog.artifact_kind != "curated":
        raise ValueError("real smoke requires the curated care bundle")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("fixture", "real", "fixture-bundle", "dinov3"), required=True
    )
    args = parser.parse_args()
    token = os.environ["TOMATO_E2E_SESSION_TOKEN"]
    origin = f"http://127.0.0.1:{args.port}"
    validate_boundary(token, origin)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", args.port))
        sock.setblocking(False)
        app = create_browser_app(args.mode, token, origin)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                workers=1,
                reload=False,
                access_log=False,
                log_level="error",
                proxy_headers=False,
                server_header=False,
                ws="none",
                timeout_graceful_shutdown=10,
            )
        )
        try:
            server.run(sockets=[sock])
        finally:
            app.state.runtime.stopping.set()


if __name__ == "__main__":
    main()
