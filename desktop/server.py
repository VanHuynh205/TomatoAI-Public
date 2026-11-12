"""Compose one CPU predictor, the existing API and local frontend assets."""

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from ai.dinov2_inference import DINO_V2_MANIFEST, DinoV2Predictor
from ai.inference import Predictor
from backend.app import PredictorContract, create_app
from backend.care import CareCatalog
from backend.security import ApiError

DEFAULT_MODEL_BUNDLE = "ai/artifacts/dinov2-v1-280"


class _FrontendFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        # Static files must never turn an unknown API route into frontend content.
        if path.replace("\\", "/").split("/", 1)[0] == "api":
            return ApiError(404, "invalid_input", "Yêu cầu không hợp lệ.").response()
        return await super().get_response(path, scope)


def create_desktop_app(
    asset_dir: Path,
    model_dir: Path,
    care_path: Path,
    session_token: str,
    origin: str,
) -> FastAPI:
    assets = _asset_directory(asset_dir)
    predictor: PredictorContract
    if (model_dir / DINO_V2_MANIFEST).is_file():
        predictor = DinoV2Predictor(model_dir)
    elif (model_dir / "leaf-classifier.json").is_file():
        from ai.dinov3_inference import DinoV3Predictor

        predictor = DinoV3Predictor(model_dir)
    else:
        predictor = Predictor(model_dir, device="cpu")
    catalog = CareCatalog.load(care_path)
    app = create_app(
        predictor=predictor,
        care_catalog=catalog,
        session_token=session_token,
        allowed_origin=origin,
    )
    add_frontend(app, assets)
    return app


def _asset_directory(asset_dir: Path) -> Path:
    assets = asset_dir.resolve(strict=True)
    index = assets / "index.html"
    if not index.is_file() or not index.resolve().is_relative_to(assets):
        raise ValueError("missing or invalid frontend index.html")
    return assets


def add_frontend(app: FastAPI, asset_dir: Path) -> None:
    """Share the same static boundary with the isolated browser test harness."""
    assets = _asset_directory(asset_dir)
    app.mount("/", _FrontendFiles(directory=assets, html=True))
