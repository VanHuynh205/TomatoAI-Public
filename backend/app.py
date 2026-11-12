"""API composition with injected runtime objects and model-free OpenAPI tooling."""

import asyncio
import threading
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from PIL import Image
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from ai.contracts import Prediction
from backend.care import CareCatalog
from backend.images import MIME_BY_FORMAT, decode_image, read_upload
from backend.schemas import (
    CareResponse,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
    ScoredClassSchema,
)
from backend.security import ApiError, SessionBoundary, validate_boundary


class PredictorContract(Protocol):
    @property
    def class_ids(self) -> Sequence[str]: ...

    @property
    def model_version(self) -> str: ...

    def predict(self, image: Image.Image) -> Prediction: ...


@dataclass
class Runtime:
    predictor: PredictorContract
    catalog: CareCatalog
    ready: bool = True
    reason: str | None = None
    slot: threading.Lock = field(default_factory=threading.Lock)
    workers: set[asyncio.Task[PredictResponse]] = field(default_factory=set)
    stopping: threading.Event = field(default_factory=threading.Event)

    def predict(self, data: bytes, mime: str) -> PredictResponse:
        try:
            with closing(decode_image(data)) as image:
                if MIME_BY_FORMAT.get(image.format or "") != mime:
                    raise ApiError(
                        415, "unsupported_image", "Loại tệp không khớp nội dung ảnh."
                    )
                try:
                    prediction = self.predictor.predict(image)
                    if (
                        prediction.model_version != self.predictor.model_version
                        or not prediction.top_k
                        or any(
                            item.class_id not in self.predictor.class_ids
                            for item in prediction.top_k
                        )
                    ):
                        raise ValueError("invalid model output")
                    return PredictResponse(
                        status=prediction.status,
                        class_id=prediction.class_id,
                        top_k=[
                            ScoredClassSchema(class_id=item.class_id, score=item.score)
                            for item in prediction.top_k
                        ],
                        model_version=prediction.model_version,
                        knowledge_version=self.catalog.knowledge_version,
                        care_card_id=prediction.class_id
                        if prediction.status == "ok"
                        else None,
                        warnings=[],
                    )
                except Exception as error:
                    self.ready = False
                    self.reason = (
                        "Mô hình không sẵn sàng. Vui lòng khởi động lại ứng dụng."
                    )
                    raise ApiError(503, "model_unavailable", self.reason) from error
        finally:
            self.slot.release()

    def worker_done(self, task: asyncio.Task[PredictResponse]) -> None:
        self.workers.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve errors even when the requesting client has left.


router = APIRouter(
    prefix="/api/v1",
    responses={
        status: {"model": ErrorResponse} for status in (401, 403, 404, 422, 500)
    },
)


@router.get(
    "/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
)
async def health(request: Request) -> JSONResponse:
    runtime = cast(Runtime, request.app.state.runtime)
    ready = runtime.ready and not runtime.stopping.is_set()
    body = HealthResponse(
        ready=ready,
        model_version=runtime.predictor.model_version,
        knowledge_version=runtime.catalog.knowledge_version,
        reason="Ứng dụng đang đóng." if runtime.stopping.is_set() else runtime.reason,
    )
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)


@router.get("/care/{class_id}", response_model=CareResponse)
async def care(class_id: str, request: Request) -> CareResponse:
    runtime = cast(Runtime, request.app.state.runtime)
    card = runtime.catalog.get(class_id)
    if card is None:
        raise ApiError(404, "care_not_found", "Không có thẻ chăm sóc cho nhãn này.")
    return CareResponse(
        **card.model_dump(), knowledge_version=runtime.catalog.knowledge_version
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={status: {"model": ErrorResponse} for status in (413, 415, 429, 503)},
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["image"],
                        "properties": {"image": {"type": "string", "format": "binary"}},
                        "additionalProperties": False,
                    }
                }
            },
        }
    },
)
async def predict(request: Request) -> PredictResponse:
    runtime = cast(Runtime, request.app.state.runtime)
    if not runtime.ready or runtime.stopping.is_set():
        raise ApiError(
            503, "model_unavailable", runtime.reason or "Mô hình không sẵn sàng."
        )
    if not runtime.slot.acquire(blocking=False):
        raise ApiError(
            429, "busy", "Ứng dụng đang nhận diện một ảnh. Vui lòng thử lại sau."
        )
    handed_off = False
    try:
        data, mime = await read_upload(request)
        if runtime.stopping.is_set():
            raise ApiError(503, "model_unavailable", "Ứng dụng đang đóng.")
        task = asyncio.create_task(asyncio.to_thread(runtime.predict, data, mime))
        runtime.workers.add(task)
        task.add_done_callback(runtime.worker_done)
        handed_off = True
        # Cancellation stops waiting, never the worker that owns the slot and image.
        return await asyncio.shield(task)
    finally:
        if not handed_off:
            runtime.slot.release()


def _base_app() -> FastAPI:
    app = FastAPI(
        title="Tomato Desktop API",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return error.response()

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return ApiError(
            422, "invalid_input", "Dữ liệu yêu cầu không hợp lệ."
        ).response()

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return ApiError(
            error.status_code, "invalid_input", "Yêu cầu không hợp lệ."
        ).response()

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        return ApiError(500, "internal_error", "Không thể xử lý yêu cầu.").response()

    return app


def create_app(
    *,
    predictor: PredictorContract,
    care_catalog: CareCatalog,
    session_token: str,
    allowed_origin: str,
) -> FastAPI:
    validate_boundary(session_token, allowed_origin)
    care_catalog.validate_model_classes(predictor.class_ids)
    app = _base_app()
    app.state.runtime = Runtime(predictor, care_catalog)
    app.add_middleware(
        SessionBoundary, session_token=session_token, allowed_origin=allowed_origin
    )
    return app


def build_openapi() -> dict[str, Any]:
    return _base_app().openapi()
