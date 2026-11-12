"""The loopback API still requires an exact host, origin and session token."""

import ipaddress
from hmac import compare_digest
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.schemas import ErrorCode, ErrorDetail, ErrorResponse

MAX_MULTIPART_BYTES = 10_065_536


class ApiError(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def response(self) -> JSONResponse:
        body = ErrorResponse(error=ErrorDetail(code=self.code, message=self.message))
        return JSONResponse(body.model_dump(), status_code=self.status)


def validate_boundary(session_token: str, allowed_origin: str) -> str:
    if not session_token or any(not 33 <= ord(char) <= 126 for char in session_token):
        raise ValueError("session token must contain visible ASCII characters")
    parsed = urlsplit(allowed_origin)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            "origin must be an HTTP loopback address with a port"
        ) from error
    if (
        not loopback
        or parsed.scheme != "http"
        or not port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or allowed_origin != f"http://{parsed.netloc}"
        or any(char.isspace() for char in allowed_origin)
    ):
        raise ValueError("origin must be an HTTP loopback address with a port")
    return parsed.netloc


class SessionBoundary:
    def __init__(
        self, app: ASGIApp, *, session_token: str, allowed_origin: str
    ) -> None:
        self.app = app
        self.host = validate_boundary(session_token, allowed_origin)
        self.token = session_token.encode("ascii")
        self.origin = allowed_origin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not (
            scope["path"] == "/api" or scope["path"].startswith("/api/")
        ):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        error: ApiError | None = None
        if headers.getlist("host") != [self.host]:
            error = ApiError(403, "forbidden_host", "Máy chủ yêu cầu không hợp lệ.")
        else:
            tokens = headers.getlist("x-session-token")
            if len(tokens) != 1 or not compare_digest(
                tokens[0].encode("latin-1"), self.token
            ):
                error = ApiError(401, "unauthorized", "Phiên làm việc không hợp lệ.")
            else:
                origins = headers.getlist("origin")
                if (origins and origins != [self.origin]) or (
                    scope["method"] == "POST" and not origins
                ):
                    error = ApiError(
                        403, "forbidden_origin", "Nguồn yêu cầu không hợp lệ."
                    )
        if error is not None:
            await error.response()(scope, receive, send)
            return
        received = 0

        async def bounded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_MULTIPART_BYTES:
                    raise ApiError(
                        413, "payload_too_large", "Yêu cầu vượt giới hạn dung lượng."
                    )
            return message

        await self.app(scope, bounded_receive, send)
