"""Bounded multipart uploads and verified, detached RGB images."""

import io
import warnings
from contextlib import closing

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from backend.security import MAX_MULTIPART_BYTES, ApiError

MAX_IMAGE_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 20_000_000
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png"}


class _ImageMultipart(MultiPartParser):
    # The bounded stream keeps uploads in RAM, including on cancellation.
    spool_max_size = MAX_MULTIPART_BYTES + 1
    complete = False

    def on_end(self) -> None:
        self.complete = True


async def read_upload(request: Request) -> tuple[bytes, str]:
    content_types = request.headers.getlist("content-type")
    if (
        len(content_types) != 1
        or content_types[0].split(";", 1)[0].strip().lower() != "multipart/form-data"
    ):
        raise ApiError(
            422, "invalid_input", "Cần gửi một ảnh bằng multipart/form-data."
        )
    parser = _ImageMultipart(
        request.headers, request.stream(), max_files=1, max_fields=0
    )
    try:
        form = await parser.parse()
        items = form.multi_items()
        if (
            not parser.complete
            or len(items) != 1
            or items[0][0] != "image"
            or not isinstance(items[0][1], UploadFile)
        ):
            raise ApiError(422, "invalid_input", "Cần đúng một tệp trong trường image.")
        upload = items[0][1]
        mime = upload.content_type
        if mime is None or mime not in MIME_BY_FORMAT.values():
            raise ApiError(415, "unsupported_image", "Chỉ hỗ trợ ảnh JPEG hoặc PNG.")
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ApiError(413, "payload_too_large", "Ảnh vượt giới hạn 10 MB.")
        return data, mime
    except MultiPartException as error:
        raise ApiError(
            422, "invalid_input", "Dữ liệu multipart không hợp lệ."
        ) from error
    finally:
        # Starlette tracks unfinished parts here; FormData omits them on truncated EOF.
        for file in parser._files_to_close_on_error:
            file.close()


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, "payload_too_large", "Ảnh vượt giới hạn 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with closing(Image.open(io.BytesIO(data))) as source:
                image_format = source.format
                if image_format not in MIME_BY_FORMAT:
                    raise ApiError(
                        415, "unsupported_image", "Chỉ hỗ trợ ảnh JPEG hoặc PNG."
                    )
                if getattr(source, "is_animated", False):
                    raise ApiError(
                        415,
                        "unsupported_image",
                        "Vui lòng chọn ảnh JPEG hoặc PNG tĩnh.",
                    )
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ApiError(
                        413, "image_too_large", "Ảnh vượt giới hạn 20 triệu điểm ảnh."
                    )
                source.verify()
            with closing(Image.open(io.BytesIO(data))) as source:
                source.load()
                with closing(ImageOps.exif_transpose(source)) as oriented:
                    result = oriented.convert("RGB")
                    result.format = image_format
                    return result
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ApiError(
            413, "image_too_large", "Ảnh vượt giới hạn 20 triệu điểm ảnh."
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ApiError(
            422, "invalid_input", "Tệp ảnh rỗng, hỏng hoặc không đọc được."
        ) from error
