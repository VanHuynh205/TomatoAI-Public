"""Validated v1 PIL transforms, explicit geometry and caller-owned randomness."""

import math
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms as T  # type: ignore[import-untyped]

from ai.data import MAX_PIXELS

_FIELDS = {
    "schema_version",
    "color_mode",
    "geometry",
    "resize_size",
    "crop_size",
    "interpolation",
    "antialias",
    "mean",
    "std",
    "horizontal_flip_p",
    "rotation_degrees",
    "random_crop_scale",
    "random_crop_ratio",
    "fill",
}


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _sequence(value: Any, size: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == size
        and all(_number(item) for item in value)
    )


def _validate(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict) or set(spec) != _FIELDS:
        raise ValueError("transform spec has missing or unknown fields")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if spec["color_mode"] != "RGB" or spec["geometry"] not in (
        "fit_pad",
        "center_crop",
    ):
        raise ValueError("invalid color_mode or geometry")
    if (
        type(spec["resize_size"]) is not int
        or spec["resize_size"] < 224
        or type(spec["crop_size"]) is not int
        or spec["crop_size"] != 224
        or spec["interpolation"] not in ("bicubic", "bilinear")
        or spec["antialias"] is not True
    ):
        raise ValueError("invalid resize/crop/interpolation/antialias")
    if spec["resize_size"] ** 2 > MAX_PIXELS:
        raise ValueError("pixel_limit: resize_size exceeds budget")
    for field in ("mean", "std"):
        if not _sequence(spec[field], 3) or (field == "std" and min(spec[field]) <= 0):
            raise ValueError(f"invalid {field}")
    mean = torch.tensor(spec["mean"], dtype=torch.float32)
    std = torch.tensor(spec["std"], dtype=torch.float32)
    bounds = (torch.tensor([[0.0], [1.0]], dtype=torch.float32) - mean) / std
    if not (
        torch.isfinite(mean).all()
        and torch.isfinite(std).all()
        and (std > 0).all()
        and torch.isfinite(bounds).all()
    ):
        raise ValueError("normalization must stay finite in float32")
    if (
        not _number(spec["horizontal_flip_p"])
        or not 0 <= spec["horizontal_flip_p"] <= 1
    ):
        raise ValueError("invalid horizontal_flip_p")
    if (
        not _number(spec["rotation_degrees"])
        or not 0 <= spec["rotation_degrees"] <= 180
    ):
        raise ValueError("invalid rotation_degrees")
    for field in ("random_crop_scale", "random_crop_ratio"):
        value = spec[field]
        if (
            not _sequence(value, 2)
            or not 0 < value[0] <= value[1]
            or (field == "random_crop_scale" and value[1] > 1)
            or (field == "random_crop_ratio" and tuple(value) != (1, 1))
            or not (torch.tensor(value, dtype=torch.float32) > 0).all()
        ):
            raise ValueError(f"invalid {field}")
    fill = spec["fill"]
    if not _sequence(fill, 3) or any(
        type(v) is not int or not 0 <= v <= 255 for v in fill
    ):
        raise ValueError("invalid fill")
    if spec["geometry"] == "fit_pad" and (
        spec["resize_size"] != spec["crop_size"]
        or spec["rotation_degrees"] != 0
        or tuple(spec["random_crop_scale"]) != (1, 1)
    ):
        raise ValueError("fit_pad requires equal sizes and no rotation/crop")


class _Transform:
    def __init__(self, spec: dict[str, Any], training: bool) -> None:
        self.fit_pad = spec["geometry"] == "fit_pad"
        self.resize_size = None if self.fit_pad or training else spec["resize_size"]
        self.size = spec["crop_size"]
        self.fill = tuple(spec["fill"])
        self.resample = (
            Image.Resampling.BICUBIC
            if spec["interpolation"] == "bicubic"
            else Image.Resampling.BILINEAR
        )
        interpolation = (
            T.InterpolationMode.BICUBIC
            if spec["interpolation"] == "bicubic"
            else T.InterpolationMode.BILINEAR
        )
        steps = []
        if not self.fit_pad:
            if training:
                steps.append(
                    T.RandomResizedCrop(
                        self.size,
                        scale=tuple(spec["random_crop_scale"]),
                        ratio=tuple(spec["random_crop_ratio"]),
                        interpolation=interpolation,
                        antialias=True,
                    )
                )
            else:
                steps.extend(
                    [
                        T.Resize(
                            spec["resize_size"],
                            interpolation=interpolation,
                            antialias=True,
                        ),
                        T.CenterCrop(self.size),
                    ]
                )
        if training:
            steps.append(T.RandomHorizontalFlip(spec["horizontal_flip_p"]))
            if spec["rotation_degrees"]:
                steps.append(
                    T.RandomRotation(
                        spec["rotation_degrees"],
                        interpolation=interpolation,
                        fill=self.fill,
                    )
                )
        steps.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(tuple(spec["mean"]), tuple(spec["std"])),
            ]
        )
        self.pipeline = T.Compose(steps)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("pixel_limit: input exceeds budget")
        if self.resize_size is not None:
            short, long = sorted(image.size)
            if self.resize_size * (self.resize_size * long // short) > MAX_PIXELS:
                raise ValueError("pixel_limit: resized image exceeds budget")
        with ImageOps.exif_transpose(image) as oriented, oriented.convert("RGB") as rgb:
            if self.fit_pad:
                long = max(rgb.size)
                with (
                    rgb.resize(
                        (
                            max(1, round(rgb.width / long * self.size)),
                            max(1, round(rgb.height / long * self.size)),
                        ),
                        resample=self.resample,
                    ) as contained,
                    Image.new("RGB", (self.size, self.size), self.fill) as padded,
                ):
                    padded.paste(
                        contained,
                        (
                            (self.size - contained.width) // 2,
                            (self.size - contained.height) // 2,
                        ),
                    )
                    result: torch.Tensor = self.pipeline(padded)
            else:
                result = self.pipeline(rgb)
        return result.cpu()


def build_transform(
    spec: dict[str, Any], training: bool = False
) -> Callable[[Image.Image], torch.Tensor]:
    _validate(spec)
    return _Transform(spec, training)
