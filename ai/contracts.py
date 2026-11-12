"""Shared prediction values; safe to import without a tensor runtime."""

from dataclasses import dataclass
from typing import Literal

CLASS_ID_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


@dataclass
class ScoredClass:
    class_id: str
    score: float


@dataclass
class Prediction:
    top_k: list[ScoredClass]
    status: Literal["ok", "uncertain"]
    class_id: str | None
    model_version: str
