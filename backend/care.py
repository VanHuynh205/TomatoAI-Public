"""Bounded care catalogs; class IDs, not model output indexes, select cards."""

import json
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas import CareCard, NonEmptyText

MAX_CARE_BYTES = 5_000_000


def _local(path: Path) -> Path:
    text = str(path)
    if text.replace("\\", "/").startswith("//") or re.match(
        r"^[A-Za-z][A-Za-z0-9+.-]+:", text
    ):
        raise ValueError("care path must be local")
    absolute = Path(os.path.abspath(path))
    if absolute.resolve() != absolute:
        raise ValueError("care path must not traverse symlinks or junctions")
    return absolute


def _read(path: Path) -> bytes:
    with _local(path).open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_CARE_BYTES:
            raise ValueError("care size limit")
        content = stream.read(MAX_CARE_BYTES + 1)
    if not 0 < len(content) <= MAX_CARE_BYTES:
        raise ValueError("care size limit")
    return content


def _json(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid care JSON") from error
    if not isinstance(value, dict):
        raise ValueError("care JSON must be an object")  # noqa: TRY004
    return value


class _Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(strict=True, ge=1, le=1)
    knowledge_version: NonEmptyText = Field(max_length=64)
    artifact_kind: Literal["curated", "fixture"]
    cards: list[CareCard] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_ids(self) -> "_Bundle":
        ids = [card.class_id for card in self.cards]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate care class_id")
        return self


class CareCatalog:
    def __init__(self, bundle: _Bundle) -> None:
        self._version = bundle.knowledge_version
        self._kind = bundle.artifact_kind
        self._cards = {card.class_id: card for card in bundle.cards}

    @classmethod
    def load(cls, path: Path) -> "CareCatalog":
        return cls(_Bundle.model_validate(_json(_read(path))))

    @property
    def knowledge_version(self) -> str:
        return self._version

    @property
    def artifact_kind(self) -> Literal["curated", "fixture"]:
        return self._kind

    def validate_model_classes(self, class_ids: Sequence[str]) -> None:
        if isinstance(class_ids, str) or not class_ids:
            raise ValueError("model class IDs must be a nonempty sequence")
        if any(label not in self._cards for label in class_ids):
            raise ValueError("missing care for model class")

    def get(self, class_id: str) -> CareCard | None:
        card = self._cards.get(class_id)
        return card.model_copy(deep=True) if card is not None else None
