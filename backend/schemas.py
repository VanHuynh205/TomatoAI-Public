"""Version 1 HTTP contracts, independent of model inference and care storage."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

from ai.contracts import CLASS_ID_PATTERN

ClassId = Annotated[str, StringConstraints(strict=True, pattern=CLASS_ID_PATTERN)]
NonEmptyText = Annotated[
    str, StringConstraints(strict=True, min_length=1, pattern=r"\S")
]


def _iso_date(value: object) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    raise ValueError("checked_at must be an ISO date (YYYY-MM-DD)")


IsoDate = Annotated[date, BeforeValidator(_iso_date)]
ErrorCode = Literal[
    "unauthorized",
    "forbidden_origin",
    "forbidden_host",
    "care_not_found",
    "payload_too_large",
    "image_too_large",
    "unsupported_image",
    "invalid_input",
    "busy",
    "model_unavailable",
    "internal_error",
]


class ScoredClassSchema(BaseModel):
    class_id: ClassId
    score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class HealthResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    ready: bool = Field(strict=True)
    model_version: NonEmptyText
    knowledge_version: NonEmptyText
    reason: NonEmptyText | None


class PredictResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    status: Literal["ok", "uncertain"]
    class_id: ClassId | None
    top_k: list[ScoredClassSchema]
    model_version: NonEmptyText
    knowledge_version: NonEmptyText
    care_card_id: ClassId | None
    warnings: list[NonEmptyText]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        class_ids = [item.class_id for item in self.top_k]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("top_k class IDs must be unique")
        if self.status == "uncertain":
            if self.class_id is not None or self.care_card_id is not None:
                raise ValueError("uncertain predictions must have null selected IDs")
        elif self.class_id not in class_ids or self.care_card_id != self.class_id:
            raise ValueError(
                "ok predictions must select a top_k class and matching card"
            )
        return self


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: NonEmptyText


class ErrorResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    error: ErrorDetail


class CareSource(BaseModel):
    source_id: ClassId
    url: HttpUrl
    locator: NonEmptyText
    geographic_scope: NonEmptyText
    rights_status: Literal["permitted"]
    rights_basis: NonEmptyText
    rights_evidence: NonEmptyText
    checked_at: IsoDate

    @field_validator("geographic_scope")
    @classmethod
    def reject_unknown_scope(cls, value: str) -> str:
        if value.strip().casefold() == "unknown":
            raise ValueError("geographic scope must be known")
        return value


class CareAction(BaseModel):
    text: NonEmptyText
    source_ids: list[ClassId] = Field(min_length=1)
    applicability: NonEmptyText

    @field_validator("source_ids")
    @classmethod
    def unique_sources(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("action source IDs must be unique")
        return value


class CareCard(BaseModel):
    class_id: ClassId
    name_vi: NonEmptyText
    name_en: NonEmptyText
    category: Literal["disease", "pest", "healthy"]
    symptoms: list[NonEmptyText] = Field(min_length=1)
    confusable_conditions: list[NonEmptyText] = Field(min_length=1)
    observe_next: list[NonEmptyText] = Field(min_length=1)
    actions: list[CareAction]
    limits: list[NonEmptyText] = Field(min_length=1)
    sources: list[CareSource] = Field(min_length=1)
    reviewer: NonEmptyText
    checked_at: IsoDate
    review_status: Literal["source_checked"]
    reviewer_role: Literal["project_author"]
    expert_verified: Literal[False]

    @field_validator("expert_verified", mode="before")
    @classmethod
    def no_expert_claim(cls, value: object) -> Literal[False]:
        if value is not False:
            raise ValueError("expert_verified must be false")
        return False

    @model_validator(mode="after")
    def resolve_action_sources(self) -> Self:
        source_ids = {source.source_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("card source IDs must be unique")
        for action in self.actions:
            if not set(action.source_ids).issubset(source_ids):
                raise ValueError("action references an unresolved source")
        return self


class CareResponse(CareCard):
    schema_version: Literal["1"] = "1"
    knowledge_version: NonEmptyText
