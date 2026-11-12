"""Synthetic contract fixtures only; no horticultural advice or approved sources."""

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from backend.schemas import (
    CareCard,
    CareResponse,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
    ScoredClassSchema,
)


@pytest.fixture
def ok_payload() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "ok",
        "class_id": "healthy",
        "top_k": [
            {"class_id": "healthy", "score": 0.8},
            {"class_id": "fixture_other", "score": 0.2},
        ],
        "model_version": "fixture-model-v1",
        "knowledge_version": "fixture-knowledge-v1",
        "care_card_id": "healthy",
        "warnings": [],
    }


@pytest.fixture
def uncertain_payload(ok_payload: dict[str, Any]) -> dict[str, Any]:
    return {**ok_payload, "status": "uncertain", "class_id": None, "care_card_id": None}


@pytest.fixture
def care_payload() -> dict[str, Any]:
    return {
        "class_id": "healthy",
        "name_vi": "Thẻ fixture, không phải hướng dẫn chăm sóc",
        "name_en": "Synthetic fixture card",
        "category": "healthy",
        "symptoms": ["Fixture symptom text"],
        "confusable_conditions": ["Fixture comparison text"],
        "observe_next": ["Fixture observation text"],
        "actions": [
            {
                "text": "Fixture action, not advice",
                "source_ids": ["fixture_source"],
                "applicability": "Synthetic fixture only",
            }
        ],
        "limits": ["Synthetic fixture; no clinical or agronomic validity"],
        "sources": [
            {
                "source_id": "fixture_source",
                "url": "https://example.com/fixture",
                "locator": "Fixture section 1",
                "geographic_scope": "Synthetic fixture only",
                "rights_status": "permitted",
                "rights_basis": "Synthetic test data authored for this test",
                "rights_evidence": "Fixture statement only; not a real source approval",
                "checked_at": "2026-11-02",
            }
        ],
        "reviewer": "fixture-author",
        "checked_at": "2026-11-02",
        "review_status": "source_checked",
        "reviewer_role": "project_author",
        "expert_verified": False,
    }


def test_ok_prediction_keeps_selected_class_and_scores(
    ok_payload: dict[str, Any],
) -> None:
    result = PredictResponse.model_validate(ok_payload)
    assert result.class_id == "healthy"
    assert result.care_card_id == "healthy"
    assert [item.score for item in result.top_k] == [0.8, 0.2]


def test_uncertain_prediction_keeps_both_ids_null(
    uncertain_payload: dict[str, Any],
) -> None:
    result = PredictResponse.model_validate(uncertain_payload)
    assert result.model_dump()["class_id"] is None
    assert result.model_dump()["care_card_id"] is None


@pytest.mark.parametrize("field", ["class_id", "care_card_id"])
def test_uncertain_rejects_any_selected_id(
    uncertain_payload: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValidationError):
        PredictResponse.model_validate({**uncertain_payload, field: "healthy"})


@pytest.mark.parametrize(
    "updates",
    [
        {"class_id": None},
        {"class_id": "not_in_top_k", "care_card_id": "not_in_top_k"},
        {"care_card_id": None},
        {"care_card_id": "fixture_other"},
        {"top_k": []},
    ],
)
def test_ok_rejects_inconsistent_selection(
    ok_payload: dict[str, Any], updates: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        PredictResponse.model_validate({**ok_payload, **updates})


def test_duplicate_top_k_ids_are_rejected(ok_payload: dict[str, Any]) -> None:
    ok_payload["top_k"][1]["class_id"] = "healthy"
    with pytest.raises(ValidationError):
        PredictResponse.model_validate(ok_payload)


@pytest.mark.parametrize("score", [0, 1, 0.5])
def test_score_inclusive_boundaries_are_kept(score: float) -> None:
    assert ScoredClassSchema(class_id="healthy", score=score).score == score


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), -float("inf"), -0.01, 1.01, True, False, "0.5"],
)
def test_score_rejects_invalid_numbers_and_coercion(score: Any) -> None:
    with pytest.raises(ValidationError):
        ScoredClassSchema(class_id="healthy", score=score)


def test_tied_scores_are_valid_and_preserved(ok_payload: dict[str, Any]) -> None:
    ok_payload["top_k"][0]["score"] = 0.5
    ok_payload["top_k"][1]["score"] = 0.5
    assert [
        item.score for item in PredictResponse.model_validate(ok_payload).top_k
    ] == [0.5, 0.5]


@pytest.mark.parametrize(
    "class_id",
    ["", "Healthy", "1healthy", "healthy-id", "healthy/../x", "a" * 65, "healthy\n", 1],
)
def test_class_id_rejects_invalid_format(class_id: Any) -> None:
    with pytest.raises(ValidationError):
        ScoredClassSchema(class_id=class_id, score=0.5)


@pytest.mark.parametrize("class_id", ["a", "a" * 64, "healthy_2"])
def test_class_id_accepts_format_boundaries(class_id: str) -> None:
    assert ScoredClassSchema(class_id=class_id, score=1).class_id == class_id


@pytest.mark.parametrize("schema_version", [1, "2", None])
@pytest.mark.parametrize("kind", ["predict", "health", "error", "care"])
def test_http_schema_version_is_string_one(
    schema_version: Any,
    kind: str,
    ok_payload: dict[str, Any],
    care_payload: dict[str, Any],
) -> None:
    payloads: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {
        "predict": (PredictResponse, ok_payload),
        "health": (
            HealthResponse,
            {
                "ready": True,
                "model_version": "fixture-model-v1",
                "knowledge_version": "fixture-knowledge-v1",
                "reason": None,
            },
        ),
        "error": (
            ErrorResponse,
            {"error": {"code": "invalid_input", "message": "Fixture error"}},
        ),
        "care": (
            CareResponse,
            {**care_payload, "knowledge_version": "fixture-knowledge-v1"},
        ),
    }
    model, payload = payloads[kind]
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "schema_version": schema_version})


@pytest.mark.parametrize("ready,reason", [(True, None), (False, "Fixture unavailable")])
def test_health_serializes_readiness(ready: bool, reason: str | None) -> None:
    result = HealthResponse(
        ready=ready,
        model_version="fixture-model-v1",
        knowledge_version="fixture-knowledge-v1",
        reason=reason,
    )
    assert result.model_dump() == {
        "schema_version": "1",
        "ready": ready,
        "model_version": "fixture-model-v1",
        "knowledge_version": "fixture-knowledge-v1",
        "reason": reason,
    }


@pytest.mark.parametrize(
    "code",
    [
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
    ],
)
def test_error_envelope_preserves_contract_codes(code: str) -> None:
    result = ErrorResponse.model_validate(
        {"error": {"code": code, "message": "Fixture error"}}
    )
    assert result.model_dump() == {
        "schema_version": "1",
        "error": {"code": code, "message": "Fixture error"},
    }


def test_unknown_error_code_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(
            {"error": {"code": "unknown", "message": "Fixture error"}}
        )


def test_care_response_is_flat_and_json_dates_are_iso(
    care_payload: dict[str, Any],
) -> None:
    result = CareResponse.model_validate(
        {**care_payload, "knowledge_version": "fixture-knowledge-v1"}
    )
    body = result.model_dump(mode="json")
    assert body["class_id"] == "healthy"
    assert body["schema_version"] == "1"
    assert body["knowledge_version"] == "fixture-knowledge-v1"
    assert body["checked_at"] == "2026-11-02"
    assert body["sources"][0]["checked_at"] == "2026-11-02"
    assert body["expert_verified"] is False
    assert CareResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "field",
    [
        "class_id",
        "name_vi",
        "name_en",
        "category",
        "symptoms",
        "confusable_conditions",
        "observe_next",
        "actions",
        "limits",
        "sources",
        "reviewer",
        "checked_at",
        "review_status",
        "reviewer_role",
        "expert_verified",
    ],
)
def test_care_required_fields_cannot_be_omitted(
    care_payload: dict[str, Any], field: str
) -> None:
    del care_payload[field]
    with pytest.raises(ValidationError):
        CareCard.model_validate(care_payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("name_vi",), "  "),
        (("class_id",), "Bad-ID"),
        (("category",), "unknown"),
        (("symptoms",), [" "]),
        (("limits",), []),
        (("sources",), []),
        (("reviewer",), ""),
        (("review_status",), "draft"),
        (("reviewer_role",), "expert"),
        (("expert_verified",), True),
        (("expert_verified",), 0),
        (("checked_at",), "20261102"),
        (("checked_at",), "2026-02-30"),
        (("checked_at",), "2026-11-02T00:00:00Z"),
        (("checked_at",), 0),
        (("sources", 0, "url"), "file:///private/source"),
        (("sources", 0, "url"), "javascript:alert(1)"),
        (("sources", 0, "source_id"), "Bad-ID"),
        (("sources", 0, "locator"), ""),
        (("sources", 0, "geographic_scope"), " Unknown "),
        (("sources", 0, "rights_status"), "pending"),
        (("sources", 0, "rights_basis"), " "),
        (("sources", 0, "rights_evidence"), ""),
        (("sources", 0, "checked_at"), "2026-02-30"),
        (("actions", 0, "text"), ""),
        (("actions", 0, "applicability"), " "),
        (("actions", 0, "source_ids"), []),
        (("actions", 0, "source_ids"), ["missing_source"]),
        (("actions", 0, "source_ids"), ["fixture_source", "fixture_source"]),
    ],
)
def test_care_rejects_incomplete_or_inconsistent_evidence(
    care_payload: dict[str, Any], path: tuple[str | int, ...], value: Any
) -> None:
    target = care_payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError):
        CareCard.model_validate(care_payload)


def test_duplicate_source_ids_are_rejected(care_payload: dict[str, Any]) -> None:
    care_payload["sources"].append(deepcopy(care_payload["sources"][0]))
    with pytest.raises(ValidationError):
        CareCard.model_validate(care_payload)


def test_care_allows_omitting_ungrounded_actions(care_payload: dict[str, Any]) -> None:
    care_payload["actions"] = []
    assert CareCard.model_validate(care_payload).actions == []
