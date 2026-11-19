"""Check smoke assertions with explicit expected results, without claiming model proof."""

import copy
from typing import Any

import pytest

from ai.contracts import Prediction, ScoredClass


@pytest.fixture
def parity_case() -> tuple[Prediction, dict[str, Any]]:
    prediction = Prediction(
        [ScoredClass("healthy", 0.75), ScoredClass("early_blight", 0.25)],
        "ok",
        "healthy",
        "test-model",
    )
    response = {
        "schema_version": "1",
        "status": "ok",
        "class_id": "healthy",
        "top_k": [
            {"class_id": "healthy", "score": 0.75},
            {"class_id": "early_blight", "score": 0.25},
        ],
        "model_version": "test-model",
        "knowledge_version": "test-care",
        "care_card_id": "healthy",
        "warnings": [],
    }
    return prediction, response


def test_parity_accepts_exact_prediction(
    parity_case: tuple[Prediction, dict[str, Any]],
) -> None:
    from desktop.tests.real_smoke import assert_parity

    prediction, response = parity_case
    assert_parity(prediction, response, "test-care")
    uncertain = Prediction(
        prediction.top_k, "uncertain", None, prediction.model_version
    )
    response.update(status="uncertain", class_id=None, care_card_id=None)
    assert_parity(uncertain, response, "test-care")


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "status",
        "class",
        "model",
        "knowledge",
        "care",
        "order",
        "score",
        "length",
    ],
)
def test_parity_rejects_drift(
    parity_case: tuple[Prediction, dict[str, Any]], field: str
) -> None:
    from desktop.tests.real_smoke import assert_parity

    prediction, original = parity_case
    response = copy.deepcopy(original)
    if field == "schema":
        response["schema_version"] = "2"
    elif field == "status":
        response.update(status="uncertain", class_id=None, care_card_id=None)
    elif field == "class":
        response.update(class_id="early_blight", care_card_id="early_blight")
    elif field == "model":
        response["model_version"] = "other-model"
    elif field == "knowledge":
        response["knowledge_version"] = "other-care"
    elif field == "care":
        response["care_card_id"] = None
    elif field == "order":
        response["top_k"].reverse()
    elif field == "score":
        response["top_k"][0]["score"] = 0.751
    else:
        response["top_k"].pop()
    with pytest.raises((AssertionError, ValueError)):
        assert_parity(prediction, response, "test-care")
