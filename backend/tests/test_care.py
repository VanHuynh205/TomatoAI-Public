"""Catalog and export boundaries use synthetic cards, never real care approval."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.tests.test_contract import care_payload as care_payload  # noqa: PLC0414


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture_bundle(path: Path, cards: list[dict[str, Any]]) -> Path:
    write(
        path,
        {
            "schema_version": 1,
            "knowledge_version": "fixture-care-v1",
            "artifact_kind": "fixture",
            "cards": cards,
        },
    )
    return path


def test_catalog_lookup_coverage_and_readonly(
    tmp_path: Path, care_payload: dict[str, Any]
) -> None:
    from backend.care import CareCatalog

    catalog = CareCatalog.load(fixture_bundle(tmp_path / "care.json", [care_payload]))
    assert catalog.knowledge_version == "fixture-care-v1"
    assert catalog.artifact_kind == "fixture"
    card = catalog.get("healthy")
    assert card is not None and card.class_id == "healthy"
    assert catalog.get("../../secret") is None
    catalog.validate_model_classes(["healthy"])
    with pytest.raises(ValueError, match="missing care"):
        catalog.validate_model_classes(["new_class"])
    with pytest.raises(AttributeError):
        catalog.knowledge_version = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        catalog.artifact_kind = "curated"  # type: ignore[misc]


def test_returned_card_cannot_mutate_catalog(care_catalog: Any) -> None:
    card = care_catalog.get("healthy")
    card.symptoms.clear()
    card.actions[0].source_ids.clear()
    assert care_catalog.get("healthy").symptoms
    assert care_catalog.get("healthy").actions[0].source_ids


@pytest.mark.parametrize("length", [0, 5_000_001])
def test_catalog_rejects_empty_or_oversized_file(tmp_path: Path, length: int) -> None:
    from backend.care import CareCatalog

    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * length)
    with pytest.raises(ValueError, match="size limit"):
        CareCatalog.load(path)


@pytest.mark.parametrize("content", [b"[]", b"null", b"{", b"\xff", b"[" * 2000])
def test_catalog_rejects_invalid_json(tmp_path: Path, content: bytes) -> None:
    from backend.care import CareCatalog

    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        CareCatalog.load(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "schema",
        "kind",
        "scope",
        "rights",
        "source",
        "empty",
        "id",
        "expert",
    ],
)
def test_catalog_rejects_invalid_bundle(
    tmp_path: Path, care_payload: dict[str, Any], mutation: str
) -> None:
    from backend.care import CareCatalog

    value: dict[str, Any] = {
        "schema_version": 1,
        "knowledge_version": "v1",
        "artifact_kind": "fixture",
        "cards": [care_payload],
    }
    if mutation == "duplicate":
        value["cards"].append(copy.deepcopy(care_payload))
    elif mutation == "schema":
        value["schema_version"] = True
    elif mutation == "kind":
        value["artifact_kind"] = "approved"
    elif mutation == "scope":
        care_payload["sources"][0]["geographic_scope"] = "unknown"
    elif mutation == "rights":
        care_payload["sources"][0]["rights_status"] = "pending"
    elif mutation == "source":
        care_payload["actions"][0]["source_ids"] = ["absent"]
    elif mutation == "empty":
        care_payload["symptoms"] = [" "]
    elif mutation == "id":
        care_payload["class_id"] = "../healthy"
    else:
        care_payload["expert_verified"] = True
    path = tmp_path / "bad.json"
    write(path, value)
    with pytest.raises(ValueError):
        CareCatalog.load(path)


@pytest.fixture
def export_case(
    tmp_path: Path, care_payload: dict[str, Any]
) -> tuple[Path, Path, Path]:
    cards = tmp_path / "drafts"
    cards.mkdir()
    write(cards / "healthy.json", care_payload)
    write(cards / "unlisted.json", {"not": "a valid care card"})
    allowlist = tmp_path / "allowlist.json"
    write(
        allowlist,
        {
            "schema_version": 1,
            "knowledge_version": "care-v1",
            "allowed_use": "offline_reference_vn",
            "cards": [
                {
                    "class_id": "healthy",
                    "file": "healthy.json",
                    "sha256": hashlib.sha256(
                        (cards / "healthy.json").read_bytes()
                    ).hexdigest(),
                }
            ],
        },
    )
    return cards, allowlist, tmp_path / "care-v1.json"


def test_export_exact_allowlist_deterministic_atomic(
    export_case: tuple[Path, Path, Path],
) -> None:
    from backend.care import CareCatalog
    from desktop.build_care import export_care

    cards, allowlist, output = export_case
    before = {p: p.read_bytes() for p in cards.iterdir()}
    export_care(cards, allowlist, output)
    first = output.read_bytes()
    export_care(cards, allowlist, output)
    assert output.read_bytes() == first
    value = json.loads(first)
    assert [card["class_id"] for card in value["cards"]] == ["healthy"]
    assert value["artifact_kind"] == "curated"
    assert CareCatalog.load(output).knowledge_version == "care-v1"
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize(
    "mutation", ["hash", "path", "id", "duplicate", "rights", "scope", "use", "schema"]
)
def test_export_fails_without_touching_existing_output(
    export_case: tuple[Path, Path, Path], mutation: str
) -> None:
    from desktop.build_care import export_care

    cards, allowlist, output = export_case
    output.write_text("old output", encoding="utf-8")
    value = json.loads(allowlist.read_text())
    row = value["cards"][0]
    if mutation == "hash":
        row["sha256"] = "f" * 64
    elif mutation == "path":
        row["file"] = "../outside.json"
    elif mutation == "id":
        row["class_id"] = "other"
    elif mutation == "duplicate":
        value["cards"].append(row.copy())
    elif mutation in ("rights", "scope"):
        card = json.loads((cards / "healthy.json").read_text())
        key = "rights_status" if mutation == "rights" else "geographic_scope"
        card["sources"][0][key] = "pending" if mutation == "rights" else "unknown"
        write(cards / "healthy.json", card)
        row["sha256"] = hashlib.sha256(
            (cards / "healthy.json").read_bytes()
        ).hexdigest()
    elif mutation == "use":
        value["allowed_use"] = "public_training"
    else:
        value["schema_version"] = True
    write(allowlist, value)
    with pytest.raises(ValueError):
        export_care(cards, allowlist, output)
    assert output.read_text() == "old output"


def test_export_cli(export_case: tuple[Path, Path, Path]) -> None:
    from desktop.build_care import main

    cards, allowlist, output = export_case
    assert (
        main(
            [
                "--cards",
                str(cards),
                "--allowlist",
                str(allowlist),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["artifact_kind"] == "curated"
