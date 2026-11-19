"""M2 grouping and sealing contracts, using synthetic metadata and evidence only."""

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai.data import CLASSES, MANIFEST_COLUMNS
from ai.split import assign_splits, prepare_dataset, seal_dataset, verify_seal
from tests.ai.conftest import AuditCase


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_prepare_inherits_locked_membership_and_seals_it(
    dataset: tuple[Path, dict[str, Any]], grouped_rows: list[dict[str, Any]]
) -> None:
    root, review = dataset
    rotate = {"train": "test", "test": "val", "val": "train"}
    membership = {
        row["sample_id"]: rotate[row["split"]] for row in assign_splits(grouped_rows)
    }
    prepare_dataset(root, review, inherited_splits=membership)
    verify_seal(root)
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["sample_id"]: row["split"] for row in rows} == membership
    inherited = root / "inherited-splits.json"
    payload = json.loads(inherited.read_text())
    payload["assignments"][rows[0]["sample_id"]] = "val"
    inherited.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal_hash"):
        verify_seal(root)


@pytest.mark.parametrize("fault", ["missing", "extra", "unknown", "cross_group"])
def test_prepare_rejects_invalid_inherited_membership(
    dataset: tuple[Path, dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    fault: str,
) -> None:
    root, review = dataset
    assigned = assign_splits(grouped_rows)
    membership = {row["sample_id"]: row["split"] for row in assigned}
    if fault == "missing":
        membership.pop(assigned[0]["sample_id"])
    elif fault == "extra":
        membership["unknown-sample"] = "train"
    elif fault == "unknown":
        membership[assigned[0]["sample_id"]] = "development"
    else:
        membership["copy-a"] = "test" if membership["copy-b"] != "test" else "val"
    with pytest.raises(ValueError, match="inherited"):
        prepare_dataset(root, review, inherited_splits=membership)
    assert not (root / "seal.json").exists()


@pytest.fixture
def grouped_rows() -> list[dict[str, Any]]:
    rows = []
    for class_id, label, _ in CLASSES[:2]:
        for number in range(20):
            sample = f"{class_id}:{number:02}"
            rows.append(
                dict(
                    zip(
                        MANIFEST_COLUMNS,
                        [
                            "1",
                            sample,
                            "synthetic",
                            "v1",
                            f"{sample}.png",
                            digest(sample),
                            digest("pixel" + sample),
                            label,
                            class_id,
                            "group:" + sample,
                            "[]",
                            "",
                            "",
                            "",
                        ],
                        strict=True,
                    )
                )
            )
    # A byte edge followed by a pixel edge and a separate group edge.
    rows.extend(
        [
            {
                **rows[0],
                "sample_id": "copy-a",
                "group_id": "copy-group",
                "pixel_sha256": digest("copy-pixel"),
            },
            {
                **rows[1],
                "sample_id": "copy-b",
                "group_id": "another-group",
                "pixel_sha256": digest("copy-pixel"),
            },
            {**rows[2], "sample_id": "copy-c", "group_id": rows[0]["group_id"]},
        ]
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(
    tmp_path: Path, grouped_rows: list[dict[str, Any]], audit_case: AuditCase
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "dataset"
    root.mkdir()
    write_csv(root / "manifest.csv", grouped_rows, MANIFEST_COLUMNS)
    write_csv(
        root / "exclusions.csv",
        [
            {
                "sample_id": "old-excluded",
                "source_path": "old.png",
                "label_original": "original",
                "reason": "status_needs_review",
            }
        ],
        ("sample_id", "source_path", "label_original", "reason"),
    )
    (root / "label_map.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classes": [
                    {
                        "index": i,
                        "class_id": c,
                        "label_original": label,
                        "name_vi": name,
                    }
                    for i, (c, label, name) in enumerate(CLASSES[:2])
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                **audit_case.provenance,
                "source_dataset": "synthetic",
                "source_version": "v1",
                "audit_stage": "M1-only",
                "split_sealed": False,
                "source_root": "../source",
                "ready_for_training": False,
            }
        ),
        encoding="utf-8",
    )
    evidence = root / "review-sheets" / "synthetic.png"
    evidence.parent.mkdir()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(evidence)
    ids = [row["sample_id"] for row in grouped_rows]
    review = {
        "schema_version": 1,
        "reviewer": "pytest synthetic screening",
        "method": "synthetic fixture",
        "checked_at": "2026-11-02",
        "expert_verified": False,
        "input_manifest_sha256": hashlib.sha256(
            (root / "manifest.csv").read_bytes()
        ).hexdigest(),
        "sample_ids": ids,
        "quarantine_sample_ids": [],
        "notes": "Synthetic only",
        "limitations": "No real visual assessment",
        "contact_sheets": [
            {
                "file": "review-sheets/synthetic.png",
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "sample_ids": ids,
            }
        ],
    }
    return root, review


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_grouped_split_is_disjoint_and_repeatable(
    grouped_rows: list[dict[str, Any]], seed: int
) -> None:
    original = copy.deepcopy(grouped_rows)
    out = assign_splits(grouped_rows, seed)
    assert out == assign_splits(grouped_rows[::-1], seed)
    assert grouped_rows == original
    assert len(out) == len(grouped_rows)
    assert len({r["sample_id"] for r in out}) == len(out)
    for key in ("group_id", "source_sha256", "pixel_sha256"):
        for value in {r[key] for r in out}:
            assert len({r["split"] for r in out if r[key] == value}) == 1
    assert {r["split"] for r in out} == {"train", "val", "test"}
    bridged = [
        r
        for r in out
        if r["sample_id"]
        in {
            "bacterial_spot:00",
            "bacterial_spot:01",
            "bacterial_spot:02",
            "copy-a",
            "copy-b",
            "copy-c",
        }
    ]
    assert len({r["component_id"] for r in bridged}) == 1
    for class_id in {r["class_id"] for r in out}:
        components = {
            r["component_id"]: r["split"] for r in out if r["class_id"] == class_id
        }
        assert (
            abs(list(components.values()).count("train") / len(components) - 0.7) < 0.1
        )


def test_conflicting_duplicate_labels_quarantined(
    grouped_rows: list[dict[str, Any]],
) -> None:
    grouped_rows[-1]["class_id"] = "early_blight"
    out = assign_splits(grouped_rows)
    quarantined = [r for r in out if r["exclusion_reason"]]
    assert len(quarantined) == 6
    assert {r["exclusion_reason"] for r in quarantined} == {"conflicting_label"}
    assert {r["split"] for r in quarantined} == {""}
    assert (
        next(r for r in out if r["sample_id"] == "copy-c")["class_id"] == "early_blight"
    )


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("sample_id", "", "sample_id"),
        ("class_id", " ", "class_id"),
        ("group_id", "", "group_id"),
        ("source_sha256", "x", "source_sha256"),
        ("pixel_sha256", None, "pixel_sha256"),
        ("schema_version", True, "schema_version"),
    ],
)
def test_bad_assignment_input(
    grouped_rows: list[dict[str, Any]], key: str, value: Any, error: str
) -> None:
    grouped_rows[0][key] = value
    with pytest.raises(ValueError, match=error):
        assign_splits(grouped_rows)


def test_duplicate_id_and_seed_rejected(grouped_rows: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="seed"):
        assign_splits(grouped_rows, True)
    with pytest.raises(ValueError, match="sample_id"):
        assign_splits(grouped_rows + [grouped_rows[0]])


def test_keys_are_separate_namespaces(grouped_rows: list[dict[str, Any]]) -> None:
    rows = grouped_rows[:3]
    rows[0]["group_id"] = rows[1]["source_sha256"]
    assert len({r["component_id"] for r in assign_splits(rows)}) == 3


def test_prepare_seal_and_preserved_metadata(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    source_hash = review["input_manifest_sha256"]
    label_map = (root / "label_map.json").read_bytes()
    prepare_dataset(root, review, 42)
    seal = verify_seal(root)
    assert seal["technical_gate"] == "pass"
    assert seal["expert_verified"] is False
    assert seal["reviewer"] == review["reviewer"]
    assert (root / "label_map.json").read_bytes() == label_map
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    assert split["source_manifest_sha256"] == source_hash
    assert len(split["assignments"]) == 43
    assert len(split["warnings"]) == 2
    assert split["counts"]["early_blight"]["train"] == {"images": 14, "components": 14}
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == MANIFEST_COLUMNS
        assert {r["split"] for r in reader} == {"train", "val", "test"}
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["audit_stage"] == "M2-sealed"
    assert provenance["ready_for_training"] is False
    assert provenance["source_root"] == "../source"
    assert provenance["license"]["identifier"] == "LicenseRef-Synthetic-Fixture"
    assert "review-sheets/synthetic.png" in seal["hashes"]
    with pytest.raises(ValueError, match="already.*seal"):
        prepare_dataset(root, review)


def test_review_quarantine_whole_component(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    review["quarantine_sample_ids"] = ["copy-a"]
    prepare_dataset(root, review)
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    assert len(split["quarantine"]) == 1
    assert len(split["quarantine"][0]["sample_ids"]) == 6
    assert split["quarantine"][0]["reason"] == "review_unresolved"
    with (root / "exclusions.csv").open(encoding="utf-8") as stream:
        exclusions = list(csv.DictReader(stream))
    assert len(exclusions) == 7
    assert exclusions[0]["sample_id"] == "old-excluded"
    assert len({r["sample_id"] for r in exclusions}) == 7
    verify_seal(root)


@pytest.mark.parametrize("groups", [0, 1, 2])
def test_insufficient_groups_blocks_seal(
    dataset: tuple[Path, dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    groups: int,
) -> None:
    root, review = dataset
    rows = [r for r in grouped_rows if r["class_id"] == "early_blight"][:groups]
    write_csv(root / "manifest.csv", rows, MANIFEST_COLUMNS)
    review["input_manifest_sha256"] = hashlib.sha256(
        (root / "manifest.csv").read_bytes()
    ).hexdigest()
    review["sample_ids"] = [r["sample_id"] for r in rows]
    review["contact_sheets"][0]["sample_ids"] = review["sample_ids"]
    before = (root / "manifest.csv").read_bytes()
    with pytest.raises(ValueError, match="missing_class_support"):
        prepare_dataset(root, review)
    assert (root / "manifest.csv").read_bytes() == before
    assert not (root / "seal.json").exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "manifest.csv",
        "split.json",
        "label_map.json",
        "provenance.json",
        "exclusions.csv",
        "review.json",
        "review-sheets/synthetic.png",
    ],
)
def test_changed_manifest_invalidates_seal(
    dataset: tuple[Path, dict[str, Any]], artifact: str
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    with (root / artifact).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="seal_hash"):
        verify_seal(root)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("reviewer", ""),
        ("method", " "),
        ("checked_at", "2026-02-30"),
        ("checked_at", "20261102"),
        ("expert_verified", True),
        ("expert_verified", 0),
        ("input_manifest_sha256", "0" * 64),
        ("sample_ids", ["extra-observation"]),
        ("quarantine_sample_ids", ["old-excluded"]),
        ("contact_sheets", []),
    ],
)
def test_invalid_review_is_atomic(
    dataset: tuple[Path, dict[str, Any]], key: str, value: Any
) -> None:
    root, review = dataset
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    review[key] = value
    with pytest.raises(ValueError, match="review"):
        prepare_dataset(root, review)
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()} == before


@pytest.mark.parametrize(
    "change", ["hash", "missing", "path", "traversal", "ids", "duplicate", "coverage"]
)
def test_invalid_evidence(dataset: tuple[Path, dict[str, Any]], change: str) -> None:
    root, review = dataset
    sheet = review["contact_sheets"][0]
    if change == "hash":
        sheet["sha256"] = "0" * 64
    elif change == "missing":
        sheet["file"] = "review-sheets/missing.png"
    elif change == "path":
        sheet["file"] = str(root / "review-sheets/synthetic.png")
    elif change == "traversal":
        sheet["file"] = "review-sheets/../review-sheets/synthetic.png"
    elif change == "ids":
        sheet["sample_ids"] = ["unseen"]
    elif change == "duplicate":
        review["sample_ids"].append(review["sample_ids"][0])
    else:
        review["sample_ids"] = review["sample_ids"][:1]
        sheet["sample_ids"] = review["sample_ids"]
    with pytest.raises(ValueError, match="review"):
        prepare_dataset(root, review)
    assert not (root / "seal.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_class",
        "mapping",
        "csv_header",
        "csv_duplicate",
        "csv_shape",
        "csv_path",
    ],
)
def test_invalid_dataset(
    dataset: tuple[Path, dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    problem: str,
) -> None:
    root, review = dataset
    if problem in {"missing_class", "mapping"}:
        path = root / "label_map.json"
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if problem == "missing_class":
            mapping["classes"].pop()
        else:
            mapping["classes"][0]["label_original"] = CLASSES[1][1]
        path.write_text(json.dumps(mapping), encoding="utf-8")
    else:
        if problem == "csv_header":
            (root / "manifest.csv").write_text(
                "sample_id,sample_id\nx,x\n", encoding="utf-8"
            )
        elif problem == "csv_shape":
            with (root / "manifest.csv").open("a", encoding="utf-8") as stream:
                stream.write("x,y\n")
        else:
            if problem == "csv_duplicate":
                grouped_rows.append(grouped_rows[0])
            else:
                grouped_rows[0]["derived_path"] = "../outside.png"
            write_csv(root / "manifest.csv", grouped_rows, MANIFEST_COLUMNS)
        review["input_manifest_sha256"] = hashlib.sha256(
            (root / "manifest.csv").read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        prepare_dataset(root, review)
    assert not (root / "seal.json").exists()


def test_unprepared_or_broken_seal_rejected(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    with pytest.raises(ValueError, match="prepare"):
        seal_dataset(root, review)
    with pytest.raises(ValueError):
        verify_seal(root)
    (root / "seal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already.*seal"):
        prepare_dataset(root, review)


def test_cli_success_and_rejects_overwrite(
    dataset: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    root, review = dataset
    review_path = tmp_path / "review-input.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "ai.split",
        "--dataset",
        str(root),
        "--seed",
        "42",
        "--review",
        str(review_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert verify_seal(root)["schema_version"] == 1
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1
    assert "already" in again.stderr


def test_cli_bad_review_preserves_input(
    dataset: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    root, review = dataset
    review["expert_verified"] = True
    path = tmp_path / "bad-review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    before = (root / "manifest.csv").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.split",
            "--dataset",
            str(root),
            "--review",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "review" in result.stderr
    assert (root / "manifest.csv").read_bytes() == before


@pytest.mark.parametrize("groups", [0, 1, 2, 3])
def test_small_assignment_support_without_splitting_groups(
    grouped_rows: list[dict[str, Any]], groups: int
) -> None:
    out = assign_splits(grouped_rows[20 : 20 + groups])
    assert len(out) == groups
    assert {r["split"] for r in out} == (
        {"train", "val", "test"} if groups == 3 else {"train"} if groups else set()
    )


def test_seed_changes_mapping_and_phash_never_unions(
    grouped_rows: list[dict[str, Any]],
) -> None:
    for row in grouped_rows:
        row["phash"] = "identical-candidate-only"
    first = assign_splits(grouped_rows, 1)
    second = assign_splits(grouped_rows, 2)
    assert {r["sample_id"]: r["split"] for r in first} != {
        r["sample_id"]: r["split"] for r in second
    }
    assert len({r["component_id"] for r in first}) == 38


@pytest.mark.parametrize(
    "problem", ["schema", "path", "removed_hash", "gate", "identity"]
)
def test_invalid_seal_schema_paths_gate_and_identity(
    dataset: tuple[Path, dict[str, Any]], problem: str
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    path = root / "seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    if problem == "schema":
        seal["schema_version"] = True
    elif problem == "path":
        seal["hashes"]["../outside.png"] = "0" * 64
    elif problem == "removed_hash":
        del seal["hashes"]["review-sheets/synthetic.png"]
    elif problem == "gate":
        seal["technical_gate"] = "fail"
    else:
        seal["reviewer"] = "somebody else"
    path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_seal(root)


def test_missing_prepared_seed_cannot_seal(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    (root / "seal.json").unlink()
    path = root / "split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    del split["seed"]
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        seal_dataset(root, review)
    assert not (root / "seal.json").exists()


def test_reviewed_label_map_order_cannot_be_reindexed(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    path = root / "label_map.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["classes"].reverse()
    for index, item in enumerate(mapping["classes"]):
        item["index"] = index
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="label_map"):
        prepare_dataset(root, review)


def test_previously_excluded_sample_cannot_return(
    dataset: tuple[Path, dict[str, Any]], grouped_rows: list[dict[str, Any]]
) -> None:
    root, review = dataset
    grouped_rows[0]["sample_id"] = "old-excluded"
    write_csv(root / "manifest.csv", grouped_rows, MANIFEST_COLUMNS)
    review["input_manifest_sha256"] = hashlib.sha256(
        (root / "manifest.csv").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="revived"):
        prepare_dataset(root, review)


def test_recomputed_hashes_do_not_hide_split_leakage(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    path = root / "split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    split["assignments"][0]["split"] = (
        "val" if split["assignments"][0]["split"] != "val" else "test"
    )
    path.write_text(json.dumps(split), encoding="utf-8")
    seal_path = root / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["hashes"]["split.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="assignments"):
        verify_seal(root)


def test_invalid_assignment_schema_is_value_error(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    path = root / "split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    split["assignments"][0]["class_id"] = []
    path.write_text(json.dumps(split), encoding="utf-8")
    seal_path = root / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["hashes"]["split.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="class_id"):
        verify_seal(root)


def test_review_preserves_list_notes_and_limitations(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    review["notes"] = ["Synthetic screening note"]
    review["limitations"] = ["No expert assessment"]
    prepare_dataset(root, review)
    saved = json.loads((root / "review.json").read_text(encoding="utf-8"))
    assert saved["notes"] == ["Synthetic screening note"]
    assert saved["limitations"] == ["No expert assessment"]
    verify_seal(root)


@pytest.mark.parametrize("stage", ["prepare", "verify"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("license", None, "license"),
        ("license", {"identifier": "synthetic", "status": "pending"}, "license"),
        ("use_scope", [], "provenance"),
        ("input_manifest_sha256", "not-a-hash", "provenance"),
        ("source_dataset", "undeclared-source", "provenance"),
        ("source_version", "undeclared-version", "provenance"),
        ("source_root", "../missing-source", "source_root"),
        ("source_root", "C:/external-source", "source_root"),
    ],
)
def test_invalid_m1_provenance_cannot_gain_a_technical_seal(
    dataset: tuple[Path, dict[str, Any]],
    stage: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    root, review = dataset
    if stage == "verify":
        prepare_dataset(root, review)
    path = root / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance[field] = value
    path.write_text(json.dumps(provenance), encoding="utf-8")
    before = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
    if stage == "verify":
        seal_path = root / "seal.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["hashes"]["provenance.json"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            verify_seal(root)
    else:
        with pytest.raises(ValueError, match=message):
            prepare_dataset(root, review)
        assert {p: p.read_bytes() for p in root.iterdir() if p.is_file()} == before
        assert not (root / "seal.json").exists()


def test_changed_license_evidence_invalidates_seal(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    root, review = dataset
    prepare_dataset(root, review)
    (root.parent / "source" / "permission.txt").write_text("changed fixture terms")
    with pytest.raises(ValueError, match="license"):
        verify_seal(root)
