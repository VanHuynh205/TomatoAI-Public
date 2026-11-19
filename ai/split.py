"""Deterministic M2 component splits and a local technical seal (no training approval)."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path, PureWindowsPath
from typing import Any

from ai.data import (
    CLASSES,
    MANIFEST_COLUMNS,
    _nonblank,
    _sha256,
    _source_path,
    _validate_provenance,
    _write_json,
)
from ai.write_lock import exclusive_writer

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
GROUP_KEYS = ("group_id", "source_sha256", "pixel_sha256")
ALGORITHM = "union-by-group-byte-pixel; sorted-components; seeded-shuffle-per-class-v1"
INHERITED_ALGORITHM = "union-by-group-byte-pixel; inherited-membership-v1"
EXCLUSION_COLUMNS = ("sample_id", "source_path", "label_original", "reason")
ARTIFACTS = (
    "manifest.csv",
    "split.json",
    "label_map.json",
    "provenance.json",
    "exclusions.csv",
    "review.json",
)


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None
    )


def _schema(value: Any, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError(f"{name} schema_version must be integer 1")
    return value


def _path(root: Path, value: Any) -> Path:
    if not _nonblank(value) or ".." in PureWindowsPath(value).parts:
        raise ValueError("invalid internal path")
    return _source_path(root, value)


def _json(root: Path, name: str) -> dict[str, Any]:
    try:
        return _schema(json.loads(_path(root, name).read_text(encoding="utf-8")), name)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}: {error}") from error


def _csv(root: Path, name: str, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        with _path(root, name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(f"{name} missing, duplicate or unexpected CSV headers")
            rows = list(reader)
            if any(None in row or None in row.values() for row in rows):
                raise ValueError(f"{name} malformed CSV row")
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid {name}: {error}") from error
    ids = [row["sample_id"] for row in rows]
    if any(not _nonblank(sample) for sample in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{name} blank or duplicate sample_id")
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _assign(
    rows: list[dict[str, Any]], seed: int, quarantine: set[str]
) -> list[dict[str, Any]]:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(rows, list):
        raise ValueError("invalid rows container")  # noqa: TRY004 - metadata errors use ValueError.
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid row object")  # noqa: TRY004 - metadata errors use ValueError.
        if not (
            (type(row.get("schema_version")) is int and row["schema_version"] == 1)
            or row.get("schema_version") == "1"
        ):
            raise ValueError("row schema_version must be 1")
        for field in ("sample_id", "class_id", "group_id"):
            if not _nonblank(row.get(field)):
                raise ValueError(f"invalid {field}")
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])
        for field in GROUP_KEYS[1:]:
            if not _hash(row.get(field)):
                raise ValueError(f"invalid {field}")
    ordered = sorted(rows, key=lambda r: r["sample_id"])
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first: dict[tuple[str, str], int] = {}
    for index, row in enumerate(ordered):
        for field in GROUP_KEYS:
            value = row[field] if field == "group_id" else row[field].lower()
            previous = first.setdefault((field, value), index)
            parent[find(index)] = find(previous)
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(ordered):
        components[find(index)].append(row)
    by_class: dict[str, list[str]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for members in components.values():
        ids = [row["sample_id"] for row in members]
        component = (
            "component:"
            + hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
        )
        classes = {row["class_id"] for row in members}
        reason = (
            "conflicting_label"
            if len(classes) > 1
            else "review_unresolved"
            if quarantine.intersection(ids)
            else ""
        )
        if not reason:
            by_class[members[0]["class_id"]].append(component)
        output.extend(
            {**row, "component_id": component, "split": "", "exclusion_reason": reason}
            for row in members
        )
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for class_id in sorted(by_class):
        component_ids = sorted(by_class[class_id])
        rng.shuffle(component_ids)
        size = len(component_ids)
        val = max(1, round(size * RATIOS["val"])) if size >= 3 else 0
        test = max(1, round(size * RATIOS["test"])) if size >= 3 else 0
        train = size - val - test
        for index, component in enumerate(component_ids):
            assignments[component] = (
                "train" if index < train else "val" if index < train + val else "test"
            )
    return sorted(
        [{**row, "split": assignments.get(row["component_id"], "")} for row in output],
        key=lambda r: r["sample_id"],
    )


def assign_splits(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Copy rows, union verified identity keys transitively, then stratify components."""
    return _assign(rows, seed, set())


def _inherit_assignments(
    rows: list[dict[str, Any]], membership: dict[str, str]
) -> list[dict[str, Any]]:
    """Preserve reviewed membership only when each connected component stays intact."""
    if (
        not isinstance(membership, dict)
        or set(membership) != {row["sample_id"] for row in rows}
        or any(value not in SPLITS for value in membership.values())
        or any(row["exclusion_reason"] for row in rows)
    ):
        raise ValueError("inherited membership must cover exactly the accepted samples")
    components: dict[str, str] = {}
    for row in rows:
        split = membership[row["sample_id"]]
        if components.setdefault(row["component_id"], split) != split:
            raise ValueError(
                "inherited membership crosses a group/byte/pixel component"
            )
    return [{**row, "split": membership[row["sample_id"]]} for row in rows]


def _labels(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    mapping = _json(root, "label_map.json")
    classes = mapping.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("label_map classes missing")
    known = {c: label for c, label, _ in CLASSES}
    known_indices = {c: index for index, (c, _, _) in enumerate(CLASSES)}
    ids: list[str] = []
    labels: dict[str, str] = {}
    for index, item in enumerate(classes):
        if (
            not isinstance(item, dict)
            or type(item.get("index")) is not int
            or item["index"] != index
            or not _nonblank(item.get("class_id"))
            or not _nonblank(item.get("label_original"))
        ):
            raise ValueError("label_map invalid class mapping")
        class_id = item["class_id"]
        if (
            class_id in ids
            or item["label_original"] in labels.values()
            or (class_id in known and item["label_original"] != known[class_id])
            or (class_id in known_indices and index != known_indices[class_id])
        ):
            raise ValueError("label_map conflicting class mapping")
        ids.append(class_id)
        labels[class_id] = item["label_original"]
    for row in rows:
        if row["class_id"] not in labels or labels[row["class_id"]] != row.get(
            "label_original"
        ):
            raise ValueError("label_map missing class or label mismatch")
        _path(root, row.get("source_path"))
        if row.get("derived_path"):
            _path(root, row["derived_path"])
    return ids


def _ids(value: Any, name: str, valid: set[str]) -> set[str]:
    if (
        not isinstance(value, list)
        or any(not _nonblank(item) for item in value)
        or len(set(value)) != len(value)
        or not set(value).issubset(valid)
    ):
        raise ValueError(f"review {name} invalid or duplicate sample_ids")
    return set(value)


def _provenance(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    provenance = _json(root, "provenance.json")
    source = provenance.get("source_root")
    if (
        not isinstance(source, str)
        or not source.strip()
        or "\x00" in source
        or Path(source).is_absolute()
        or PureWindowsPath(source).drive
        or PureWindowsPath(source).root
    ):
        raise ValueError("provenance source_root must be relative")
    try:
        source_root = (root / source).resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError("provenance source_root must be a directory")
    except OSError as error:
        raise ValueError("provenance source_root is unavailable") from error
    input_hash = provenance.get("input_manifest_sha256")
    if not isinstance(input_hash, str) or not _hash(input_hash):
        raise ValueError("provenance input_manifest_sha256 invalid")
    _validate_provenance(provenance, source_root, input_hash)
    if any(
        row[field] != provenance[field]
        for row in rows
        for field in ("source_dataset", "source_version")
    ):
        raise ValueError("provenance source identity does not match manifest")
    return provenance


def _review(
    root: Path, review: dict[str, Any], rows: list[dict[str, Any]], source_hash: str
) -> list[str]:
    _schema(review, "review")
    if (
        any(not _nonblank(review.get(key)) for key in ("reviewer", "method"))
        or review.get("expert_verified") is not False
    ):
        raise ValueError("review requires reviewer, method, expert_verified=false")
    checked = review.get("checked_at")
    if not isinstance(checked, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", checked):
        raise ValueError("review checked_at must be ISO date")
    try:
        date.fromisoformat(checked)
    except ValueError as error:
        raise ValueError("review checked_at invalid") from error
    if not _hash(source_hash) or review.get("input_manifest_sha256") != source_hash:
        raise ValueError("review input_manifest_sha256 mismatch")
    for key in ("notes", "limitations"):
        value = review.get(key)
        if not isinstance(value, str) and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise ValueError(f"review {key} must be text or a list of text")
    valid = {row["sample_id"] for row in rows}
    reviewed = _ids(review.get("sample_ids"), "sample_ids", valid)
    _ids(review.get("quarantine_sample_ids"), "quarantine_sample_ids", valid)
    sheets = review.get("contact_sheets")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("review contact_sheets required")
    files: list[str] = []
    displayed: set[str] = set()
    for sheet in sheets:
        if not isinstance(sheet, dict) or not _hash(sheet.get("sha256")):
            raise ValueError("review sheet hash invalid")
        try:
            path = _path(root, sheet.get("file"))
            if (
                not Path(sheet["file"]).as_posix().startswith("review-sheets/")
                or path == root / "review-sheets"
                or not path.is_file()
                or _sha256(path) != sheet["sha256"].lower()
            ):
                raise ValueError("sheet missing or hash mismatch")
        except (ValueError, OSError) as error:
            raise ValueError(f"review evidence: {error}") from error
        if sheet["file"] in files:
            raise ValueError("review duplicate sheet")
        files.append(sheet["file"])
        displayed.update(_ids(sheet.get("sample_ids"), "sheet sample_ids", valid))
    if displayed != reviewed:
        raise ValueError("review sheet sample_ids do not match review sample_ids")
    for class_id in {row["class_id"] for row in rows}:
        members = {row["sample_id"] for row in rows if row["class_id"] == class_id}
        if len(members.intersection(reviewed)) < min(20, len(members)):
            raise ValueError(f"review coverage insufficient for {class_id}")
    return files


def _summary(
    rows: list[dict[str, Any]], classes: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    counts: dict[str, Any] = {}
    warnings = []
    for class_id in classes:
        counts[class_id] = {}
        for split in SPLITS:
            members = [
                r for r in rows if r["class_id"] == class_id and r["split"] == split
            ]
            counts[class_id][split] = {
                "images": len(members),
                "components": len({r["component_id"] for r in members}),
            }
        if counts[class_id]["test"]["components"] < 30:
            warnings.append(
                f"low_test_support:{class_id}:fewer_than_30_components; not field-representative"
            )
    quarantine: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["exclusion_reason"]:
            item = quarantine.setdefault(
                row["component_id"],
                {
                    "component_id": row["component_id"],
                    "reason": row["exclusion_reason"],
                    "sample_ids": [],
                },
            )
            item["sample_ids"].append(row["sample_id"])
    return (
        counts,
        sorted(quarantine.values(), key=lambda item: item["component_id"]),
        warnings,
    )


def _support(counts: dict[str, Any]) -> None:
    if not counts or any(
        counts[c][s]["components"] < 1 for c in counts for s in SPLITS
    ):
        raise ValueError(
            "missing_class_support: each class requires train/val/test components"
        )


def _prepared(root: Path, review: dict[str, Any]) -> list[str]:
    split = _json(root, "split.json")
    assignments = split.get("assignments")
    if not isinstance(assignments, list) or any(
        not isinstance(r, dict) or not set(MANIFEST_COLUMNS).issubset(r)
        for r in assignments
    ):
        raise ValueError("prepare assignments missing")
    originals = [{key: row[key] for key in MANIFEST_COLUMNS} for row in assignments]
    seed = split.get("seed")
    if type(seed) is not int:
        raise ValueError("prepare seed missing or invalid")
    expected = assign_splits(originals, seed)
    classes = _labels(root, originals)
    evidence = _review(root, review, originals, split.get("source_manifest_sha256", ""))
    if review["quarantine_sample_ids"]:
        expected = _assign(originals, seed, set(review["quarantine_sample_ids"]))
    inherited = split.get("grouping_algorithm") == INHERITED_ALGORITHM
    if inherited:
        membership = _json(root, "inherited-splits.json")
        if set(membership) != {"schema_version", "assignments"}:
            raise ValueError("inherited membership fields invalid")
        expected = _inherit_assignments(expected, membership["assignments"])
        evidence.append("inherited-splits.json")
    counts, quarantine, warnings = _summary(expected, classes)
    _support(counts)
    if (
        assignments != expected
        or split.get("counts") != counts
        or split.get("quarantine") != quarantine
        or split.get("warnings") != warnings
        or split.get("target_ratios") != (None if inherited else RATIOS)
        or split.get("grouping_algorithm")
        != (INHERITED_ALGORITHM if inherited else ALGORITHM)
    ):
        raise ValueError("prepare split assignments, rules or counts inconsistent")
    accepted = [
        {key: row[key] for key in MANIFEST_COLUMNS}
        for row in expected
        if not row["exclusion_reason"]
    ]
    if _csv(root, "manifest.csv", MANIFEST_COLUMNS) != accepted:
        raise ValueError("prepare manifest assignment mismatch")
    exclusions = _csv(root, "exclusions.csv", EXCLUSION_COLUMNS)
    by_id = {row["sample_id"]: row for row in exclusions}
    if {r["sample_id"] for r in accepted}.intersection(by_id):
        raise ValueError("prepare excluded records cannot be revived")
    for row in expected:
        if row["exclusion_reason"] and by_id.get(row["sample_id"]) != {
            "sample_id": row["sample_id"],
            "source_path": row["source_path"],
            "label_original": row["label_original"],
            "reason": row["exclusion_reason"],
        }:
            raise ValueError("prepare quarantine exclusion mismatch")
    if _json(root, "review.json") != review:
        raise ValueError("review differs from prepared review")
    provenance = _provenance(root, originals)
    if (
        provenance.get("audit_stage") != "M2-sealed"
        or provenance.get("split_sealed") is not True
    ):
        raise ValueError("prepare provenance stage mismatch")
    return evidence


def seal_dataset(dataset_dir: Path, review: dict[str, Any]) -> Path:
    """Seal only prepared, consistent metadata with review evidence and class support."""
    root = dataset_dir.resolve()
    seal_path = root / "seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        raise ValueError("dataset already has a seal")
    if not (root / "split.json").is_file():
        raise ValueError("prepare dataset before seal")
    evidence = _prepared(root, review)
    hashes = {name: _sha256(_path(root, name)) for name in (*ARTIFACTS, *evidence)}
    _write_json(
        seal_path,
        {
            "schema_version": 1,
            "hashes": hashes,
            "reviewer": review["reviewer"],
            "checked_at": review["checked_at"],
            "method": review["method"],
            "expert_verified": False,
            "technical_gate": "pass",
        },
    )
    return seal_path


def verify_seal(dataset_dir: Path) -> dict[str, Any]:
    """Validate schema, confined paths, covered hashes and the technical gate."""
    root = dataset_dir.resolve()
    seal = _json(root, "seal.json")
    hashes = seal.get("hashes")
    if not isinstance(hashes, dict) or not set(ARTIFACTS).issubset(hashes):
        raise ValueError("seal_hash: missing covered artifacts")
    for name, expected in hashes.items():
        try:
            path = _path(root, name)
            if not _hash(expected) or not path.is_file() or _sha256(path) != expected:
                raise ValueError("covered artifact missing or changed")
        except (ValueError, OSError) as error:
            raise ValueError(f"seal_hash: {name}: {error}") from error
    review = _json(root, "review.json")
    evidence = _prepared(root, review)
    if set(hashes) != set(ARTIFACTS).union(evidence):
        raise ValueError("seal_hash: evidence coverage mismatch")
    if (
        seal.get("technical_gate") != "pass"
        or seal.get("expert_verified") is not False
        or any(
            seal.get(key) != review[key] for key in ("reviewer", "checked_at", "method")
        )
    ):
        raise ValueError("seal review or technical gate invalid")
    return seal


def prepare_dataset(
    dataset_dir: Path,
    review: dict[str, Any],
    seed: int = 42,
    *,
    inherited_splits: dict[str, str] | None = None,
) -> Path:
    """Validate and stage M2 metadata, then publish atomic files with the seal last.

    shortcut: publication is per-file; interrupted datasets have no valid seal and consumers must verify_seal.
    """
    root = dataset_dir.resolve()
    if (root / "seal.json").exists() or (root / "seal.json").is_symlink():
        raise ValueError("dataset already has a seal")
    if any(
        (root / name).exists()
        for name in ("split.json", "review.json", "inherited-splits.json")
    ):
        raise ValueError("dataset already prepared or publication interrupted")
    m1_files = ("manifest.csv", "exclusions.csv", "label_map.json", "provenance.json")
    try:
        before = {name: _sha256(_path(root, name)) for name in m1_files}
    except OSError as error:
        raise ValueError(f"prepare missing M1 artifact: {error}") from error
    rows = _csv(root, "manifest.csv", MANIFEST_COLUMNS)
    source_hash = before["manifest.csv"]
    assigned = assign_splits(rows, seed)
    classes = _labels(root, rows)
    exclusions = _csv(root, "exclusions.csv", EXCLUSION_COLUMNS)
    if {r["sample_id"] for r in exclusions}.intersection(r["sample_id"] for r in rows):
        raise ValueError("excluded records cannot be revived")
    provenance = _provenance(root, rows)
    if (
        provenance.get("audit_stage") != "M1-only"
        or provenance.get("split_sealed") is not False
    ):
        raise ValueError("prepare requires M1-only unsealed provenance")
    if any(row["split"] for row in rows):
        raise ValueError("prepare requires unsplit M1 manifest")
    evidence = _review(root, review, rows, source_hash)
    if review["quarantine_sample_ids"]:
        assigned = _assign(rows, seed, set(review["quarantine_sample_ids"]))
    if inherited_splits is not None:
        assigned = _inherit_assignments(assigned, inherited_splits)
    counts, quarantine, warnings = _summary(assigned, classes)
    _support(counts)
    originals = (*m1_files, *evidence)
    before.update({name: _sha256(_path(root, name)) for name in evidence})
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}-m2-", dir=root.parent)
    ).resolve()
    try:
        for name in originals:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_path(root, name).read_bytes())
        _write_csv(
            staging / "manifest.csv",
            [r for r in assigned if not r["exclusion_reason"]],
            MANIFEST_COLUMNS,
        )
        additions = [
            {
                "sample_id": r["sample_id"],
                "source_path": r["source_path"],
                "label_original": r["label_original"],
                "reason": r["exclusion_reason"],
            }
            for r in assigned
            if r["exclusion_reason"]
        ]
        _write_csv(
            staging / "exclusions.csv", exclusions + additions, EXCLUSION_COLUMNS
        )
        _write_json(
            staging / "split.json",
            {
                "schema_version": 1,
                "source_manifest_sha256": source_hash,
                "seed": seed,
                "target_ratios": RATIOS if inherited_splits is None else None,
                "grouping_algorithm": ALGORITHM
                if inherited_splits is None
                else INHERITED_ALGORITHM,
                "assignments": assigned,
                "counts": counts,
                "quarantine": quarantine,
                "warnings": warnings,
            },
        )
        _write_json(staging / "review.json", review)
        inherited_files = () if inherited_splits is None else ("inherited-splits.json",)
        if inherited_splits is not None:
            _write_json(
                staging / "inherited-splits.json",
                {"schema_version": 1, "assignments": inherited_splits},
            )
        _write_json(
            staging / "provenance.json",
            {**provenance, "audit_stage": "M2-sealed", "split_sealed": True},
        )
        seal_dataset(staging, review)
        verify_seal(staging)
        with exclusive_writer(root / ".m2-publish.lock"):
            if any(
                _sha256(_path(root, name)) != value for name, value in before.items()
            ):
                raise ValueError("input changed during prepare")
            if (root / "seal.json").exists() or (root / "seal.json").is_symlink():
                raise ValueError("dataset already has a seal")
            for name in (
                "manifest.csv",
                "exclusions.csv",
                "split.json",
                "review.json",
                "provenance.json",
                *inherited_files,
                "seal.json",
            ):
                os.replace(staging / name, root / name)
    finally:
        if staging.is_relative_to(root.parent) and staging.name.startswith(
            f".{root.name}-m2-"
        ):
            shutil.rmtree(staging)
    verify_seal(root)
    return root / "seal.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        review = json.loads(args.review.read_text(encoding="utf-8"))
        seal = prepare_dataset(args.dataset, review, args.seed)
    except (ValueError, OSError) as error:
        print(f"split: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"seal": str(seal), "technical_gate": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
