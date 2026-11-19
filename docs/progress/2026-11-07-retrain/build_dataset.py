"""Build the research dataset from two reviewed, locked memberships; raw stays read-only."""

import copy
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from ai.data import CLASSES, MANIFEST_COLUMNS, _pixel_hash, _sha256, _source_path
from ai.split import (
    EXCLUSION_COLUMNS,
    _write_csv,
    _write_json,
    prepare_dataset,
    verify_seal,
)

ROOT = Path(__file__).resolve().parents[3]
INFRA = ROOT / "infra"
PV = INFRA / "derived/pv8-v1"
PD = INFRA / "derived/plantdoc-split-v1"
OUT = INFRA / "derived/pv8-plantdoc-v1"
VERSION = "pv8-plantdoc-v1-20261107"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to replace dataset: {OUT}")
    verify_seal(PV)
    lock = read_json(PD / "lock.json")
    assert lock["lock_kind"] == "plantdoc_development_split"
    inputs = {
        PV / "seal.json": _sha256(PV / "seal.json"),
        PD / "lock.json": _sha256(PD / "lock.json"),
    }
    inputs.update(
        {
            PV / name: digest
            for name, digest in read_json(PV / "seal.json")["hashes"].items()
        }
    )
    inputs.update(
        {
            _source_path(PD, name): digest
            for name, digest in lock["artifact_sha256"].items()
        }
    )
    inputs.update(
        {
            _source_path(ROOT, name): digest
            for name, digest in lock["input_sha256"].items()
        }
    )
    inputs[_source_path(ROOT, lock["generator"])] = lock["generator_sha256"]
    assert all(_sha256(path) == digest for path, digest in inputs.items())
    assert (PV / "label_map.json").read_bytes() == (PD / "label_map.json").read_bytes()
    pv_provenance = read_json(PV / "provenance.json")
    pv_raw = (PV / pv_provenance["source_root"]).resolve()
    rows, membership = [], {}
    for old in read_csv(PV / "manifest.csv"):
        row = dict(old)
        row["source_path"] = (
            _source_path(pv_raw, old["source_path"]).relative_to(INFRA).as_posix()
        )
        row["group_id"] = "pv:" + old["group_id"]
        membership[row["sample_id"]] = old["split"]
        rows.append(row)
    labels = {class_id: label for class_id, label, _ in CLASSES}
    pd_rows = read_csv(PD / "manifest.csv") + read_csv(PD / "reserved_test.csv")
    for old in pd_rows:
        assert old["label_status"] == "approved" and old["expert_verified"] == "false"
        is_test = old["intended_use"] == "reserved_test"
        assert old["intended_use"] in ("development", "reserved_test")
        sample_id = "plantdoc:" + old["id"]
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(
            schema_version="1",
            sample_id=sample_id,
            source_path=_source_path(ROOT, old["image_path"])
            .relative_to(INFRA)
            .as_posix(),
            source_sha256=old["sha256"],
            pixel_sha256=old["pixel_sha256"],
            label_original=labels[old["class_id"]],
            class_id=old["class_id"],
            group_id="plantdoc:" + old["component_id"],
            group_evidence=json.dumps(
                {
                    "original_label": old["source_label"],
                    "label_basis": old["label_basis"],
                    "review_record": old["id"],
                    "component_id": old["component_id"],
                }
            ),
        )
        membership[sample_id] = "test" if is_test else old["split"]
        rows.append(row)
    assert len(rows) == len(membership) == 11758
    assert Counter(membership.values()) == {"train": 8238, "val": 1785, "test": 1735}
    for row in rows:
        path = _source_path(INFRA, row["source_path"])
        assert _sha256(path) == row["source_sha256"], row["sample_id"]
        assert _pixel_hash(path) == (row["pixel_sha256"], ""), row["sample_id"]
        row.update(
            source_dataset="PlantVillage-PlantDoc-tomato",
            source_version=VERSION,
            split="",
        )
    rows.sort(key=lambda row: row["sample_id"])
    OUT.mkdir()
    _write_csv(OUT / "manifest.csv", rows, MANIFEST_COLUMNS)
    _write_csv(OUT / "exclusions.csv", [], EXCLUSION_COLUMNS)
    shutil.copyfile(PV / "label_map.json", OUT / "label_map.json")
    manifest_hash = _sha256(OUT / "manifest.csv")
    review = copy.deepcopy(read_json(PV / "review.json"))
    review.update(
        checked_at="2026-11-07",
        reviewer="Codex; inherited source reviews",
        method="Preserve PV8 visual screening and user-authorized PlantDoc folder/visual decisions; verify raw bytes/pixels and frozen memberships.",
        input_manifest_sha256=manifest_hash,
        quarantine_sample_ids=[],
        notes="Original PV8 sheets are copied without changing image content. Listed IDs retain only accepted samples. PlantDoc review and split references are pinned in provenance; no new disease labels are inferred.",
        limitations=[
            "expert_verified=false",
            "PV8 TEST previously consumed",
            "PlantDoc validation previously used for diagnosis",
            "PlantDoc reserve has six supported classes only",
            "Local research only; no redistribution approval",
        ],
    )
    review["sample_ids"] = [
        sample for sample in review["sample_ids"] if sample in membership
    ]
    for sheet in review["contact_sheets"]:
        sheet["sample_ids"] = [
            sample for sample in sheet["sample_ids"] if sample in membership
        ]
        target = OUT / sheet["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PV / sheet["file"], target)
    provenance = copy.deepcopy(pv_provenance)
    for evidence in provenance["license"]["evidence"]:
        evidence["file"] = (
            _source_path(pv_raw, evidence["file"]).relative_to(INFRA).as_posix()
        )
    provenance["license"]["identifier"] = (
        "PlantVillage: CC-BY-SA-3.0; PlantDoc: CC-BY-4.0"
    )
    provenance["license"]["evidence"].append(
        {
            "file": "plant_doc/LICENSE.txt",
            "sha256": _sha256(INFRA / "plant_doc/LICENSE.txt"),
            "url": "https://github.com/pratikkayal/PlantDoc-Dataset/blob/5467f6012d78d1c446145d5f582da6096f852ae8/LICENSE.txt",
            "locator": "Pinned publisher license; rights to individual web images remain unreviewed for redistribution.",
        }
    )
    provenance.update(
        source_dataset="PlantVillage-PlantDoc-tomato",
        source_version=VERSION,
        snapshot_version=VERSION,
        source_root=os.path.relpath(INFRA, OUT),
        audit_stage="M1-only",
        split_sealed=False,
        input_manifest_sha256=manifest_hash,
        counts={"accepted": len(rows), "excluded": 0},
        parent_artifacts={
            str(path.relative_to(ROOT)).replace("\\", "/"): digest
            for path, digest in inputs.items()
        },
        parent_sources={
            "pv:": pv_provenance,
            "plantdoc:": {
                "source_url": "https://github.com/pratikkayal/PlantDoc-Dataset",
                "revision": "5467f6012d78d1c446145d5f582da6096f852ae8",
                "review": "infra/derived/plantdoc-folder-review-v2",
                "split": "infra/derived/plantdoc-split-v1",
                "expert_verified": False,
            },
        },
        test_history="PV8 TEST consumed at M7; PlantDoc reserved_test remains excluded from model selection. Combining them does not create a fresh holdout.",
    )
    provenance["limitations"].append(
        "Composite source metadata inherits two reviewed source lists; excluded parent rows are not revived and remain documented in their parent datasets."
    )
    _write_json(OUT / "provenance.json", provenance)
    assert all(_sha256(path) == digest for path, digest in inputs.items())
    prepare_dataset(OUT, review, inherited_splits=membership)
    verify_seal(OUT)
    actual = read_csv(OUT / "manifest.csv")
    assert {row["sample_id"]: row["split"] for row in actual} == membership
    report = {
        "dataset": str(OUT.relative_to(ROOT)),
        "counts": dict(Counter(membership.values())),
        "raw_files_rehashed_decoded": len(rows),
        "memberships_preserved": True,
        "seal_sha256": _sha256(OUT / "seal.json"),
        "split_sha256": _sha256(OUT / "split.json"),
        "expert_verified": False,
        "source_test_history_preserved": True,
    }
    _write_json(
        Path(__file__).parent / "dataset-verification.json",
        {"schema_version": 1, **report},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
