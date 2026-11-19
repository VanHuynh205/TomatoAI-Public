"""Consolidate known exposures without opening images or granting split admission."""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STUDY = HERE.name
sys.path.insert(0, str(ROOT))

protect = import_module(f"docs.progress.{STUDY}.protect_splits")
near = import_module(f"docs.research.{STUDY}.fieldplant_near_duplicates")
PV_MANIFEST = "infra/tomato_plantvillage/TomatoAI/data/processed/plantvillage/pv-d9e4c21eec8e9172/manifest.csv"
SICHUAN_GROUP = "sichuan-c2x8rynybg-v1-single-greenhouse"
EXPOSURE_FLAGS = {"baseline_exposure", "plantdoc_prior_review", "prior_visual_audit", "prior_audit_exact"}
PIN_ALIASES = {
    "fieldplant_coco": f"docs/research/{STUDY}/fieldplant-metadata/train_annotations.coco.json",
    "sichuan_annotations": f"docs/research/{STUDY}/sichuan-v1-metadata/annotations.json",
}


def add_record(records, id_, byte_hash, keys=()):
    if not id_ or not re.fullmatch(r"[0-9a-f]{64}", byte_hash):
        raise ValueError("Missing record ID or invalid image SHA-256")
    row = records.setdefault(id_, {"id": id_, "sha256": byte_hash, "keys": set()})
    if row["sha256"] != byte_hash:
        raise ValueError(f"Conflicting image hash for {id_}")
    row["keys"].add(("byte", byte_hash))
    row["keys"].update((kind, value) for kind, value in keys if value)


def exclusion_sets(records, relations, training_seeds, holdout_seeds):
    """Apply the existing transitive guard separately to each protection purpose."""
    owners = {}
    edges = list(relations)
    for id_, row in records.items():
        for key in sorted(row["keys"]):
            edges.append((owners.setdefault(key, id_), id_))
    rows = [{"id": id_} for id_ in sorted(records)]
    return {
        "blocked_from_training": sorted(protect.closure(rows, training_seeds, edges)),
        "excluded_from_final_holdout": sorted(protect.closure(rows, holdout_seeds, edges)),
    }


def write_immutable(path, result):
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Exposure inputs changed; use a new reviewed version")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def build():
    sources, _, relations, inputs = near.audit_inputs()
    records = {}
    training, holdout = defaultdict(set), defaultdict(set)
    expected = {}

    def read(relative, *, pin_root=""):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Exposure metadata escaped workspace")
        raw = path.read_bytes()
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if isinstance(data, dict):
            for field in ("inputs_sha256", "input_sha256", "source_sha256"):
                pins = data.get(field, {})
                if isinstance(pins, dict):
                    for name, digest in pins.items():
                        name = PIN_ALIASES.get(name, name)
                        if "/" not in name and "\\" not in name:
                            name = (Path(pin_root) / name).as_posix()
                        if name in expected and expected[name] != digest:
                            raise ValueError(f"Conflicting metadata pins: {name}")
                        expected[name] = digest
        return data

    def seed(target, ids, reason):
        for id_ in ids:
            target[id_].add(reason)

    def bryan_id(export, member):
        return f"bryan:{export}:{member}"

    for id_, source in sources.items():
        add_record(records, id_, source.sha256, source.keys)
        if "protected_split" in source.flags:
            seed(training, [id_], "old_validation_or_test_protection")
        for flag in source.flags & EXPOSURE_FLAGS:
            seed(holdout, [id_], flag)

    # Include all PV variants so a third variant cannot reconnect two split groups.
    inputs[PV_MANIFEST] = protect.sha256(ROOT / PV_MANIFEST)
    for row in protect.table(ROOT / PV_MANIFEST):
        add_record(records, row["record_id"], row["sha256"], (
            ("pixel", row["pixel_sha256"]), ("pv_uuid", row["source_uuid"]),
            ("pv_leaf_group", row["leaf_group_id"]), ("pv_component", row["split_component_id"]),
        ))

    progress = f"docs/progress/{STUDY}/"
    research = f"docs/research/{STUDY}/"
    derived = "infra/derived/"
    baseline = read(progress + "baseline-exposure.json")
    protected = read(progress + "protected-splits.json")
    for row in baseline["source_records"]:
        add_record(records, row["sample_id"], row["source_sha256"], (
            ("pixel", row["pixel_sha256"]), ("pixel", row.get("legacy_source_pixel_sha256")),
        ))
    seed(training, (f"plantdoc:{id_}" for id_ in protected["plantdoc_train_protected_ids"]), "old_validation_or_test_protection")
    seed(training, (row["sample_id"] for row in protected["plantvillage_protected_records"]), "old_validation_or_test_protection")
    seed(holdout, (row["sample_id"] for row in baseline["source_records"]), "baseline_fitting_or_selection")

    # These links may now reach PV variants outside the prior near-duplicate audit.
    for row in protect.table(ROOT / derived / "plantdoc-clean-v1/pv-pairs.csv"):
        relations.append((f"plantdoc:{row['id']}", row["pv_sample_id"]))
    field_near = read(derived + "yolo26s-fieldplant-near-duplicates-v2/candidate-audit.json")
    relations.extend((edge["left"], edge["right"]) for edge in field_near["edges"])
    supplement = read(progress + "exposure-supplement-v1.json")
    seed(holdout, supplement["known_relation_closure_ids"], "prior_exposure_supplement")
    for id_, row in supplement["source_records"].items():
        if records[id_]["sha256"] != row["sha256"]:
            raise ValueError(f"Supplement source hash changed: {id_}")

    for version in (1, 2):
        selection = read(derived + f"yolo26s-fieldplant-leaf-review-v{version}/selection.json",
                         pin_root=derived + "yolo26s-fieldplant-source-audit-v1")
        seed(holdout, (row["id"] for row in selection["records"]), f"fieldplant_visual_review_v{version}")
        seed(holdout, selection.get("final_holdout_excluded_ids", []), f"fieldplant_review_reservation_v{version}")

    files = read(research + "sichuan-v1-metadata/files.json")
    for row in files:
        if row["release_path"].startswith("images/"):
            add_record(records, "sichuan-" + Path(row["filename"]).stem,
                       row["content_details"]["sha256_hash"], (("sichuan_site", SICHUAN_GROUP),))
    admission = read(derived + "yolo26s-sichuan-leaf-subset-v1/admission-disease-only-v3.json")
    if admission["source_policy"] != {"group": SICHUAN_GROUP, "role": "train_only", "final_holdout_eligible": False}:
        raise ValueError("Sichuan source policy changed")
    for row in admission["records"]:
        add_record(records, row["id"], row["image_sha256"])
    seed(holdout, (row["id"] for row in admission["records"]), "candidate_training_admission")
    reviewed = read(derived + "yolo26s-sichuan-leaf-subset-v1/candidates.json",
                    pin_root=research + "sichuan-v1-metadata")
    seed(holdout, (row["id"] for row in reviewed["records"]), "sichuan_gray_mold_visual_review")
    reviewed = read(derived + "yolo26s-sichuan-wilt-review-v1/selection.json",
                    pin_root=research + "sichuan-v1-metadata")
    seed(holdout, reviewed["ids"], "sichuan_wilt_review_reservation")
    pilot = read(derived + "yolo26s-field-source-audit-v1/selection.json")
    seed(holdout, (row["id"] for row in pilot["records"]), "initial_source_visual_audit")

    bryan = derived + "yolo26s-bryan-export-comparison-v1/"
    audit = read(bryan + "audit.json")
    for export, data in audit["exports"].items():
        for row in data["records"]:
            add_record(records, bryan_id(export, row["member"]), row["sha256"], (
                ("bryan_filename_family", row["filename_key"]),
            ))
    for row in supplement["bryan_existing_export_exclusions"]:
        id_ = bryan_id(row["export"], row["member"])
        add_record(records, id_, row["sha256"])
        seed(holdout, [id_], "bryan_prior_review_quarantine")
    links = read(bryan + "pixel-links-v1.json")
    if links["metadata_audit_sha256"] != inputs[bryan + "audit.json"]:
        raise ValueError("Bryan correspondence references changed metadata")
    for row in links["records"]:
        for match in row["matched_references"]:
            if match["pixel_comparison"]["accepted"] is not True:
                raise ValueError("Unaccepted Bryan pixel match")
            left, right = bryan_id("fork_v1", row["fork_member"]), bryan_id("v63", match["member"])
            add_record(records, left, row["fork_sha256"])
            add_record(records, right, match["sha256"])
            relations.append((left, right))
    review = read(bryan + "visual-review-v1/review.json")
    for row in review["quarantined_records"]:
        seed(holdout, [bryan_id(row["export"], row["member"])], "bryan_visual_review")
    detail = read(bryan + "detail-review-v1/review.json")
    for row in detail["records"]:
        left = bryan_id("fork_v1", row["fork_member"])
        seed(holdout, [left], "bryan_detail_review")
        for match in row["similarity_candidates"]:
            add_record(records, match["id"], match["sha256"])
            relations.append((left, match["id"]))
            seed(holdout, [match["id"]], "bryan_reference_review")

    imtiaz = derived + "yolo26s-imtiaz-official-export-v1/"
    audit = read(imtiaz + "metadata-audit.json")
    for row in audit["records"]:
        add_record(records, "imtiaz:" + row["member"], row["sha256"], (
            ("imtiaz_filename_family", row["source_filename_key"]),
        ))
    review = read(imtiaz + "visual-review-v1/review.json")
    selection = read(imtiaz + "visual-review-v1/selection.json")
    if review["selection_sha256"] != inputs[imtiaz + "visual-review-v1/selection.json"] or not selection:
        raise ValueError("Imtiaz visual selection changed")
    for row in review["final_holdout_excluded_members"]:
        id_ = "imtiaz:" + row["member"]
        add_record(records, id_, row["sha256"])
        seed(holdout, [id_], "imtiaz_visual_review_family")

    for name, expected_sha in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid pinned metadata path: {name}")
        actual = protect.sha256(path)
        if actual != expected_sha:
            raise ValueError(f"Pinned input changed: {name}")
        inputs[path.relative_to(ROOT).as_posix()] = actual
    inputs[Path(__file__).relative_to(ROOT).as_posix()] = protect.sha256(Path(__file__))
    inputs[Path(protect.__file__).relative_to(ROOT).as_posix()] = protect.sha256(Path(protect.__file__))
    result = exclusion_sets(records, relations, training, holdout)
    return {
        "schema_version": 1,
        "status": "known_exposure_closure_not_split_admission",
        "training_ready": False,
        "final_holdout_established": False,
        "inputs_sha256": dict(sorted(inputs.items())),
        "summary": {"records": len(records), **{key: len(value) for key, value in result.items()},
                    "records_by_source": dict(sorted(Counter(id_.split(':')[0].split('-')[0] for id_ in records).items()))},
        **result,
        "training_seed_reasons": {id_: sorted(reasons) for id_, reasons in sorted(training.items())},
        "holdout_seed_reasons": {id_: sorted(reasons) for id_, reasons in sorted(holdout.items())},
        "explicit_relations": sorted(set(relations)),
        "records": [{**row, "keys": sorted(row["keys"])} for _, row in sorted(records.items())],
        "limitations": [
            "Not being excluded does not grant training or holdout eligibility; annotation and source admission still apply.",
            "Known metadata relations only; undiscovered crop, scene, plant and cross-source relations remain possible.",
            "Bryan/Imtiaz filename families and three Bryan/PlantDoc similarity links are conservative exclusions, not identity certificates.",
            "Sichuan shares the already declared single-greenhouse train-only group; this does not prove all images show one plant.",
            "Quality/conflict quarantine remains in source admission ledgers and must be enforced separately.",
            "This snapshot does not yet enforce a Colab training gate or certify an independent final holdout.",
        ],
    }


if __name__ == "__main__":
    result = build()
    digest = write_immutable(HERE / "master-exposure-v1.json", result)
    print(json.dumps({**result["summary"], "sha256": digest}, indent=2))
