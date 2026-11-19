"""Check prospective splits against known exposures; never grant training admission."""

import hashlib
import json
from importlib import import_module
from pathlib import Path

STUDY = "yolo26s-field-colab-2026-11-10"
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
master_module = import_module(f"docs.progress.{STUDY}.master_exposure")
MASTER = f"docs/progress/{STUDY}/master-exposure-v1.json"
BRYAN = "infra/derived/yolo26s-bryan-export-comparison-v1/audit.json"
PINS = {
    MASTER: "c3e61e3a43d3e93271a45efa0eeb132dbe04ff8764a2422bd28a8c7e2d460dcf",
    BRYAN: "b73e87c4fa454d1746782bd378e272507505200649ff7ef15f62d93facd72e97",
    "infra/derived/yolo26s-plantdoc-disease-only-v2/manifest.json": "b2a2074083e1728fca78cb30ac8c1682f6ae0e44064fc99931c0ee6cb56e25eb",
    "infra/derived/yolo26s-sichuan-leaf-subset-v1/admission-disease-only-v3.json": "70600fb4821955bb425cf04c939a13b7a82ec0b4d17446da15f555abe95aca75",
}


def record_index(master):
    rows = master["records"]
    index = {row["id"]: {**row, "keys": {tuple(key) for key in row["keys"]}} for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate master record ID")
    return index


def training_quarantine(master, audit):
    """Unresolved filename candidates restrict training without claiming identity."""
    index = record_index(master)
    seeds = set()
    for export, overlap in audit["overlap_with_existing_data"].items():
        export_rows = {row["member"]: row for row in audit["exports"][export]["records"]}
        for member in overlap["protected_plantdoc_filename_candidates"]:
            id_ = f"bryan:{export}:{member}"
            if member not in export_rows or id_ not in index:
                raise ValueError("Unknown Bryan quarantine candidate")
            if index[id_]["sha256"] != export_rows[member]["sha256"]:
                raise ValueError("Bryan candidate image hash mismatch")
            seeds.add(id_)
    result = master_module.exclusion_sets(index, master["explicit_relations"], seeds, set())
    return {"seed_ids": sorted(seeds), "quarantined_from_training": result["blocked_from_training"]}


def validate_records(master, quarantine, rows):
    """Validate canonical source IDs/hashes and known split relationships only."""
    index = record_index(master)
    splits = {name: set() for name in ("train", "development", "final_holdout")}
    seen = set()
    for row in rows:
        id_ = row["id"]
        if id_ in seen or id_ not in index or row["split"] not in splits:
            raise ValueError("Unknown or duplicate source ID, or invalid split")
        if row["sha256"] != index[id_]["sha256"]:
            raise ValueError(f"Source image hash mismatch: {id_}")
        seen.add(id_)
        splits[row["split"]].add(id_)
    blocked = set(master["blocked_from_training"]) | set(quarantine["quarantined_from_training"])
    if splits["train"] & blocked:
        raise ValueError("Training contains protected or unresolved Bryan candidate")
    related = master_module.exclusion_sets(
        index, master["explicit_relations"], splits["train"], splits["train"] | splits["development"],
    )
    if splits["development"] & set(related["blocked_from_training"]):
        raise ValueError("Training and development share a known related group")
    final_excluded = set(master["excluded_from_final_holdout"]) | set(related["excluded_from_final_holdout"])
    if splits["final_holdout"] & final_excluded:
        raise ValueError("Final holdout contains known exposure or candidate development relatives")
    return {name: len(ids) for name, ids in splits.items()}


def load_inputs():
    loaded = {}
    for name, digest in PINS.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Pinned input changed: {name}")
        loaded[name] = json.loads(raw)
    for name, digest in loaded[MASTER]["inputs_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or master_module.protect.sha256(path) != digest:
            raise ValueError(f"Master dependency changed: {name}")
    return loaded


def main():
    inputs = load_inputs()
    master = inputs[MASTER]
    quarantine = training_quarantine(master, inputs[BRYAN])
    plantdoc = inputs["infra/derived/yolo26s-plantdoc-disease-only-v2/manifest.json"]["records"]
    sichuan = inputs["infra/derived/yolo26s-sichuan-leaf-subset-v1/admission-disease-only-v3.json"]["records"]
    accepted = [row for row in plantdoc if row["decision"] == "accepted"]
    rows = [{"id": "plantdoc:" + row["id"], "sha256": row["source_sha256"], "split": row["split"]} for row in accepted]
    rows.extend({"id": row["id"], "sha256": row["image_sha256"], "split": "train"} for row in sichuan)
    result = {
        "status": "known_exposure_check_only_not_training_authorization",
        "training_ready": False,
        "inputs_sha256": PINS,
        "script_sha256": master_module.protect.sha256(Path(__file__)),
        "unresolved_bryan": quarantine,
        "effective_training_protection_count": len(set(master["blocked_from_training"]) | set(quarantine["quarantined_from_training"])),
        "verified_current_admission_counts": validate_records(master, quarantine, rows),
        "limitations": [
            "Filename quarantine is not proof of image identity or baseline fitting exposure.",
            "This check does not establish annotation completeness, source rights, taxonomy support or final-holdout independence.",
            "Image payloads and Colab execution are not checked; the frozen training gate remains pending.",
        ],
    }
    digest = master_module.write_immutable(HERE / "candidate-split-guard-v1.json", result)
    print(json.dumps({"sha256": digest, "quarantine_count": len(quarantine["quarantined_from_training"]),
                      "protected_train_count": result["effective_training_protection_count"],
                      "admitted_split_counts": result["verified_current_admission_counts"]}))


if __name__ == "__main__":
    main()
