"""Build exclusion metadata without opening protected images or changing old seals."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def closure(rows, seeds, edges):
    adjacency = defaultdict(set)
    index = defaultdict(list)
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or not set(seeds) <= ids:
        raise ValueError("duplicate or unknown record IDs")
    for row in rows:
        for field in ("id", "component_id", "v1_component_id", "sha256", "pixel_sha256"):
            value = row.get(field)
            if value:
                family = "component" if field in {"id", "component_id", "v1_component_id"} else field
                index[family, value].append(row["id"])
    pairs = [(members[0], member) for members in index.values() for member in members[1:]]
    pairs.extend(edges)
    for left, right in pairs:
        if left not in ids or right not in ids:
            raise ValueError("relationship references unknown record")
        adjacency[left].add(right)
        adjacency[right].add(left)
    protected, pending = set(seeds), list(seeds)
    while pending:
        node = pending.pop()
        for other in adjacency[node] - protected:
            protected.add(other)
            pending.append(other)
    return protected


def main():
    inputs = {}

    def read(relative):
        path = ROOT / relative
        inputs[relative] = sha256(path)
        return table(path)

    split_dir = ROOT / "infra/derived/plantdoc-split-v1"
    lock_path = split_dir / "lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in lock["artifact_sha256"].items():
        if sha256(split_dir / relative) != expected:
            raise ValueError(f"PlantDoc locked artifact changed: {relative}")
    # Validate metadata, without opening the review-sheet images listed in the seal.
    combined_dir = ROOT / "infra/derived/pv8-plantdoc-v1"
    seal_path = combined_dir / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for relative in ("manifest.csv", "split.json", "inherited-splits.json"):
        if sha256(combined_dir / relative) != seal["hashes"][relative]:
            raise ValueError(f"Combined sealed artifact changed: {relative}")
    rows = read("infra/derived/plantdoc-folder-review-v2/manifest.csv")
    train = read("infra/derived/plantdoc-split-v1/train.csv")
    val = read("infra/derived/plantdoc-split-v1/val.csv")
    reserved = read("infra/derived/plantdoc-split-v1/reserved_test.csv")
    combined = read("infra/derived/pv8-plantdoc-v1/manifest.csv")
    pv_full = read("infra/tomato_plantvillage/TomatoAI/data/processed/plantvillage/pv-d9e4c21eec8e9172/manifest.csv")
    pairs = []
    for relative in ("infra/derived/plantdoc-folder-review-v2/review-links.csv", "infra/derived/plantdoc-clean-v1/related-pairs.csv"):
        pairs.extend((row["left"], row["right"]) for row in read(relative))
    cross = read("infra/derived/plantdoc-clean-v1/pv-pairs.csv")
    for relative, digest in inputs.items():
        expected = lock["input_sha256"].get(relative)
        if expected is not None and digest != expected:
            raise ValueError(f"PlantDoc locked input changed: {relative}")

    pv_protected = {row["sample_id"]: row for row in combined if row["sample_id"].startswith("pv:") and row["split"] in {"val", "test"}}
    pv_index = {row["record_id"]: row for row in pv_full}
    if not pv_protected.keys() <= pv_index.keys():
        raise ValueError("protected PlantVillage record missing from source metadata")
    test_seeds = {row["id"] for row in rows if row["source_split"] == "test"}
    test_seeds.update(row["id"] for row in reserved)
    test_seeds.update(row["id"] for row in cross if row["pv_sample_id"] in pv_protected and pv_protected[row["pv_sample_id"]]["split"] == "test")
    protected_test = closure(rows, test_seeds, pairs)
    all_seeds = test_seeds | {row["id"] for row in val}
    all_seeds.update(row["id"] for row in cross if row["pv_sample_id"] in pv_protected)
    protected_train = closure(rows, all_seeds, pairs)
    allowed = [row for row in train if row["id"] not in protected_train]
    eligible_val = [row for row in val if row["id"] not in protected_test]
    protected_rows = [row for row in rows if row["id"] in protected_train]
    pv_records = [{"sample_id": record_id, "split": split_row["split"], **{key: pv_index[record_id][key] for key in ("source_file", "source_uuid", "leaf_group_id", "split_component_id", "duplicate_of", "sha256", "pixel_sha256")}} for record_id, split_row in sorted(pv_protected.items())]
    hashes = {row[key] for row in [*pv_records, *protected_rows] for key in ("sha256",) if row.get(key)}
    pixels = {row["pixel_sha256"] for row in [*pv_records, *protected_rows] if row.get("pixel_sha256")}
    result = {
        "schema_version": 1,
        "purpose": "Exclude original TEST, old validation, known related groups and pixel/byte duplicates from new training.",
        "inputs_sha256": {**inputs, str(lock_path.relative_to(ROOT)).replace("\\", "/"): sha256(lock_path), str(seal_path.relative_to(ROOT)).replace("\\", "/"): sha256(seal_path)},
        "plantdoc_test_protected_ids": sorted(protected_test),
        "plantdoc_train_protected_ids": sorted(protected_train),
        "plantdoc_allowed_train": allowed,
        "plantdoc_allowed_development": eligible_val,
        "plantdoc_protected_records": protected_rows,
        "plantvillage_protected_records": pv_records,
        "protected_byte_sha256": sorted(hashes),
        "protected_pixel_sha256": sorted(pixels),
        "limitations": ["Metadata-only closure covers already recorded relations, not undiscovered near duplicates.", "Classifier screening does not establish complete detection annotations.", "PlantVillage contribution through baseline initialization inherits its train/development history."],
    }
    output = HERE / "protected-splits.json"
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError("Protected metadata changed; create a new version for review")
    if not output.exists():
        output.write_text(content, encoding="utf-8")
    print(json.dumps({"plantdoc_test_protected": len(protected_test), "plantdoc_train_protected": len(protected_train), "plantdoc_allowed_train": len(allowed), "plantdoc_allowed_development": len(eligible_val), "plantvillage_protected": len(pv_records), "output_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
