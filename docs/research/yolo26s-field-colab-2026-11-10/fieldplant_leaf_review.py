"""Review previously exposed, nonconflicting FieldPlant whole-leaf candidates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_source_samples import tile
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "infra/derived/yolo26s-fieldplant-source-audit-v1"
OUTPUT = ROOT / "infra/derived/yolo26s-fieldplant-leaf-review-v1"
BATCH_OUTPUT = ROOT / "infra/derived/yolo26s-fieldplant-leaf-review-v2"
NEAR = ROOT / "infra/derived/yolo26s-fieldplant-near-duplicates-v2/candidate-audit.json"
NEAR_SHA256 = "e7e56e5fc07185d67e5d2b4b5f695a39e43eff842e979cfed43a812053ed1363"
PINS = {
    "source-manifest.json": "1a32e70d0931ca0636a6b49073e6231b718e307bac74e2c54bb9e23b23ce4b53",
    "conflict-resolution.json": "a132853b321e81f7b14bb724218a193b21c35e5b2fb9dc04afac875db1416fe1",
}


def candidates(manifest, ledger):
    allowed = {key for group in ledger["groups"].values() for key, status in group["records"].items() if status == "prior_exposure_group"}
    rows = [row for row in manifest["records"] if row["id"] in allowed]
    if {row["id"] for row in rows} != allowed or len(rows) != len(allowed):
        raise ValueError("Candidate source IDs missing or repeated")
    return sorted(rows, key=lambda row: row["image"]["id"])


def immutable_json(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Evidence changed; use a new version")
    if not path.exists():
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fresh_batch(manifest, ledger, near, per_label=12):
    """Reserve abundant source categories for train review before showing pixels."""
    allowed = {key for group in ledger["groups"].values() for key, status in group["records"].items() if status == "disease_visual_review_candidate"}
    by_id = {row["id"]: row for row in manifest["records"]}
    if len(by_id) != len(manifest["records"]) or not allowed <= by_id.keys():
        raise ValueError("Candidate source IDs missing or repeated")
    groups = {}
    for row in by_id.values():
        groups.setdefault(row["provisional_group"], set()).add(row["id"])
    links = list(groups.values()) + [set(group["fieldplant_records"]) for group in near["candidate_groups"]]
    blocked = set(by_id) - allowed
    for group in near["candidate_groups"]:
        if group["exclusion_witnesses"]:
            blocked.update(group["fieldplant_records"])

    def closure(ids):
        result = set(ids)
        while True:
            previous = len(result)
            for members in links:
                if result & members:
                    result.update(members)
            if len(result) == previous:
                return result

    blocked = closure(blocked)
    selected, reserved = [], set()
    ordered = sorted(by_id.values(), key=lambda row: hashlib.sha256((row["id"] + "/batch-2").encode()).hexdigest())
    for label in ("Tomato Brown Spots", "Tomato blight leaf"):
        count = 0
        for row in ordered:
            if row["id"] in blocked | reserved or not any(box["source_label"] == label for box in row["annotations"]):
                continue
            selected.append(row)
            reserved.update(closure({row["id"]}))
            count += 1
            if count == per_label:
                break
    return selected, sorted(reserved)


def batch_selection(inputs):
    raw = NEAR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEAR_SHA256:
        raise ValueError("Near-duplicate evidence changed")
    rows, reserved = fresh_batch(inputs["source-manifest.json"], inputs["conflict-resolution.json"], json.loads(raw))
    return {"purpose": "train_candidate_visual_review; not final holdout", "source_sha256": PINS,
            "near_duplicate_sha256": NEAR_SHA256, "new_audit_exposure": len(rows),
            "final_holdout_excluded_ids": reserved, "training_admitted": False,
            "final_holdout_eligible": False,
            "grouping_limit": "Known source and similarity links only; plant/session identity remains unresolved.",
            "records": rows}


def seal_review(batch=False):
    output = BATCH_OUTPUT if batch else OUTPUT
    selection_path = output / "selection.json"
    raw = selection_path.read_bytes()
    expected = "1383620527a9f46eb310900f78b239c813dfdaac5eb747efce3782eda6649fef"
    if batch:
        expected = hashlib.sha256((json.dumps(batch_selection(load_inputs()), ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Review selection changed")
    selected = json.loads(raw)["records"]
    with (output / "decisions.tsv").open(encoding="utf-8", newline="") as stream:
        decisions = list(csv.DictReader(stream, delimiter="\t"))
    if [row["id"] for row in selected] != [row["id"] for row in decisions]:
        raise ValueError("Review must cover each selected image once in order")
    records = []
    for row, decision in zip(selected, decisions):
        if not decision["observation"]:
            raise ValueError("Visual decision needs evidence")
        records.append({**decision, "source_image_sha256": row["acquisition"]["sha256"], "provisional_group": row["provisional_group"], "final_holdout_eligible": False, "training_admitted": False, "source_disease_annotations": [box for box in row["annotations"] if box["source_label"] != "Tomato healthy"]})
    qualified = [row for row in records if row["visual_status"] == "leaf_review_pass"]
    result = {
        "status": "visual_review_complete_pending_cross_source_checks_and_final_taxonomy",
        "new_audit_exposure": len(selected) if batch else 0,
        "training_admitted": False,
        "final_holdout_eligible": False,
        "leaf_review_pass_ids": [row["id"] for row in qualified],
        "leaf_review_pass_disease_boxes": sum(len(row["source_disease_annotations"]) for row in qualified),
        "class_policy": "Preserve Tomato blight leaf as a source category. No equivalence to Early Blight, Late Blight or Leaf Mold is established.",
        "evidence_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.suffix in {".jpg", ".tsv"} or path == selection_path},
        "records": records,
    }
    if batch:
        result["final_holdout_excluded_ids"] = json.loads(raw)["final_holdout_excluded_ids"]
    digest = immutable_json(output / "review.json", result)
    print(json.dumps({"leaf_review_pass": len(qualified), "disease_boxes": result["leaf_review_pass_disease_boxes"], "review_sha256": digest}))


def load_inputs():
    inputs = {}
    for name, expected in PINS.items():
        raw = (SOURCE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Source changed: {name}")
        inputs[name] = json.loads(raw)
    return inputs


def main(batch=False):
    inputs = load_inputs()
    output = BATCH_OUTPUT if batch else OUTPUT
    if batch:
        selection = batch_selection(inputs)
    else:
        rows = candidates(inputs["source-manifest.json"], inputs["conflict-resolution.json"])
        selection = {"purpose": "disease_only_whole_leaf_review", "source_sha256": PINS, "new_audit_exposure": 0, "training_admitted": False, "final_holdout_eligible": False, "records": rows}
    rows = selection["records"]
    output.mkdir(exist_ok=True)
    digest = immutable_json(output / "selection.json", selection)
    tiles = []
    for row in rows:
        path = SOURCE / row["acquisition"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["acquisition"]["sha256"]:
            raise ValueError(f"Source pixels changed: {row['id']}")
        boxes = [{"name": box["source_label"], "xyxy": [box["bbox"][0], box["bbox"][1], box["bbox"][0] + box["bbox"][2], box["bbox"][1] + box["bbox"][3]]} for box in row["annotations"]]
        tiles.append(tile({"id": row["id"], "boxes": boxes}, {"path": path.relative_to(ROOT).as_posix(), "exif_orientation": row["acquisition"]["exif_orientation"]}))
    for start in range(0, len(tiles), 6):
        path = output / f"overlay-{start // 6 + 1:02d}.jpg"
        if path.exists():
            continue
        page = Image.new("RGB", (1200, 1500), "white")
        for index, image in enumerate(tiles[start:start + 6]):
            page.paste(image, (index % 2 * 600, index // 2 * 500))
        page.save(path, quality=95)
    print(json.dumps({"selected": len(rows), "selection_sha256": digest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-review", action="store_true")
    parser.add_argument("--batch-2", action="store_true", help="Reserve 12 candidates from each abundant source category for train review")
    arguments = parser.parse_args()
    if arguments.seal_review:
        seal_review(arguments.batch_2)
    else:
        main(arguments.batch_2)
