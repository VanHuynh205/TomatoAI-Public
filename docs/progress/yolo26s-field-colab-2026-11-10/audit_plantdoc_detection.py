"""Join approved development images to original boxes; do not inspect reserved pixels."""

import csv
import hashlib
import io
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMIT = "4730a233a555b30ee98e0879c63ad25d82407455"
CSV_SHA256 = "87f50acc53fd47a3abdd68c2b9546de263a48d99b45d09ec4c6cc070f5e0506c"
PROTECTED_SHA256 = "05262148cd5ea9205192d5a45afec60141c4184191faa837cefabe07e71b0f28"
SOURCE = f"https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/{COMMIT}/train_labels.csv"
NAMES = {
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato mold leaf": "leaf_mold",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "spider_mites",
    "Tomato leaf yellow virus": "yellow_leaf_curl",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato leaf": "tomato_leaf_unspecified",
}


def key(filename):
    return unicodedata.normalize("NFC", filename).casefold().replace("?", "_")


def box_record(row, size):
    width, height = size
    if (int(row["width"]), int(row["height"])) != size:
        raise ValueError("CSV dimensions differ from source image")
    coords = [float(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")]
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("non-finite box")
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("invalid box bounds")
    return {"source_class": row["class"], "class_id": NAMES[row["class"]], "xyxy": coords, "xywhn": [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]}


def validate_member(member, split, protection):
    if split not in {"train", "development"}:
        raise ValueError("unknown audit split")
    if member["id"] in protection["plantdoc_test_protected_ids"]:
        raise ValueError("test-protected record reached source audit")
    if split == "train" and member["id"] in protection["plantdoc_train_protected_ids"]:
        raise ValueError("protected record reached training audit")
    path = ROOT / member["image_path"]
    if member["source_split"] != "train" or not path.resolve().is_relative_to((ROOT / "infra/plant_doc/train").resolve()):
        raise ValueError("attempt to open an image outside approved source TRAIN")
    if split == "train" and (member["sha256"] in protection["protected_byte_sha256"] or member["pixel_sha256"] in protection["protected_pixel_sha256"]):
        raise ValueError("protected image metadata reached training audit")
    return path


def main():
    protected_path = HERE / "protected-splits.json"
    protected_raw = protected_path.read_bytes()
    if hashlib.sha256(protected_raw).hexdigest() != PROTECTED_SHA256:
        raise ValueError("protected split snapshot checksum mismatch")
    protection = json.loads(protected_raw)
    csv_path = HERE / "plantdoc-source-train-labels.csv"
    if csv_path.exists():
        raw = csv_path.read_bytes()
    else:
        response = requests.get(SOURCE, timeout=30)
        response.raise_for_status()
        raw = response.content
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("published source CSV checksum mismatch")
    if not csv_path.exists():
        csv_path.write_bytes(raw)
    index = defaultdict(list)
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        index[key(row["filename"])].append(row)
    results = []
    for split, members in (("train", protection["plantdoc_allowed_train"]), ("development", protection["plantdoc_allowed_development"])):
        for member in members:
            path = validate_member(member, split, protection)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != member["sha256"]:
                raise ValueError("approved source image changed")
            if split == "train" and (digest in protection["protected_byte_sha256"] or member["pixel_sha256"] in protection["protected_pixel_sha256"]):
                raise ValueError("protected image reached training audit")
            labels = index[key(path.name)]
            errors = []
            if len({row["filename"] for row in labels}) != 1:
                errors.append("missing or ambiguous CSV filename join")
            if any(row["class"] not in NAMES for row in labels):
                errors.append("image includes unsupported species annotation")
            with Image.open(path) as image:
                size = image.size
                if image.getexif().get(274, 1) != 1:
                    errors.append("nontrivial EXIF orientation needs coordinate review")
                image.verify()
            boxes = []
            if not errors:
                try:
                    boxes = [box_record(row, size) for row in labels]
                except (ValueError, KeyError) as exc:
                    errors.append(str(exc))
            results.append({"id": member["id"], "group_id": member["group_id"], "split": split, "source_image": member["image_path"], "source_sha256": digest, "source_pixel_sha256": member["pixel_sha256"], "folder_class": member["class_id"], "width": size[0], "height": size[1], "boxes": boxes, "errors": errors})
    summary = {}
    for split in ("train", "development"):
        rows = [row for row in results if row["split"] == split]
        summary[split] = {"images": len(rows), "technically_valid_images": sum(not row["errors"] for row in rows), "box_counts": dict(Counter(box["class_id"] for row in rows for box in row["boxes"])), "errors": [{"id": row["id"], "errors": row["errors"]} for row in rows if row["errors"]]}
    output = {"source": SOURCE, "commit": COMMIT, "source_csv_sha256": hashlib.sha256(raw).hexdigest(), "protected_splits_sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest(), "status": "technical_box_audit_not_training_admission", "summary": summary, "records": results}
    (HERE / "plantdoc-detection-audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    fixture = {"width": "100", "height": "50", "xmin": "10", "ymin": "5", "xmax": "90", "ymax": "45", "class": "Tomato Early blight leaf"}
    assert box_record(fixture, (100, 50))["xywhn"] == [0.5, 0.5, 0.8, 0.8]
    main()
