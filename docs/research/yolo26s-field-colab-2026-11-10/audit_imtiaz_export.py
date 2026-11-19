"""Verify the official Imtiaz label IDs against Mendeley without displaying images."""

import ast
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "infra/derived/yolo26s-imtiaz-official-export-v1"
ARCHIVE_SHA = "f59de1fce39affe021712b795b6ae24fa19d15ec0b09fa902070a7c9d01d2f52"
NAMES = ["Bacterial Spot", "Early_Blight", "Healthy", "Late_blight", "Leaf Mold", "Target_Spot", "black spot"]


def parse_boxes(raw):
    boxes = []
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {str(i) for i in range(len(NAMES))}:
            raise ValueError("Invalid source annotation row")
        x, y, w, h = map(float, fields[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("Invalid source box extent")
        if min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
            raise ValueError("Source box outside image")
        boxes.append({"source_id": int(fields[0]), "source_label": NAMES[int(fields[0])], "xywh_normalized": [x, y, w, h]})
    return boxes


def main():
    archive_path = OUTPUT / "raw/export.yolov8.zip"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == ARCHIVE_SHA
    metadata_path = OUTPUT / "mendeley-annotated-metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    assert metadata["source"] == "https://data.mendeley.com/datasets/bpfd9cns5g/2"
    expected = {}
    for folder in metadata["folders"]:
        relative = PurePosixPath(folder["source_path"]).relative_to("TomatoLeafMulticlass (Annotated)")
        for row in folder["files"]:
            name = str(relative / row["filename"])
            assert name not in expected, "Duplicate source metadata"
            expected[name] = row
    assert len(expected) == 1378
    records, evidence = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        assert len(members) == len(set(members))
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and "\\" not in n for n in members)
        config = archive.read("data.yaml")
        # This pinned release uses a quoted Python-compatible literal for names, not a general YAML parser.
        names_line, = [line for line in config.decode().splitlines() if line.startswith("names: ")]
        assert ast.literal_eval(names_line.partition(": ")[2]) == NAMES
        assert "nc: 7" in config.decode().splitlines()
        actual = {n for n in members if n.endswith((".jpg", ".txt")) and "/" in n}
        assert actual == expected.keys(), "ZIP/Mendeley member coverage differs"
        for name in sorted(actual):
            payload = archive.read(name)
            assert len(payload) == expected[name]["size"], name
            assert hashlib.sha256(payload).hexdigest() == expected[name]["sha256"], name
        for name in sorted(n for n in actual if "/images/" in n):
            label = str(PurePosixPath(name.replace("/images/", "/labels/")).with_suffix(".txt"))
            assert label in actual
            with Image.open(io.BytesIO(archive.read(name))) as image:
                size = image.size
                assert size == (640, 640) and image.getexif().get(274, 1) == 1
            records.append({"member": name, "sha256": expected[name]["sha256"], "mendeley_image_id": expected[name]["id"],
                            "label_member": label, "label_sha256": expected[label]["sha256"], "mendeley_label_id": expected[label]["id"],
                            "split": name.split("/")[0], "size": size, "source_filename_key": PurePosixPath(name).name.split(".rf.")[0],
                            "annotations": parse_boxes(archive.read(label))})
        for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
            raw = archive.read(name)
            destination = OUTPUT / name
            if destination.exists():
                assert destination.read_bytes() == raw, "Pinned export text changed"
            else:
                destination.write_bytes(raw)
            evidence[name] = hashlib.sha256(raw).hexdigest()
    assert Counter(r["split"] for r in records) == {"train": 597, "valid": 61, "test": 31}
    groups, hashes = defaultdict(list), defaultdict(list)
    for row in records:
        groups[row["source_filename_key"]].append(row)
        hashes[row["sha256"]].append(row)
    cross = lambda table: {key: [r["member"] for r in rows] for key, rows in table.items() if len({r["split"] for r in rows}) > 1}
    summary = {"images": len(records), "mendeley_byte_verified_members": len(expected),
               "boxes_by_source_label": dict(Counter(a["source_label"] for r in records for a in r["annotations"])),
               "empty_labels_by_split": dict(Counter(r["split"] for r in records if not r["annotations"])),
               "source_filename_groups": len(groups), "cross_split_filename_groups": len(cross(groups)), "cross_split_image_hashes": len(cross(hashes))}
    result = {"status": "label_id_correspondence_verified_not_training_admission", "archive_sha256": ARCHIVE_SHA,
              "source_page": "https://universe.roboflow.com/ahmed-imtiaz-htzgn/tomato-leaf-multi-disease/dataset/1",
              "license": "CC BY 4.0", "mendeley_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "export_text_sha256": evidence,
              "source_names": NAMES, "summary": summary, "cross_split_filename_groups": cross(groups), "cross_split_image_hashes": cross(hashes),
              "training_admitted": False, "final_holdout_eligible": False, "new_visual_exposure": 0,
              "limitations": ["Source has existing rotations and stretched images; filename groups are not plant/session certificates.",
                              "Byte correspondence verifies source class IDs, not disease diagnosis, whole-leaf boxes or complete annotations.",
                              "Empty labels are unresolved, not approved healthy negatives. No cross-source similarity screen completed."], "records": records}
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    path = OUTPUT / "metadata-audit.json"
    if path.exists():
        assert path.read_bytes() == encoded, "Evidence changed; use a new version"
    else:
        path.write_bytes(encoded)
    print(json.dumps({**summary, "audit_sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
