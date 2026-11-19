"""Prepare source-box review of Sichuan scenes annotated only with wilt leaves."""

import json
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from sichuan_leaf_subset import GROUP, METADATA, PILOT, PROJECT, TRAIN_POLICY, digest, inspect, tile

OUTPUT = PROJECT / "infra/derived/yolo26s-sichuan-wilt-review-v1"


def select_records(records):
    selected = [row for row in records if row["boxes"] and all(box["name"] == "Wilt_Leaf" for box in row["boxes"])]
    if any(row.get("error") or not row["image_exists"] for row in selected):
        raise ValueError("Invalid source annotation or missing image")
    if len({row["filename"] for row in selected}) != len(selected):
        raise ValueError("Duplicate source filename")
    return sorted(selected, key=lambda row: row["filename"])


def write_evidence(path, value):
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Evidence changed; use a new version")
    if not path.exists():
        path.write_bytes(encoded)


def main():
    inputs = {name: (METADATA / name).read_bytes() for name in ("annotations.json", "files.json")}
    rows = select_records(json.loads(inputs["annotations.json"]))
    if len(rows) != 44:
        raise ValueError("Unexpected pinned source selection")
    files = {row["release_path"]: row for row in json.loads(inputs["files.json"])}
    pilot_ids = {row["id"] for row in json.loads((PILOT / "selection.json").read_bytes())["records"]}
    (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
    selection = {"status": "visual_review_required_not_training_admission", "source": "https://data.mendeley.com/datasets/c2x8rynybg/1", "source_policy": TRAIN_POLICY, "inputs_sha256": {name: digest(raw) for name, raw in inputs.items()}, "selection_rule": "All source annotations in the scene are Wilt_Leaf; excludes mixed-organ scenes for this initial review only.", "ids": ["sichuan-" + row["filename"].removesuffix(".jpg") for row in rows]}
    write_evidence(OUTPUT / "selection.json", selection)
    with ThreadPoolExecutor(max_workers=3) as pool:
        reviewed = list(pool.map(lambda row: inspect(row, files, pilot_ids, output=OUTPUT), rows))
    write_evidence(OUTPUT / "candidates.json", {**selection, "records": reviewed})
    for start in range(0, len(reviewed), 4):
        path = OUTPUT / f"overlay-{start // 4 + 1:02d}.jpg"
        if not path.exists():
            page = Image.new("RGB", (1560, 1110), "white")
            for index, row in enumerate(reviewed[start:start + 4]):
                page.paste(tile(row, output=OUTPUT, leaf_class="Wilt_Leaf"), (index % 2 * 780, index // 2 * 555))
            page.save(path, quality=94)
    print(json.dumps({"images_verified": len(reviewed), "boxes": sum(len(row["boxes"]) for row in reviewed), "bytes": sum(row["image_bytes"] for row in reviewed), "group": GROUP, "admitted": 0, "candidate_manifest_sha256": digest((OUTPUT / "candidates.json").read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
