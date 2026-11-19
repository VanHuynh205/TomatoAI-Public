"""Fetch deterministic source-audit samples; reserve every selected image from final tests."""

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUTPUT = PROJECT / "infra/derived/yolo26s-field-source-audit-v1"
LIMIT = 25_000_000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def select_samples():
    coco_path = HERE / "fieldplant-metadata/train_annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    if digest(coco_path.read_bytes()) != "b44d734b1b90b498d327f25702bd527b9e4fd61607da4d72c8572d2137c87d79":
        raise ValueError("FieldPlant annotation version changed")
    categories = {row["id"]: row["name"] for row in coco["categories"]}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        annotations[row["image_id"]].append(row)
    samples = {}
    for category_id, name in categories.items():
        if not name.startswith("Tomato"):
            continue
        candidates = [row for row in coco["images"] if any(box["category_id"] == category_id for box in annotations[row["id"]])]
        for row in sorted(candidates, key=lambda item: digest(item["file_name"].encode()))[:3]:
            boxes = []
            for box in annotations[row["id"]]:
                x, y, w, h = box["bbox"]
                boxes.append({"name": categories[box["category_id"]], "xyxy": [x, y, x + w, y + h]})
            sample_id = f"fieldplant-{row['id']}"
            samples[sample_id] = {
                "id": sample_id, "source": "fieldplant-v11-kaggle-v1", "filename": row["file_name"],
                "group_hint": row["file_name"].split(".rf.")[0],
                "url": "https://www.kaggle.com/api/v1/datasets/download/prathmeshnikam821/fieldplant/" + quote("train/" + row["file_name"], safe="") + "?datasetVersionNumber=1",
                "width": row["width"], "height": row["height"], "boxes": boxes,
            }
    annotations_path = HERE / "sichuan-v1-metadata/annotations.json"
    rows = json.loads(annotations_path.read_text(encoding="utf-8"))
    files = json.loads((HERE / "sichuan-v1-metadata/files.json").read_text(encoding="utf-8"))
    by_path = {row["release_path"]: row for row in files}
    for name in ("GrayMold_Leaf", "Viral_Leaf", "Wilt_Leaf"):
        candidates = [row for row in rows if any(box["name"] == name for box in row["boxes"])]
        dates = sorted({row["date_token"] for row in candidates})
        for date in dates:
            row = min((row for row in candidates if row["date_token"] == date), key=lambda item: digest(item["release_path"].encode()))
            image_path = row["release_path"].replace("annotations/", "images/", 1).removesuffix(".xml") + ".jpg"
            details = by_path[image_path]["content_details"]
            sample_id = "sichuan-" + Path(image_path).stem
            samples[sample_id] = {
                "id": sample_id, "source": "sichuan-c2x8rynybg-v1", "filename": row["filename"],
                "group_hint": "date-" + date, "url": details["download_url"],
                "expected_sha256": details["sha256_hash"], "expected_bytes": details["size"],
                "width": row["width"], "height": row["height"], "boxes": row["boxes"],
            }
    return {
        "purpose": "annotation review only; selected images and related groups must not enter final holdout",
        "training_admitted": False,
        "inputs_sha256": {"fieldplant_coco": digest(coco_path.read_bytes()), "sichuan_annotations": digest(annotations_path.read_bytes())},
        "records": [samples[key] for key in sorted(samples)],
    }


def extract_image(raw, filename):
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        return raw
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) != 1 or Path(entries[0].filename).name != filename or entries[0].file_size > LIMIT:
            raise ValueError("unexpected single-image archive")
        return archive.read(entries[0])


def fetch(row):
    path = OUTPUT / "raw" / f"{row['id']}.jpg"
    ledger = OUTPUT / "download-results.json"
    previous = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
    by_id = {item["id"]: item for item in previous}
    if len(by_id) != len(previous):
        raise ValueError("duplicate audit evidence IDs")
    if path.exists():
        raw = path.read_bytes()
    else:
        with requests.get(row["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=45, stream=True) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > LIMIT:
                    raise ValueError("source image exceeds byte limit")
        raw = extract_image(bytes(data), row["filename"])
    if row["id"] in by_id and digest(raw) != by_id[row["id"]]["sha256"]:
        raise ValueError("previously audited image checksum changed")
    if row.get("expected_sha256") and digest(raw) != row["expected_sha256"]:
        raise ValueError("publisher image checksum mismatch")
    if row.get("expected_bytes") and len(raw) != row["expected_bytes"]:
        raise ValueError("publisher image size mismatch")
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (row["width"], row["height"]):
            raise ValueError("image/annotation dimensions differ")
        orientation = image.getexif().get(274, 1)
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
    result = {"id": row["id"], "path": path.relative_to(PROJECT).as_posix(), "sha256": digest(raw), "bytes": len(raw), "exif_orientation": orientation}
    if row["id"] in by_id and result != by_id[row["id"]]:
        raise ValueError("previously audited image metadata changed")
    if row["id"] not in by_id:
        # Record first-seen bytes before caching so interruption cannot unseal them.
        ledger.write_text(json.dumps([*previous, result], indent=2) + "\n", encoding="utf-8")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return result


def tile(row, result):
    with Image.open(PROJECT / result["path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("red", "cyan", "yellow", "magenta")
    for index, box in enumerate(row["boxes"]):
        draw.rectangle(box["xyxy"], outline=colors[index % len(colors)], width=max(3, image.width // 250))
    image = ImageOps.contain(image, (600, 440))
    canvas = Image.new("RGB", (600, 500), "white")
    canvas.paste(image, ((600 - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 444), row["id"], fill="black")
    draw.text((5, 462), ", ".join(sorted({box["name"] for box in row["boxes"]})), fill="black")
    draw.text((5, 480), f"{len(row['boxes'])} boxes; EXIF={result['exif_orientation']}; source audit, never test", fill="black")
    return canvas


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = select_samples()
    manifest = OUTPUT / "selection.json"
    content = json.dumps(selection, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError("audit selection changed; use a new version")
    if not manifest.exists():
        manifest.write_text(content, encoding="utf-8")
    results, tiles = [], []
    for row in selection["records"]:
        result = fetch(row)
        results.append(result)
        tiles.append(tile(row, result))
        print(row["id"], result["bytes"], flush=True)
    for start in range(0, len(tiles), 6):
        page = Image.new("RGB", (1200, 1500), "white")
        for offset, image in enumerate(tiles[start:start + 6]):
            page.paste(image, ((offset % 2) * 600, (offset // 2) * 500))
        page.save(OUTPUT / f"overlays-{start // 6 + 1:02d}.jpg", quality=92)
    print("SOURCE_SAMPLE_AUDIT_READY", len(results))


if __name__ == "__main__":
    assert extract_image(b"not-a-zip", "sample.jpg") == b"not-a-zip"
    main()
