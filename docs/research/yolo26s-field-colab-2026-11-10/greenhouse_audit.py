"""Inspect published VOC metadata; never infer disease labels or open test pixels."""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.parsers import expat

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE / "sichuan-v1-metadata"
API = "https://data.mendeley.com/public-api/datasets/c2x8rynybg"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def request(url, max_bytes=2_000_000):
    with requests.get(url, headers=HEADERS, timeout=45, stream=True) as response:
        response.raise_for_status()
        raw = bytearray()
        for chunk in response.iter_content(65536):
            if len(raw) + len(chunk) > max_bytes:
                raise ValueError("source response exceeds byte limit")
            raw.extend(chunk)
    return bytes(raw)


def reject_doctype(*_args):
    raise ValueError("DTD and entities are forbidden")


def parse_voc(raw):
    if len(raw) > 100_000:
        raise ValueError("oversized or unsupported XML")
    parser = expat.ParserCreate()
    parser.StartDoctypeDeclHandler = reject_doctype
    try:
        parser.Parse(raw, True)
    except expat.ExpatError as exc:
        raise ValueError("invalid XML") from exc
    tree = ET.fromstring(raw)
    width, height = (int(tree.findtext(f"size/{key}")) for key in ("width", "height"))
    if min(width, height) <= 0:
        raise ValueError("invalid image size")
    boxes = []
    for obj in tree.findall("object"):
        name = obj.findtext("name")
        coords = [float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = coords
        if not name or not all(math.isfinite(v) for v in coords):
            raise ValueError("missing name or non-finite coordinates")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("out-of-bounds box")
        boxes.append({"name": name, "xyxy": coords, "difficult": obj.findtext("difficult")})
    return {"filename": tree.findtext("filename"), "width": width, "height": height, "boxes": boxes}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    folders = json.loads(request(f"{API}/folders/1"))
    by_id = {row["id"]: row for row in folders}
    files = []
    for folder in folders:
        parent = by_id.get(folder.get("parent_id"), {}).get("name")
        if parent in {"annotations", "images"}:
            rows = json.loads(request(f"{API}/files?folder_id={folder['id']}&version=1"))
            for row in rows:
                row["release_path"] = f"{parent}/{folder['name']}/{row['filename']}"
            files.extend(rows)
    (ROOT / "files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    image_names = {Path(row["release_path"]).with_suffix("").as_posix().replace("images/", "", 1) for row in files if row["release_path"].startswith("images/")}
    annotations = [row for row in files if row["release_path"].startswith("annotations/")]

    def inspect(row):
        rel = Path(row["release_path"])
        if rel.is_absolute() or ".." in rel.parts or rel.name != row["filename"]:
            raise ValueError("unsafe release path")
        path = ROOT / rel
        details = row["content_details"]
        if path.exists():
            raw = path.read_bytes()
        else:
            if details["size"] > 100_000:
                raise ValueError("oversized annotation")
            raw = request(details["download_url"], max_bytes=100_000)
        if hashlib.sha256(raw).hexdigest() != details["sha256_hash"] or len(raw) != details["size"]:
            raise ValueError(f"publisher hash/size mismatch: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
        try:
            parsed = parse_voc(raw)
            error = None
        except (ValueError, TypeError, ET.ParseError) as exc:
            parsed, error = {}, str(exc)
        match = re.fullmatch(r"[A-Za-z]+_(\d{4})_\d+", rel.stem)
        return {"release_path": rel.as_posix(), "sha256": details["sha256_hash"], "date_token": match.group(1) if match else None, "image_exists": rel.with_suffix("").as_posix().replace("annotations/", "", 1) in image_names, "error": error, **parsed}

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(inspect, annotations))
    counts = Counter(box["name"] for row in results for box in row.get("boxes", []))
    leaf_rows = [row for row in results if any(box["name"].endswith("_Leaf") for box in row.get("boxes", []))]
    summary = {
        "dataset": "c2x8rynybg", "version": 1,
        "annotation_count": len(results), "image_count": len(image_names),
        "box_counts": dict(counts), "leaf_images": len(leaf_rows),
        "leaf_image_counts": dict(Counter(name for row in leaf_rows for name in {box["name"] for box in row["boxes"] if box["name"].endswith("_Leaf")})),
        "leaf_date_tokens": dict(Counter(row["date_token"] for row in leaf_rows)),
        "missing_pairs": [row["release_path"] for row in results if not row["image_exists"]],
        "errors": [{"path": row["release_path"], "error": row["error"]} for row in results if row["error"]],
        "status": "metadata_only_not_admitted_for_training",
        "grouping_limit": "Filename date token is not a plant ID; plant-level independence remains unverified.",
    }
    (ROOT / "annotations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    fixture = b"<annotation><filename>a.jpg</filename><size><width>10</width><height>8</height></size><object><name>Viral_Leaf</name><bndbox><xmin>0</xmin><ymin>1</ymin><xmax>10</xmax><ymax>8</ymax></bndbox></object></annotation>"
    assert parse_voc(fixture)["boxes"][0]["name"] == "Viral_Leaf"
    try:
        parse_voc(fixture.replace(b"<xmax>10", b"<xmax>11"))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-bounds annotation accepted")
    main()
