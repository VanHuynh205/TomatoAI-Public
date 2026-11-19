"""Verify the pinned Sichuan v1 gray-mold leaf evidence and train admission."""

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audit_source_samples import LIMIT, PROJECT, digest, extract_image
from audit_source_samples import OUTPUT as PILOT
from greenhouse_audit import ROOT as METADATA
from greenhouse_audit import parse_voc, request
from PIL import Image, ImageDraw, ImageOps

OUTPUT = PROJECT / "infra/derived/yolo26s-sichuan-leaf-subset-v1"
CLASS = "GrayMold_Leaf"
GROUP = "sichuan-c2x8rynybg-v1-single-greenhouse"
TRAIN_POLICY = {"group": GROUP, "role": "train_only", "final_holdout_eligible": False}
ADMITTED_IDS = {"sichuan-Gray_0617_013"}
DISEASE_ONLY_IDS = ADMITTED_IDS | {"sichuan-Gray_0617_011", "sichuan-Gray_0617_012"}
DISEASE_ONLY_VERSIONS = {2: DISEASE_ONLY_IDS, 3: DISEASE_ONLY_IDS | {"sichuan-Gray_0617_010"}}
SOURCE_FIELDS = (
    "filename", "source_image", "source_annotation", "image_url",
    "image_sha256", "image_bytes", "annotation_sha256", "width", "height", "exif_orientation",
)


def inspect(row, files, pilot_ids, output=OUTPUT):
    xml_path = row["release_path"]
    image_path = Path(xml_path.replace("annotations/", "images/", 1)).with_suffix(".jpg").as_posix()
    image_details = files[image_path]["content_details"]
    xml_details = files[xml_path]["content_details"]
    raw_xml = (METADATA / xml_path).read_bytes()
    if len(raw_xml) != xml_details["size"] or digest(raw_xml) != xml_details["sha256_hash"]:
        raise ValueError(f"XML publisher checksum or size mismatch: {xml_path}")
    annotation = parse_voc(raw_xml)
    if annotation != {key: row[key] for key in annotation} or annotation["filename"] != Path(image_path).name:
        raise ValueError(f"XML manifest mismatch: {xml_path}")

    sample_id = "sichuan-" + Path(image_path).stem
    path = output / "raw" / f"{sample_id}.jpg"
    pilot = PILOT / "raw" / path.name
    if path.is_file():
        raw_image = path.read_bytes()
    elif sample_id in pilot_ids and pilot.is_file():
        raw_image = pilot.read_bytes()
    else:
        raw_image = extract_image(request(image_details["download_url"], max_bytes=LIMIT), path.name.removeprefix("sichuan-"))
    if len(raw_image) != image_details["size"] or digest(raw_image) != image_details["sha256_hash"]:
        raise ValueError(f"JPG publisher checksum or size mismatch: {image_path}")
    with Image.open(io.BytesIO(raw_image)) as image:
        if image.format != "JPEG" or image.size != (annotation["width"], annotation["height"]):
            raise ValueError(f"unexpected JPG format or dimensions: {image_path}")
        orientation = image.getexif().get(274, 1)
        image.verify()
    with Image.open(io.BytesIO(raw_image)) as image:
        image.load()
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(raw_image)
    return {
        "id": sample_id,
        "group": GROUP,
        # Preserve the pilot marker in the pinned candidate evidence. It applies to holdout only.
        "eligibility": "prior_audit_quarantine" if sample_id in pilot_ids else "visual_review_required",
        "final_holdout_eligible": False,
        "source_image": image_path,
        "source_annotation": xml_path,
        "image_url": image_details["download_url"],
        "image_sha256": image_details["sha256_hash"],
        "image_bytes": image_details["size"],
        "annotation_sha256": xml_details["sha256_hash"],
        "exif_orientation": orientation,
        **annotation,
    }


def verify_admission(admission, manifest, *, disease_only=False, version=2):
    if version not in DISEASE_ONLY_VERSIONS:
        raise ValueError("Unknown Sichuan admission version")
    admitted_ids = DISEASE_ONLY_VERSIONS[version] if disease_only else ADMITTED_IDS
    if disease_only and admission.get("target_policy") != "disease_only_visible_leaf_extent":
        raise ValueError("Sichuan admission target policy changed")
    candidates = manifest["records"]
    if admission.get("source_policy") != TRAIN_POLICY or any(
        row["group"] != GROUP or row["final_holdout_eligible"] is not False
        for row in candidates
    ):
        raise ValueError("Sichuan source group must stay train-only")

    by_id = {row["id"]: row for row in candidates}
    records = admission["records"]
    if (
        len(by_id) != len(candidates)
        or len(records) != len(admitted_ids)
        or {row["id"] for row in records} != admitted_ids
        or any(row["id"] not in by_id for row in records)
        or any(admission.get(key) != manifest[key] for key in ("source", "coordinates"))
        or admission.get("reviewed_images") != len(candidates)
        or admission.get("reviewed_leaf_boxes")
        != sum(box["name"] == CLASS for row in candidates for box in row["boxes"])
    ):
        raise ValueError("Sichuan admission differs from the reviewed candidate set")

    for record in records:
        sample_id = record["id"]
        source = by_id[sample_id]
        leaf_boxes = [box for box in source["boxes"] if box["name"] == CLASS]
        expected = {
            **{key: source[key] for key in SOURCE_FIELDS},
            **TRAIN_POLICY,
            "local_image": f"raw/{sample_id}.jpg",
            "boxes": leaf_boxes,
        }
        if (
            not leaf_boxes
            or source["exif_orientation"] != 6
            or any(record.get(key) != value for key, value in expected.items())
            or record["final_holdout_eligible"] is not False
        ):
            raise ValueError(f"Sichuan admission differs from publisher data or policy: {sample_id}")

        height = source["height"]
        oriented_boxes = []
        for box in leaf_boxes:
            x1, y1, x2, y2 = box["xyxy"]
            oriented_boxes.append({**box, "xyxy": [height - y2, x1, height - y1, x2]})
        if record.get("exif_transposed") != {
            "width": height, "height": source["width"], "boxes": oriented_boxes,
        }:
            raise ValueError(f"Sichuan admission EXIF 6 coordinates mismatch: {sample_id}")


def tile(row, output=OUTPUT, leaf_class=CLASS):
    with Image.open(output / "raw" / f"{row['id']}.jpg") as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in row["boxes"]:
        color = "red" if box["name"] == leaf_class else "cyan"
        draw.rectangle(box["xyxy"], outline=color, width=12)
    image = ImageOps.contain(ImageOps.exif_transpose(image), (780, 530))
    canvas = Image.new("RGB", (780, 555), "white")
    canvas.paste(image, ((780 - image.width) // 2, 0))
    ImageDraw.Draw(canvas).text((8, 534), row["filename"] + "   red: leaf   cyan: other", fill="black", font_size=17)
    return canvas


def build_disease_only_admission(manifest, version=2):
    original = json.loads((OUTPUT / "admission.json").read_bytes())
    verify_admission(original, manifest)
    result = {**original, "target_policy": "disease_only_visible_leaf_extent", "records": []}
    for source in manifest["records"]:
        if source["id"] not in DISEASE_ONLY_VERSIONS[version]:
            continue
        if source["exif_orientation"] != 6:
            raise ValueError("Sichuan admission orientation changed")
        boxes = [box for box in source["boxes"] if box["name"] == CLASS]
        height = source["height"]
        oriented = [{**box, "xyxy": [height - box["xyxy"][3], box["xyxy"][0], height - box["xyxy"][1], box["xyxy"][2]]} for box in boxes]
        result["records"].append({"id": source["id"], **{key: source[key] for key in SOURCE_FIELDS}, **TRAIN_POLICY, "local_image": f"raw/{source['id']}.jpg", "boxes": boxes, "exif_transposed": {"width": height, "height": source["width"], "boxes": oriented}})
    verify_admission(result, manifest, disease_only=True, version=version)
    return result


def main(*, disease_only=False, version=2):
    inputs = {name: (METADATA / name).read_bytes() for name in ("annotations.json", "files.json")}
    files = {row["release_path"]: row for row in json.loads(inputs["files.json"])}
    rows = sorted(
        (row for row in json.loads(inputs["annotations.json"]) if any(box["name"] == CLASS for box in row["boxes"])),
        key=lambda row: row["filename"],
    )
    pilot_ids = {row["id"] for row in json.loads((PILOT / "selection.json").read_bytes())["records"]}
    assert len(rows) == 43 and sum(box["name"] == CLASS for row in rows for box in row["boxes"]) == 76
    assert len(pilot_ids) == 23
    (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        reviewed = list(pool.map(lambda row: inspect(row, files, pilot_ids), rows))
    manifest = {
        "source": "https://data.mendeley.com/datasets/c2x8rynybg/1",
        "inputs_sha256": {name: digest(raw) for name, raw in inputs.items()},
        "coordinates": "publisher Pascal VOC xyxy in raw pixel coordinates",
        "records": reviewed,
    }
    admission = json.loads((OUTPUT / "admission.json").read_bytes())
    verify_admission(admission, manifest)
    path = OUTPUT / "candidates.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("candidate manifest changed; use a new evidence version")
    else:
        path.write_text(content, encoding="utf-8")
    for start in range(0, len(reviewed), 4):
        overlay = OUTPUT / f"overlay-{start // 4 + 1:02d}.jpg"
        if overlay.exists():
            continue
        page = Image.new("RGB", (1560, 1110), "white")
        for offset, row in enumerate(reviewed[start:start + 4]):
            page.paste(tile(row), (offset % 2 * 780, offset // 2 * 555))
        page.save(overlay, quality=94)
    print("SICHUAN_LEAF_CANDIDATES_VERIFIED", len(reviewed), sum(row["image_bytes"] for row in reviewed))
    print("SICHUAN_LEAF_ADMISSION_VERIFIED", len(admission["records"]), GROUP)
    if disease_only:
        if digest((OUTPUT / "candidates.json").read_bytes()) != "a00ccd3d22904314059a072d305b9108682fe27717cd167b12cdeab5c70a2edd":
            raise ValueError("Pinned candidate evidence changed")
        revised = build_disease_only_admission(manifest, version=version)
        encoded = (json.dumps(revised, indent=2) + "\n").encode()
        target = OUTPUT / f"admission-disease-only-v{version}.json"
        if target.exists() and target.read_bytes() != encoded:
            raise ValueError("Admission changed; create a new evidence version")
        if not target.exists():
            target.write_bytes(encoded)
        print("SICHUAN_DISEASE_ONLY_ADMISSION_VERIFIED", len(revised["records"]), digest(encoded))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--disease-only-v2", action="store_true", help="Verify the three-image reviewed admission")
    group.add_argument("--disease-only-v3", action="store_true", help="Verify the four-image reviewed admission")
    args = parser.parse_args()
    main(disease_only=args.disease_only_v2 or args.disease_only_v3, version=3 if args.disease_only_v3 else 2)
