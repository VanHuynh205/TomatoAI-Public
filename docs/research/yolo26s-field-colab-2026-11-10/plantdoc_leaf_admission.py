"""Prepare a guarded visual review and materialize its PlantDoc admission decisions."""

import argparse
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROGRESS = ROOT / "docs/progress" / HERE.name
OUTPUT = ROOT / "infra/derived/yolo26s-plantdoc-leaf-admission-v1"
DISEASE_OUTPUT = ROOT / "infra/derived/yolo26s-plantdoc-disease-only-v2"
V1_MANIFEST_SHA256 = "21af798493261a0043c9b5ea034a45730a490fb0909fce59cb60f21e39d1c868"
V1_SCRIPT_SHA256 = "d9ca24306ccf98e01dafb648abc8503e14f4ec4634018cea3a663e8630b32584"
PUBLISHER_BLOBS_SHA256 = (
    "6d097a51351b315e5a9a8d124463b60a988a9448a67e6276452763d11a0ce5e5"
)
sys.path[:0] = [str(PROGRESS), str(ROOT)]

from audit_plantdoc_detection import (
    COMMIT,
    CSV_SHA256,
    NAMES,
    PROTECTED_SHA256,
    SOURCE,
    box_record,
    key,
    validate_member,
)
from protect_splits import sha256, table

from ai.data import _pixel_hash_stream

REVIEW = ROOT / "infra/derived/plantdoc-folder-review-v2"
MOSAIC = "Tomato leaf mosaic virus"
MOSAIC_DETAILS = {"PD0433", "PD0438", "PD0440", "PD0442", "PD0444", "PD0449"}
TREE = f"https://api.github.com/repos/pratikkayal/PlantDoc-Object-Detection-Dataset/git/trees/{COMMIT}?recursive=1"
INPUTS = {
    PROGRESS / "protected-splits.json": PROTECTED_SHA256,
    PROGRESS / "plantdoc-source-train-labels.csv": CSV_SHA256,
    PROGRESS
    / "plantdoc-detection-audit.json": "ee8905aa8b76ee362450c04f2c866a4614a5ee93f77ab278af4da794d1bbd5bd",
    REVIEW
    / "manifest.csv": "58a566aa39f1d99dae65fab7d9594f352e6e1a92668582969878eaf83cd7d389",
    REVIEW
    / "README.md": "74f61f0b37941951dd4520071292c52e8fc6be368d5946d1bb8f232d12b42893",
}
VISUAL_REASONS = {
    "single_leaf_units_visible": "Source boxes each bound one whole leaf or one visible leaflet blade. Foreground instances are covered at the review scale.",
    "unboxed_foreground_leaf": "A distinct, discernible foreground tomato leaf or leaflet has no corresponding source box.",
    "non_leaf_or_multi_leaf_box": "A source box marks a lesion, several distinct blades, a cluster, or another unit outside the leaf contract.",
    "unclear_leaf_geometry": "Blur, truncation, overlap, size, or poor box fit prevents verifying one complete visible leaf unit per source box.",
    "panel_or_composite": "The image is a montage, diagram, presentation slide, or composite whose separate panels do not represent one scene.",
    "third_party_stock_mark": "A visible stock-photo watermark requires separate rights review before dataset use.",
}
DISEASE_REVIEW_V1_REASONS = {"unboxed_foreground_leaf", "unclear_leaf_geometry"}
DISEASE_REASONS = {
    "visible_target_leaves_covered": "Publisher target boxes enclose the visible extent of individual leaves. Discernible unboxed foreground leaves do not appear symptomatic at review scale.",
    "publisher_healthy_negative": "Publisher and folder review agree on healthy; the image has no discernible suspicious leaf at review scale.",
    "unboxed_potential_target": "A discernible unboxed leaf has visible symptoms or cannot be separated from a potentially symptomatic foreground instance.",
    "target_box_geometry_unverified": "A box cuts off visible leaf tissue, spans multiple units, or its leaf extent cannot be verified.",
    "third_party_stock_mark": VISUAL_REASONS["third_party_stock_mark"],
}
DISEASE_ACCEPTED = {"visible_target_leaves_covered", "publisher_healthy_negative"}


def stable_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise ValueError(
            f"Evidence changed: {path.name}. Create a new admission version."
        )
    if not path.exists():
        path.write_bytes(data)


def load_inputs():
    for path, expected in INPUTS.items():
        if sha256(path) != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
    protected = json.loads((PROGRESS / "protected-splits.json").read_bytes())
    audit = json.loads((PROGRESS / "plantdoc-detection-audit.json").read_bytes())
    source_rows = defaultdict(list)
    for row in table(PROGRESS / "plantdoc-source-train-labels.csv"):
        source_rows[key(row["filename"])].append(row)
    folders = table(REVIEW / "manifest.csv")
    base = {row["id"]: row for row in audit["records"]}
    test_ids = set(protected["plantdoc_test_protected_ids"])
    all_protected_ids = set(protected["plantdoc_train_protected_ids"])
    protected_paths = {
        (ROOT / row["image_path"]).resolve()
        for row in protected["plantdoc_protected_records"]
    }
    test_rows = [row for row in folders if row["id"] in test_ids]
    test_paths = {(ROOT / row["image_path"]).resolve() for row in test_rows}
    test_rows += [
        row
        for row in protected["plantvillage_protected_records"]
        if row["split"] == "test"
    ]
    test_bytes = {row["sha256"] for row in test_rows}
    test_pixels = {row["pixel_sha256"] for row in test_rows}
    mosaic = [
        row
        for row in folders
        if row["source_split"] == "train" and row["source_label"] == MOSAIC
    ]
    allowed_mosaic = [
        row for row in mosaic if row["reason"] == row["v1_reason"] == "outside_pv8"
    ]
    assert len(allowed_mosaic) == 40 and len(mosaic) == 44
    admitted_legacy = {
        row["id"]: (split, row)
        for split in ("train", "development")
        for row in protected[f"plantdoc_allowed_{split}"]
    }
    assert base.keys() == admitted_legacy.keys() and len(base) == 342
    paths = {}
    for sample_id, (split, row) in {
        **admitted_legacy,
        **{row["id"]: ("development", row) for row in allowed_mosaic},
    }.items():
        path = validate_member(row, split, protected).resolve()
        if (
            path in test_paths
            or row["sha256"] in test_bytes
            or row["pixel_sha256"] in test_pixels
        ):
            raise ValueError(f"Protected test path or hash: {sample_id}")
        if (split == "train" or sample_id not in base) and (
            sample_id in all_protected_ids
            or path in protected_paths
            or row["sha256"] in protected["protected_byte_sha256"]
            or row["pixel_sha256"] in protected["protected_pixel_sha256"]
        ):
            raise ValueError(f"Mosaic expansion reached protected data: {sample_id}")
        paths[sample_id] = path
    return (
        base,
        admitted_legacy,
        mosaic,
        source_rows,
        paths,
        test_bytes,
        test_pixels,
    )


def load_blobs(source_rows, paths, *, sealed=False):
    path = OUTPUT / "publisher-train-blobs.json"
    if sealed and sha256(path) != PUBLISHER_BLOBS_SHA256:
        raise ValueError("Sealed publisher blob checksum mismatch")
    if path.exists():
        return json.loads(path.read_bytes())["train_files"]
    response = requests.get(TREE, timeout=30)
    response.raise_for_status()
    tree = response.json()
    if tree["truncated"] or tree["sha"] != COMMIT:
        raise ValueError("Unexpected publisher Git tree")
    filenames = {
        f"TRAIN/{row['filename']}"
        for path in paths.values()
        for row in source_rows[key(path.name)]
    }
    train = {
        row["path"].removeprefix("TRAIN/"): {
            "blob_sha1": row["sha"],
            "size": row["size"],
        }
        for row in tree["tree"]
        if row["type"] == "blob" and row["path"] in filenames
    }
    stable_json(
        path,
        {
            "url": TREE,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "train_files": train,
        },
    )
    return train


def inspect(sample_id, member, legacy, path, labels, blobs, test_bytes, test_pixels):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != member["sha256"] or digest in test_bytes:
        raise ValueError(f"Unexpected or test-protected bytes: {sample_id}")
    pixel, error = _pixel_hash_stream(io.BytesIO(raw))
    if error or pixel != member["pixel_sha256"] or pixel in test_pixels:
        raise ValueError(f"Unexpected or test-protected pixels: {sample_id}")
    filenames = {row["filename"] for row in labels}
    blob = None
    reasons = []
    if len(filenames) != 1:
        reasons.append("missing_or_ambiguous_publisher_filename")
    else:
        blob = blobs.get(next(iter(filenames)))
        header = f"blob {len(raw)}\0".encode("ascii")
        if (
            blob is None
            or blob["size"] != len(raw)
            or blob["blob_sha1"] != hashlib.sha1(header + raw).hexdigest()
        ):
            reasons.append("local_image_differs_from_publisher_detection_blob")
    with Image.open(io.BytesIO(raw)) as image:
        size = image.size
        orientation = image.getexif().get(274, 1)
    if orientation != 1:
        reasons.append("nontrivial_exif_orientation")
    if any(row["class"] not in NAMES for row in labels):
        reasons.append("unsupported_source_box_class")
    boxes = []
    if not reasons:
        try:
            boxes = [box_record(row, size) for row in labels]
        except (ValueError, KeyError) as exc:
            reasons.append(f"source_box_invalid: {exc}")
    if legacy and not legacy["errors"] and boxes != legacy["boxes"] and not reasons:
        raise ValueError(f"Original box audit changed: {sample_id}")
    if boxes and any(row["class"] != member["source_label"] for row in labels):
        reasons.append("mixed_box_labels_outside_approved_image_mapping")
    for box in boxes:
        if box["source_class"] == "Tomato leaf" and member["class_id"] == "healthy":
            box["class_id"] = "healthy"
    return {
        "width": size[0],
        "height": size[1],
        "exif_orientation": orientation,
        "publisher_blob": blob,
        "publisher_filenames": sorted(filenames),
        "source_box_count": len(labels),
        "boxes": boxes,
        "technical_exclusions": reasons,
    }


def tile(path, row, width=324, height=264):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
        raise ValueError(f"Source changed after verification: {row['id']}")
    with Image.open(io.BytesIO(raw)) as source:
        original = ImageOps.contain(source.convert("RGB"), (width - 4, height - 26))
    draw = ImageDraw.Draw(original)
    sx, sy = original.width / row["width"], original.height / row["height"]
    for box in row["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#ff1856", width=2)
    canvas = Image.new("RGB", (width, height), "#fafafa")
    canvas.paste(
        original, ((width - original.width) // 2, (height - 26 - original.height) // 2)
    )
    ImageDraw.Draw(canvas).text(
        (5, height - 20),
        f"{row['id']} | {row['split']} | {len(row['boxes'])} boxes",
        fill="black",
        font_size=14,
    )
    return canvas


def render(rows, paths):
    for start in range(0, len(rows), 20):
        page = Image.new("RGB", (1620, 1056), "white")
        for i, row in enumerate(rows[start : start + 20]):
            page.paste(tile(paths[row["id"]], row), (i % 5 * 324, i // 5 * 264))
        page.save(OUTPUT / f"overlay-{start // 20 + 1:02d}.jpg", quality=92)
    details = Image.new("RGB", (1600, 1530), "white")
    for i, row in enumerate(row for row in rows if row["id"] in MOSAIC_DETAILS):
        details.paste(
            tile(paths[row["id"]], row, 800, 510), (i % 2 * 800, i // 2 * 510)
        )
    details.save(OUTPUT / "mosaic-details.jpg", quality=94)


def disease_review_candidates(records, v1):
    return [
        row
        for row in records
        if v1[row["id"]]["visual_review"] in DISEASE_REVIEW_V1_REASONS
    ]


def disease_only_labels(row, reason, source_labels):
    boxes = row.get("boxes", [])
    if len(boxes) != len(source_labels) or len(boxes) != row["source_box_count"]:
        raise ValueError(f"Publisher box count mismatch: {row['id']}")
    target_boxes, dropped_ids = [], []
    for box, (source_id, source_class) in zip(boxes, source_labels, strict=True):
        if source_class != box["source_class"] or box["class_id"] != (
            "healthy" if source_class == "Tomato leaf" else NAMES[source_class]
        ):
            raise ValueError(f"Publisher class mismatch: {row['id']}")
        if box["class_id"] == "healthy":
            if row["source_label"] != "Tomato leaf" or row["source_split"] != "train":
                raise ValueError(f"Unverified healthy label: {row['id']}")
            dropped_ids.append(source_id)
        else:
            target_boxes.append({**box, "source_box_id": source_id})
    if not boxes or (reason == "publisher_healthy_negative") != (not target_boxes):
        raise ValueError(
            f"Disease-only decision conflicts with publisher labels: {row['id']}"
        )
    return target_boxes, dropped_ids


def disease_summary(accepted):
    summary = {}
    for split in ("train", "development"):
        rows = [row for row in accepted if row["split"] == split]
        classes = {
            class_id: {"images": 0, "groups": set(), "boxes": 0}
            for source_class, class_id in NAMES.items()
            if source_class != "Tomato leaf"
        }
        for row in rows:
            for class_id, count in Counter(
                box["class_id"] for box in row["boxes"]
            ).items():
                stats = classes[class_id]
                stats["images"] += 1
                stats["groups"].add(row["group_id"])
                stats["boxes"] += count
        summary[split] = {
            "images": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "disease_negative_images": sum(not row["boxes"] for row in rows),
            "disease_negative_groups": len(
                {row["group_id"] for row in rows if not row["boxes"]}
            ),
            "classes": {
                name: {**stats, "groups": len(stats["groups"])}
                for name, stats in sorted(classes.items())
            },
        }
    return summary


def disease_only(step, records, paths):
    v1_path = OUTPUT / "manifest.json"
    if sha256(v1_path) != V1_MANIFEST_SHA256:
        raise ValueError("Sealed PlantDoc v1 manifest checksum mismatch")
    v1_manifest = json.loads(v1_path.read_bytes())
    v1 = {row["id"]: row for row in v1_manifest["records"]}
    if v1.keys() != {row["id"] for row in records}:
        raise ValueError("PlantDoc v1 membership changed")
    for row in records:
        previous = v1[row["id"]]
        for field in (
            "split",
            "group_id",
            "source_split",
            "source_label",
            "source_image",
            "source_sha256",
            "source_pixel_sha256",
            "technical_exclusions",
            "boxes",
        ):
            if row.get(field) != previous.get(field):
                raise ValueError(f"Sealed v1 evidence changed: {row['id']} {field}")
    candidates = disease_review_candidates(records, v1)
    DISEASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(candidates):
        row["disease_overlay"] = f"review-{i // 12 + 1:02d}.jpg"
    if step == "disease-only-render":
        for start in range(0, len(candidates), 12):
            page = Image.new("RGB", (1536, 1136), "white")
            for i, row in enumerate(candidates[start : start + 12]):
                page.paste(
                    tile(paths[row["id"]], row, 384, 378), (i % 4 * 384, i // 4 * 378)
                )
            page.save(DISEASE_OUTPUT / candidates[start]["disease_overlay"], quality=93)
        print("PLANTDOC_DISEASE_REVIEW_CANDIDATES", len(candidates))
        return

    decisions_path = DISEASE_OUTPUT / "visual-decisions.csv"
    reviewed = table(decisions_path)
    decisions = {row["id"]: row["reason"] for row in reviewed}
    if len(decisions) != len(reviewed) or decisions.keys() != {
        row["id"] for row in candidates
    }:
        raise ValueError(
            "Expected one disease-only visual decision per eligible v1 exclusion"
        )
    if set(decisions.values()) - DISEASE_REASONS.keys():
        raise ValueError("Unknown disease-only visual reason")
    source_annotations = defaultdict(list)
    for source_id, label in enumerate(
        table(PROGRESS / "plantdoc-source-train-labels.csv"), start=1
    ):
        source_annotations[key(label["filename"])].append((source_id, label["class"]))
    accepted = []
    accepted_source_ids = set()
    for row in records:
        previous = v1[row["id"]]
        reason = decisions.get(row["id"])
        row["v1_visual_review"] = previous["visual_review"]
        row["disease_visual_review"] = reason
        if previous["decision"] == "accepted":
            reason = (
                "publisher_healthy_negative"
                if row["source_label"] == "Tomato leaf"
                else "visible_target_leaves_covered"
            )
        row["decision"] = "excluded"
        if reason in DISEASE_ACCEPTED:
            if row["technical_exclusions"]:
                raise ValueError(f"Technical exclusion reached admission: {row['id']}")
            source_labels = source_annotations[key(paths[row["id"]].name)]
            target_boxes, dropped_ids = disease_only_labels(row, reason, source_labels)
            source_ids = {source_id for source_id, _ in source_labels}
            if source_ids & accepted_source_ids:
                raise ValueError(f"Duplicate admitted publisher row: {row['id']}")
            accepted_source_ids.update(source_ids)
            row["boxes"] = target_boxes
            row["dropped_healthy_source_box_ids"] = dropped_ids
            row["decision"] = "accepted"
            row["admission_reason"] = reason
            accepted.append(row)
        else:
            row.pop("boxes", None)

    for field in ("group_id", "source_sha256", "source_pixel_sha256"):
        train = {row[field] for row in accepted if row["split"] == "train"}
        development = {row[field] for row in accepted if row["split"] == "development"}
        if train & development:
            raise ValueError(f"Disease-only accepted split overlap: {field}")
    if any(
        row["source_label"] == MOSAIC and row["split"] != "development"
        for row in accepted
    ):
        raise ValueError("Mosaic reached disease-only training")
    summary = disease_summary(accepted)
    inputs = [
        *INPUTS,
        v1_path,
        OUTPUT / "publisher-train-blobs.json",
        decisions_path,
        Path(__file__),
    ]
    stable_json(
        DISEASE_OUTPUT / "manifest.json",
        {
            "schema_version": 2,
            "source": SOURCE,
            "source_commit": COMMIT,
            "status": "disease_only_admitted_for_detector_preparation",
            "training_run_verified": False,
            "expert_verified": False,
            "annotation_unit": "one leaf or leaflet, including partial leaves whose entire visible extent is enclosed",
            "source_box_id": "1-based data-row ordinal in the pinned publisher CSV, excluding its header",
            "coordinates": v1_manifest["coordinates"],
            "inputs_sha256": {
                path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs
            },
            "overlays_sha256": {
                name: sha256(DISEASE_OUTPUT / name)
                for name in sorted({row["disease_overlay"] for row in candidates})
            },
            "visual_review_reasons": DISEASE_REASONS,
            "source_to_target_class": {
                source_class: class_id
                for source_class, class_id in NAMES.items()
                if source_class != "Tomato leaf"
            },
            "summary": summary,
            "rereview_counts": dict(Counter(decisions.values())),
            "records": records,
        },
    )
    print("PLANTDOC_DISEASE_ONLY_ADMISSION_VERIFIED", json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", choices=("render", "admit", "disease-only-render", "disease-only-admit")
    )
    args = parser.parse_args()
    disease_mode = args.step.startswith("disease-only-")
    (
        base,
        legacy,
        mosaic,
        source_rows,
        paths,
        test_bytes,
        test_pixels,
    ) = load_inputs()
    if not disease_mode:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    blobs = load_blobs(source_rows, paths, sealed=disease_mode)
    candidates = {
        **legacy,
        **{row["id"]: ("development", row) for row in mosaic},
    }
    records = []
    for sample_id, (split, member) in sorted(candidates.items()):
        record = {
            "id": sample_id,
            "split": split,
            "group_id": member.get("group_id", f"plantdoc:{member['component_id']}"),
            "source_image": member["image_path"],
            "source_split": member["source_split"],
            "source_label": member["source_label"],
            "subset": "mosaic_development_only"
            if member["source_label"] == MOSAIC
            else "inherited_split",
            "source_sha256": member["sha256"],
            "source_pixel_sha256": member["pixel_sha256"],
            "final_holdout_eligible": False,
        }
        if sample_id in paths:
            path = paths[sample_id]
            record.update(
                inspect(
                    sample_id,
                    member,
                    base.get(sample_id),
                    path,
                    source_rows[key(path.name)],
                    blobs,
                    test_bytes,
                    test_pixels,
                )
            )
        else:
            record["technical_exclusions"] = [
                f"retained_v2_exclusion: {member['reason']}"
            ]
        records.append(record)
    if disease_mode:
        disease_only(args.step, records, paths)
        return
    visual = [row for row in records if not row["technical_exclusions"]]
    for i, row in enumerate(visual):
        row["overlay"] = f"overlay-{i // 20 + 1:02d}.jpg"
    excluded = Counter(
        reason for row in records for reason in row["technical_exclusions"]
    )
    if args.step == "render":
        render(visual, paths)
        print("PLANTDOC_VISUAL_CANDIDATES", len(visual), dict(excluded))
        print(
            "SHEET_IDS",
            [
                [row["id"] for row in visual[i : i + 20]]
                for i in range(0, len(visual), 20)
            ],
        )
        return

    decisions_path = OUTPUT / "visual-decisions.csv"
    reviewed = table(decisions_path)
    decisions = {row["id"]: row["reason"] for row in reviewed}
    if len(decisions) != len(reviewed) or decisions.keys() != {
        row["id"] for row in visual
    }:
        raise ValueError("Expected exactly one visual decision per technical candidate")
    if set(decisions.values()) - VISUAL_REASONS.keys():
        raise ValueError("Unknown visual reason")
    for row in records:
        reason = decisions.get(row["id"])
        row["visual_review"] = reason
        row["decision"] = (
            "accepted" if reason == "single_leaf_units_visible" else "excluded"
        )
    accepted = [row for row in records if row["decision"] == "accepted"]
    summary = {}
    for split in ("train", "development"):
        rows = [row for row in accepted if row["split"] == split]
        summary[split] = {
            "images": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "boxes": dict(
                sorted(
                    Counter(
                        box["class_id"] for row in rows for box in row["boxes"]
                    ).items()
                )
            ),
        }
    for field in ("group_id", "source_sha256", "source_pixel_sha256"):
        split_values = [
            {row[field] for row in accepted if row["split"] == split}
            for split in summary
        ]
        if split_values[0] & split_values[1]:
            raise ValueError(f"Accepted split overlap: {field}")
    if any(
        row["source_label"] == MOSAIC and row["split"] != "development"
        for row in accepted
    ):
        raise ValueError("Mosaic reached training")
    inputs = [
        *INPUTS,
        OUTPUT / "publisher-train-blobs.json",
        decisions_path,
    ]
    stable_json(
        OUTPUT / "manifest.json",
        {
            "schema_version": 1,
            "source": SOURCE,
            "source_commit": COMMIT,
            "status": "leaf_admitted_for_detector_preparation",
            "training_run_verified": False,
            "expert_verified": False,
            "annotation_unit": "whole leaf or one visible leaflet blade",
            "coordinates": "Source CSV raw unrotated pixel xyxy; xywhn uses continuous width and height division without a one-pixel shift. Accepted EXIF orientation equals 1.",
            "inputs_sha256": {
                **{path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
                # The v1 artifact records the script revision that originally produced it.
                Path(__file__).relative_to(ROOT).as_posix(): V1_SCRIPT_SHA256,
            },
            "overlays_sha256": {
                name: sha256(OUTPUT / name)
                for name in sorted(
                    {"mosaic-details.jpg", *(row["overlay"] for row in visual)}
                )
            },
            "visual_review_reasons": VISUAL_REASONS,
            "source_to_admitted_class": {
                **{
                    source: name
                    for source, name in NAMES.items()
                    if source != "Tomato leaf"
                },
                "Tomato leaf": "healthy only on the folder-review-v2 approved healthy images",
                "Tomato leaf yellow virus": "yellow_leaf_curl only on the folder-review-v2 reviewed TYLCV images",
            },
            "summary": summary,
            "excluded_counts": dict(
                excluded
                + Counter(
                    reason
                    for reason in decisions.values()
                    if reason != "single_leaf_units_visible"
                )
            ),
            "records": records,
        },
    )
    print("PLANTDOC_LEAF_ADMISSION_VERIFIED", json.dumps(summary))


if __name__ == "__main__":
    main()
