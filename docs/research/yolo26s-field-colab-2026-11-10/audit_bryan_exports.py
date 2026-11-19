"""Audit pinned COCO exports, pixel links and whole-leaf review evidence."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "infra/derived/yolo26s-bryan-export-comparison-v1"
SOURCES = {
    "v63": ("yolo26s-bryan-v63-source-audit-v1", "7ddd592de64868a5ee1562d69c30fef04259d34a8ecddaa9fbe320bbbe806112"),
    "fork_v1": ("yolo26s-bryan-fork-v1-source-audit-v1", "544acea575f57c9d95eb1c3ffc1ff77d8b90d9484fd527b213d8d048fb29281d"),
}


def filename_key(name):
    """A candidate join only; filenames never prove pixel identity or plant groups."""
    stem = PurePosixPath(name.replace("\\", "/")).name.split(".rf.", 1)[0].casefold()
    stem = re.sub(r"[._-](?:jpg|jpeg|png)$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem)


def valid_box(box, width, height):
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return False
    x, y, w, h = box
    return w > 0 and h > 0 and x >= -0.5 and y >= -0.5 and x + w <= width + 0.5 and y + h <= height + 0.5


def pair_boxes(original, reference, tolerance=2 / 512):
    """Require a unique bijection in normalized source geometry, without rotating boxes."""
    if len(original["boxes"]) != len(reference["boxes"]) or not original["boxes"]:
        return None
    pairs = []
    for box in original["boxes"]:
        normalized = [v / original[axis] for v, axis in zip(box["bbox_xywh"], ("width", "height", "width", "height"))]
        matches = []
        for i, other in enumerate(reference["boxes"]):
            coordinates = [v / reference[axis] for v, axis in zip(other["bbox_xywh"], ("width", "height", "width", "height"))]
            if max(abs(a - b) for a, b in zip(normalized, coordinates)) <= tolerance:
                matches.append(i)
        if len(matches) != 1:
            return None
        i = matches[0]
        if box["source_class"] != "Other Disease" and box["source_class"] != reference["boxes"][i]["source_class"]:
            return None
        pairs.append(i)
    return pairs if len(set(pairs)) == len(pairs) else None


def compare_pixels(original, reference):
    """Conservative correspondence after resize/JPEG; never a plant-group certificate."""
    import numpy as np
    from PIL import Image

    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    if float(reference_array.std()) < 10:
        return {"accepted": False, "reason": "insufficient_pixel_variation"}
    candidates = []
    for resampler in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS):
        array = np.asarray(original.convert("RGB").resize(reference.size, resampler), dtype=np.float32)
        difference = np.abs(array - reference_array)
        left, right = array.ravel(), reference_array.ravel()
        left, right = left - left.mean(), right - right.mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denominator) if denominator else 0.0
        mae, p95 = float(difference.mean()), float(np.percentile(difference, 95))
        candidates.append({"accepted": mae <= 4.0 and p95 <= 12.0 and correlation >= 0.995, "mae": mae, "p95_absolute_error": p95, "correlation": correlation, "resampler": resampler.name, "exact_resized_pixels": bool(np.array_equal(array, reference_array))})
    return min(candidates, key=lambda row: (not row["accepted"], row["mae"]))


def pixel_links():
    from PIL import Image
    from PIL import __version__ as pillow_version

    audit_path = OUTPUT / "audit.json"
    audit_bytes = audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != "b73e87c4fa454d1746782bd378e272507505200649ff7ef15f62d93facd72e97":
        raise ValueError("Metadata audit changed")
    audit = json.loads(audit_bytes)
    exports = audit["exports"]
    for export in exports.values():
        with (ROOT / export["archive"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != export["archive_sha256"]:
                raise ValueError("Export archive changed")
    index = defaultdict(list)
    for record in exports["v63"]["records"]:
        if not record["collage_filename"] and not record["errors"]:
            index[record["filename_key"]].append(record)
    records = []
    with zipfile.ZipFile(ROOT / exports["fork_v1"]["archive"]) as original_zip, zipfile.ZipFile(ROOT / exports["v63"]["archive"]) as reference_zip:
        for original in exports["fork_v1"]["records"]:
            if original["collage_filename"] or original["errors"]:
                continue
            row = {"fork_member": original["member"], "fork_sha256": original["sha256"], "filename_key": original["filename_key"], "status": "no_verified_correspondence", "training_admitted": False, "final_holdout_eligible": False, "matched_references": []}
            compatible = [(reference, pair_boxes(original, reference)) for reference in index[original["filename_key"]]]
            compatible = [(reference, pairs) for reference, pairs in compatible if pairs is not None]
            row["geometry_candidates"] = len(compatible)
            if compatible:
                with Image.open(io.BytesIO(original_zip.read(original["member"]))) as original_image:
                    if original_image.size != (original["width"], original["height"]) or original_image.getexif().get(274, 1) != 1:
                        row["status"] = "unresolved_dimensions_or_orientation"
                    else:
                        for reference, pairs in compatible:
                            with Image.open(io.BytesIO(reference_zip.read(reference["member"]))) as reference_image:
                                if reference_image.size != (reference["width"], reference["height"]) or reference_image.getexif().get(274, 1) != 1:
                                    continue
                                metric = compare_pixels(original_image, reference_image)
                            if metric["accepted"]:
                                row["matched_references"].append({"member": reference["member"], "sha256": reference["sha256"], "pixel_comparison": metric, "annotations": [{"fork_annotation_id": original["boxes"][i]["annotation_id"], "v63_annotation_id": reference["boxes"][j]["annotation_id"], "v63_source_class": reference["boxes"][j]["source_class"]} for i, j in enumerate(pairs)]})
            matches = row["matched_references"]
            signatures = {tuple(a["v63_source_class"] for a in match["annotations"]) for match in matches}
            if len(signatures) == 1:
                row["status"] = "correspondence_verified_pending_source_and_visual_admission"
            elif len(signatures) > 1:
                row["status"] = "conflicting_reference_labels"
            records.append(row)
            if len(records) % 250 == 0:
                print(json.dumps({"processed": len(records), "verified": sum(r["status"].startswith("correspondence_verified") for r in records)}), flush=True)
    result = {"status": "pixel_annotation_correspondence_only_not_training_admission", "pillow_version": pillow_version, "metadata_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(), "thresholds": {"mae_max": 4.0, "absolute_error_p95_max": 12.0, "correlation_min": 0.995, "box_coordinate_tolerance_normalized": 2 / 512}, "counts": dict(Counter(r["status"] for r in records)), "records": records, "limitations": ["No automatic disease mapping or training admission.", "Pixel correspondence does not certify disease diagnosis, leaf-box completeness, rights, or independent plant groups.", "No source images displayed; original data and labels remain unchanged."]}
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    destination = OUTPUT / "pixel-links-v1.json"
    if destination.exists() and destination.read_bytes() != encoded:
        raise ValueError("Pixel evidence changed; use a new version")
    destination.write_bytes(encoded)
    print(json.dumps({"counts": result["counts"], "sha256": hashlib.sha256(encoded).hexdigest()}, indent=2))


def render_link_samples():
    from PIL import Image, ImageDraw, ImageOps

    link_bytes = (OUTPUT / "pixel-links-v1.json").read_bytes()
    if hashlib.sha256(link_bytes).hexdigest() != "323336ada4da2f178ff45f1264341bbeac604c0d6e75a3703c0618336daed367":
        raise ValueError("Pixel evidence changed")
    links = json.loads(link_bytes)
    audit = json.loads((OUTPUT / "audit.json").read_bytes())
    original = {r["member"]: r for r in audit["exports"]["fork_v1"]["records"]}
    known_plantdoc = set(audit["overlap_with_existing_data"]["fork_v1"]["all_plantdoc_filename_candidates"])
    exposure = json.loads((ROOT / "docs/progress/yolo26s-field-colab-2026-11-10/baseline-exposure.json").read_bytes())
    protected = json.loads((ROOT / "docs/progress/yolo26s-field-colab-2026-11-10/protected-splits.json").read_bytes())
    excluded_hashes = set(exposure["known_byte_sha256"]) | set(protected["protected_byte_sha256"])
    candidates = [r for r in links["records"] if r["status"].startswith("correspondence_verified") and r["fork_member"].startswith("train/") and r["fork_member"] not in known_plantdoc and r["fork_sha256"] not in excluded_hashes and not r["filename_key"].startswith(("v001", "v006"))]
    classes = sorted({b["v63_source_class"] for r in candidates for b in r["matched_references"][0]["annotations"]} - {"Healthy"})
    selected, hashes = [], set()
    for name in classes:
        pool = [r for r in candidates if name in {b["v63_source_class"] for b in r["matched_references"][0]["annotations"]}]
        chosen = 0
        for row in sorted(pool, key=lambda r: hashlib.sha256(r["fork_member"].encode()).hexdigest()):
            if row["fork_sha256"] not in hashes:
                selected.append({**row, "sampling_class": name, "visual_review_status": "pending", "audit_exposed": True, "related_group_holdout_quarantine": "required_before_final_split"})
                hashes.add(row["fork_sha256"])
                chosen += 1
            if chosen == 3:
                break
    directory = OUTPUT / "visual-review-v1"
    directory.mkdir(exist_ok=True)
    selection = {"purpose": "train_source_quality_audit_not_final_holdout", "pixel_links_sha256": hashlib.sha256(link_bytes).hexdigest(), "selection_rule": "Three unique source-train images per disease, SHA-256 filename order; exclude known PlantDoc-name matches, exact baseline/protected hashes, and AIHub filename families.", "records": selected}
    encoded = (json.dumps(selection, indent=2) + "\n").encode()
    path = directory / "selection.json"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Selection changed; preserve prior exposure evidence")
    if not path.exists():
        path.write_bytes(encoded)
    with zipfile.ZipFile(ROOT / audit["exports"]["fork_v1"]["archive"]) as archive:
        for start in range(0, len(selected), 4):
            destination = directory / f"overlay-{start // 4 + 1:02d}.jpg"
            if destination.exists():
                continue
            sheet = Image.new("RGB", (1400, 1160), "white")
            for slot, row in enumerate(selected[start:start + 4]):
                source = original[row["fork_member"]]
                payload = archive.read(row["fork_member"])
                if hashlib.sha256(payload).hexdigest() != row["fork_sha256"]:
                    raise ValueError("Selected source image changed")
                labels = {b["fork_annotation_id"]: b["v63_source_class"] for b in row["matched_references"][0]["annotations"]}
                with Image.open(io.BytesIO(payload)) as raw:
                    rendered = raw.convert("RGB")
                draw = ImageDraw.Draw(rendered)
                for box in source["boxes"]:
                    x, y, w, h = box["bbox_xywh"]
                    name = labels[box["annotation_id"]]
                    draw.rectangle((x, y, x + w, y + h), outline="cyan" if name == "Healthy" else "red", width=max(2, rendered.width // 300))
                    draw.text((x, y), name, fill="red", stroke_width=1, stroke_fill="white")
                rendered = ImageOps.contain(rendered, (700, 530))
                x, y = slot % 2 * 700, slot // 2 * 580
                sheet.paste(rendered, (x + (700 - rendered.width) // 2, y))
                caption = f"{start + slot + 1:02d} {row['sampling_class']}\n{source['source_name'][:80]}"
                ImageDraw.Draw(sheet).text((x + 5, y + 533), caption, fill="black", font_size=16)
            sheet.save(destination, quality=95)
    print(json.dumps({"selected": len(selected), "class_counts": dict(Counter(r["sampling_class"] for r in selected)), "selection_sha256": hashlib.sha256(encoded).hexdigest()}))


def exposure_closure(records, seeds, families=()):
    """Propagate audit exposure; filename/family joins are conservative exclusions."""
    if any(not family for family in families):
        raise ValueError("Empty capture-family prefix")
    keys = {row["filename_key"] for row in seeds if row["filename_key"]}
    hashes = {row["sha256"] for row in seeds if row["sha256"]}
    exposed = set()
    changed = True
    while changed:
        changed = False
        for row in records:
            identity = (row["export"], row["member"])
            key, digest = row["filename_key"], row["sha256"]
            related = key in keys or digest in hashes or any(key.startswith(prefix) for prefix in families)
            if identity not in exposed and related:
                exposed.add(identity)
                if key:
                    keys.add(key)
                if digest:
                    hashes.add(digest)
                changed = True
    return exposed


def geometry_candidates(selection, review):
    """Only revisit exposed images with a matching sealed source identity."""
    by_member = {row["fork_member"]: row for row in selection["records"]}
    chosen = []
    for decision in review["reviewed_images"]:
        if decision["visual_status"] != "geometry_candidate":
            continue
        source = by_member.get(decision["fork_member"])
        if source is None or source["fork_sha256"] != decision["fork_sha256"]:
            raise ValueError("Reviewed source identity changed")
        chosen.append({**source, "sample": decision["sample"]})
    return chosen


def render_geometry_details():
    from PIL import Image, ImageDraw, ImageOps

    pins = {"audit.json": "b73e87c4fa454d1746782bd378e272507505200649ff7ef15f62d93facd72e97",
            "visual-review-v1/selection.json": "be2d881ae3051df88d1c3341731383c1e4bd8ad91001de8ecf268720becdaad7",
            "visual-review-v1/review.json": "340860d2e56433f3c1209e57f03fb0295d30eaa906195c81ce7e3ced2353e7c3"}
    inputs = {}
    for name, expected in pins.items():
        raw = (OUTPUT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Evidence changed: {name}")
        inputs[name] = json.loads(raw)
    selected = geometry_candidates(inputs["visual-review-v1/selection.json"], inputs["visual-review-v1/review.json"])
    export = inputs["audit.json"]["exports"]["fork_v1"]
    original = {row["member"]: row for row in export["records"]}
    directory = OUTPUT / "detail-review-v1"
    directory.mkdir(exist_ok=True)
    selection = {"purpose": "source_resolution_review_of_prior_geometry_candidates", "inputs_sha256": pins,
                 "new_audit_exposure": 0, "training_admitted": False, "final_holdout_eligible": False,
                 "records": [{**row, "fork_source_record": original[row["fork_member"]]} for row in selected]}
    encoded = (json.dumps(selection, ensure_ascii=False, indent=2) + "\n").encode()
    path = directory / "selection.json"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Detail selection changed; use a new version")
    if not path.exists():
        path.write_bytes(encoded)
    with zipfile.ZipFile(ROOT / export["archive"]) as archive:
        for row in selected:
            payload = archive.read(row["fork_member"])
            if hashlib.sha256(payload).hexdigest() != row["fork_sha256"]:
                raise ValueError("Selected source image changed")
            labels = {box["fork_annotation_id"]: box["v63_source_class"] for box in row["matched_references"][0]["annotations"]}
            with Image.open(io.BytesIO(payload)) as source:
                rendered = source.convert("RGB")
            draw = ImageDraw.Draw(rendered)
            for box in original[row["fork_member"]]["boxes"]:
                x, y, w, h = box["bbox_xywh"]
                draw.rectangle((x, y, x + w, y + h), outline="cyan" if labels[box["annotation_id"]] == "Healthy" else "red", width=max(2, rendered.width // 400))
            destination = directory / f"detail-{int(row['sample']):02d}.jpg"
            if not destination.exists():
                ImageOps.contain(rendered, (1600, 1600)).save(destination, quality=96)
    print(json.dumps({"selected": len(selected), "new_audit_exposure": 0, "selection_sha256": hashlib.sha256(encoded).hexdigest()}))


def screen_geometry_candidates():
    import fieldplant_near_duplicates as near
    import numpy as np
    from PIL import Image

    directory = OUTPUT / "detail-review-v1"
    raw = (directory / "selection.json").read_bytes()
    selection_sha = hashlib.sha256(raw).hexdigest()
    if selection_sha != "581af616e070ecda79836dc71aaaec61166791c424a6bd53f90cd7fec12cf9a5":
        raise ValueError("Detail selection changed")
    selected = json.loads(raw)["records"]
    if len(selected) != 7 or len({row["fork_member"] for row in selected}) != 7:
        raise ValueError("Unexpected query coverage")
    sources, _, _, digests = near.audit_inputs()
    print(f"Verifying and fingerprinting {len(sources)} references; pixels will not be displayed", flush=True)
    fingerprints, coverage, unavailable = near.collect_signatures(sources, {}, jpeg_reference_previews=True)
    ids = list(fingerprints)
    hashes = np.array([fingerprints[key][0][0] for key in ids], dtype=np.uint64)
    pixels = np.stack([fingerprints[key][0][1] for key in ids])
    audit_raw = (OUTPUT / "audit.json").read_bytes()
    if hashlib.sha256(audit_raw).hexdigest() != "b73e87c4fa454d1746782bd378e272507505200649ff7ef15f62d93facd72e97":
        raise ValueError("Metadata audit changed")
    archive_path = ROOT / json.loads(audit_raw)["exports"]["fork_v1"]["archive"]
    results = []
    with zipfile.ZipFile(archive_path) as archive:
        for row in selected:
            payload = archive.read(row["fork_member"])
            if hashlib.sha256(payload).hexdigest() != row["fork_sha256"]:
                raise ValueError("Selected source image changed")
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != (row["fork_source_record"]["width"], row["fork_source_record"]["height"]) or image.getexif().get(274, 1) != 1:
                    raise ValueError("Query dimensions or orientation changed")
                query = near.signatures(image.convert("RGB"), near.TRANSFORMS)
            findings = near.compare(query, hashes, pixels)
            matches = []
            for index, (bits, mae, transform) in sorted(findings.items()):
                key = ids[index]
                reference = sources[key]
                if not (bits <= near.STRICT_BITS or (bits <= near.MAX_BITS and mae <= near.MAX_RGB_MAE)):
                    raise ValueError("Similarity threshold disagrees")
                matches.append({"id": key, "path": reference.path.relative_to(ROOT).as_posix(), "sha256": reference.sha256,
                                "byte_identical": reference.sha256 == row["fork_sha256"], "phash_bits": bits,
                                "rgb_mae": round(mae, 3), "query_transform": transform, "flags": sorted(reference.flags),
                                "group_keys": sorted(reference.keys), "refs": sorted(reference.refs)})
            results.append({"sample": row["sample"], "fork_member": row["fork_member"], "fork_sha256": row["fork_sha256"], "matches": matches})
    result = {"status": "similarity_candidates_only_not_admission", "selection_sha256": selection_sha,
              "reference_inputs_sha256": digests, "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "query_count": len(selected), "reference_count": len(sources), "fingerprinted_references": len(ids),
              "coverage": dict(coverage), "unavailable_references": unavailable,
              "method": "Existing 63-bit pHash; eight query rotations/mirrors; <=4 bits or <=8 bits with RGB MAE <=24. Oversized JPEG reference preview as in FieldPlant v2.",
              "limitations": "Whole-frame screen may miss crops, collage constituents and same-plant views. Flags describe pinned input ledgers, not a complete current exposure closure. No reference pixels were displayed.",
              "training_admitted": False, "final_holdout_eligible": False, "records": results}
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    destination = directory / "near-duplicates.json"
    if destination.exists() and destination.read_bytes() != encoded:
        raise ValueError("Similarity evidence changed; use a new version")
    if not destination.exists():
        destination.write_bytes(encoded)
    print(json.dumps({"queries": len(selected), "references": len(ids), "unavailable": len(unavailable), "matches": sum(len(row["matches"]) for row in results), "sha256": hashlib.sha256(encoded).hexdigest()}))


def save_visual_review():
    directory = OUTPUT / "visual-review-v1"
    selection_bytes = (directory / "selection.json").read_bytes()
    if hashlib.sha256(selection_bytes).hexdigest() != "be2d881ae3051df88d1c3341731383c1e4bd8ad91001de8ecf268720becdaad7":
        raise ValueError("Visual selection changed")
    selected = json.loads(selection_bytes)["records"]
    with (directory / "decisions.tsv").open(encoding="utf-8", newline="") as stream:
        decisions = list(csv.DictReader(stream, delimiter="\t"))
    if len(decisions) != len(selected):
        raise ValueError("Every selected image needs a decision")
    for index, (decision, row) in enumerate(zip(decisions, selected), 1):
        if decision["sample"] != str(index) or decision["filename_key"] != row["filename_key"] or not decision["observation"]:
            raise ValueError("Decision identity mismatch")
    audit_bytes = (OUTPUT / "audit.json").read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != "b73e87c4fa454d1746782bd378e272507505200649ff7ef15f62d93facd72e97":
        raise ValueError("Metadata audit changed")
    audit = json.loads(audit_bytes)
    prior_path = ROOT / "infra/derived/yolo26s-field-source-audit-v1/bryan-access/audit.json"
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)["train_overlay_samples"]
    records = [{**row, "export": name} for name, export in audit["exports"].items() for row in export["records"]]
    seeds = [{"filename_key": row["filename_key"], "sha256": row["fork_sha256"]} for row in selected]
    seeds += [{"filename_key": filename_key(row["image"]), "sha256": row["image_sha256"]} for row in prior]
    excluded = exposure_closure(records, seeds, ("iphone690c0",))
    quarantined = [{key: row[key] for key in ("export", "member", "filename_key", "sha256", "split")} for row in records if (row["export"], row["member"]) in excluded]
    evidence = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("overlay-*.jpg"))}
    evidence.update({"selection.json": hashlib.sha256(selection_bytes).hexdigest(), "decisions.tsv": hashlib.sha256((directory / "decisions.tsv").read_bytes()).hexdigest(), "metadata_audit": hashlib.sha256(audit_bytes).hexdigest(), "prior_bryan_audit": hashlib.sha256(prior_bytes).hexdigest()})
    result = {
        "status": "visual_triage_and_audit_exposure_only",
        "evidence_sha256": evidence,
        "training_admitted": 0,
        "final_holdout_selected": 0,
        "reviewed_images": [{**decision, "fork_member": row["fork_member"], "fork_sha256": row["fork_sha256"], "training_admitted": False, "final_holdout_eligible": False} for decision, row in zip(decisions, selected)],
        "visual_counts": dict(Counter(row["visual_status"] for row in decisions)),
        "exposure_seed_images": {"current_review": len(selected), "prior_review": len(prior)},
        "quarantine_rule": "Transitive shared byte SHA-256 or normalized filename; entire iphone690c0 family excluded conservatively after related-scene observations in samples 12 and 14. Family does not prove a single plant or capture session.",
        "quarantined_by_export": dict(Counter(row["export"] for row in quarantined)),
        "quarantined_records": sorted(quarantined, key=lambda row: (row["export"], row["member"])),
        "limitations": ["Geometry candidates still need source-resolution, provenance, rights and cross-source deduplication checks.", "Sheet observations do not independently diagnose disease or prove exhaustive labels.", "Unlisted records are not certified holdout eligible. Near duplicates, crops and collage constituents remain unresolved."],
    }
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    path = directory / "review.json"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Review changed; preserve it and create a new version")
    path.write_bytes(encoded)
    print(json.dumps({"visual_counts": result["visual_counts"], "quarantined_by_export": result["quarantined_by_export"], "review_sha256": hashlib.sha256(encoded).hexdigest()}))


def load_export(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Archive hash mismatch: {path.name}")
    records, summaries, readmes = [], {}, {}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(set(members)) != len(members):
            raise ValueError("Duplicate ZIP members")
        for split in ("train", "valid", "test"):
            member = f"{split}/_annotations.coco.json"
            raw = archive.read(member)
            coco = json.loads(raw)
            categories = {row["id"]: row["name"] for row in coco["categories"]}
            images = {row["id"]: row for row in coco["images"]}
            if len(images) != len(coco["images"]) or len(categories) != len(coco["categories"]):
                raise ValueError("Duplicate COCO IDs")
            grouped = defaultdict(list)
            seen_annotations = set()
            for annotation in coco["annotations"]:
                if annotation["id"] in seen_annotations or annotation["image_id"] not in images or annotation["category_id"] not in categories:
                    raise ValueError("Duplicate or unresolvable annotation")
                seen_annotations.add(annotation["id"])
                grouped[annotation["image_id"]].append(annotation)
            split_records = []
            for image_id, image in images.items():
                name = image["file_name"]
                if PurePosixPath(name).name != name or "\\" in name or image["width"] <= 0 or image["height"] <= 0:
                    raise ValueError("Unsafe filename or invalid dimensions")
                image_member = f"{split}/{name}"
                if image_member not in members:
                    raise ValueError(f"Missing image: {image_member}")
                # Raw hashing does not decode pixels or expose protected images to a reviewer.
                digest = hashlib.sha256(archive.read(image_member)).hexdigest()
                boxes = [{"annotation_id": a["id"], "source_class": categories[a["category_id"]], "bbox_xywh": a["bbox"], "iscrowd": a.get("iscrowd", 0)} for a in grouped[image_id]]
                errors = []
                if not boxes:
                    errors.append("no_boxes")
                if any(not valid_box(b["bbox_xywh"], image["width"], image["height"]) for b in boxes):
                    errors.append("invalid_box_geometry")
                if any(b["iscrowd"] != 0 for b in boxes):
                    errors.append("crowd_annotation")
                row = {"split": split, "source_image_id": image_id, "member": image_member, "source_name": image.get("extra", {}).get("name", name), "filename_key": filename_key(name), "width": image["width"], "height": image["height"], "sha256": digest, "collage_filename": "collage" in name.casefold(), "boxes": boxes, "errors": errors}
                split_records.append(row)
            records.extend(split_records)
            summaries[split] = {"images": len(split_records), "annotations": len(coco["annotations"]), "categories": categories, "annotation_sha256": hashlib.sha256(raw).hexdigest(), "collage_filename_images": sum(r["collage_filename"] for r in split_records), "images_with_geometry_or_metadata_errors": sum(bool(r["errors"]) for r in split_records), "dimensions": dict(Counter(f'{r["width"]}x{r["height"]}' for r in split_records)), "box_counts": dict(Counter(b["source_class"] for r in split_records for b in r["boxes"]))}
        for member in ("README.dataset.txt", "README.roboflow.txt"):
            readmes[member] = archive.read(member).decode("utf-8")
    cross_split = {}
    for field in ("filename_key", "sha256"):
        grouped = defaultdict(list)
        for row in records:
            grouped[row[field]].append(row["member"])
        cross_split[field] = {key: values for key, values in grouped.items() if len({v.split("/")[0] for v in values}) > 1}
    return {"archive": path.relative_to(ROOT).as_posix(), "archive_sha256": actual, "splits": summaries, "readmes": readmes, "cross_split": cross_split, "records": records}


def main():
    exports = {name: load_export(ROOT / "infra/derived" / folder / "raw/export.coco.zip", digest) for name, (folder, digest) in SOURCES.items()}
    indexes = {name: defaultdict(list) for name in exports}
    for name, export in exports.items():
        for record in export["records"]:
            indexes[name][record["filename_key"]].append(record)
    protection = json.loads((ROOT / "docs/progress/yolo26s-field-colab-2026-11-10/protected-splits.json").read_bytes())
    with (ROOT / "infra/derived/plantdoc-folder-review-v2/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        plantdoc = list(csv.DictReader(stream))
    protected_ids = set(protection["plantdoc_train_protected_ids"])
    protected_names = {filename_key(row["image_path"]) for row in plantdoc if row["id"] in protected_ids}
    all_names = {filename_key(row["image_path"]) for row in plantdoc}
    overlap = {}
    for name, export in exports.items():
        rows = export["records"]
        overlap[name] = {
            "all_plantdoc_filename_candidates": [r["member"] for r in rows if r["filename_key"] in all_names],
            "protected_plantdoc_filename_candidates": [r["member"] for r in rows if r["filename_key"] in protected_names],
            "protected_exact_byte_matches": [r["member"] for r in rows if r["sha256"] in protection["protected_byte_sha256"]],
        }
    shared = sorted(indexes["v63"].keys() & indexes["fork_v1"].keys())
    result = {"status": "metadata_audit_only_not_training_admission", "pixels_decoded_or_viewed": False, "exports": exports, "overlap_with_existing_data": overlap, "shared_filename_keys": shared, "fork_keys_missing_from_v63": sorted(indexes["fork_v1"].keys() - indexes["v63"].keys()), "limitations": ["Filename matches are candidates, not image identity.", "No independent plant/scene grouping established.", "Fork class Other Disease is not an admissible specific disease label.", "No labels are restored from names or predictions."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    target = OUTPUT / "audit.json"
    if target.exists() and target.read_bytes() != encoded:
        raise ValueError("Audit changed; use a new evidence version")
    target.write_bytes(encoded)
    print(json.dumps({"audit_sha256": hashlib.sha256(encoded).hexdigest(), "shared_filename_keys": len(shared), "fork_keys_missing_from_v63": len(result["fork_keys_missing_from_v63"]), "sources": {name: {"images": len(export["records"]), "noncollage_images": sum(not r["collage_filename"] for r in export["records"]), "cross_split_filename_keys": len(export["cross_split"]["filename_key"]), "cross_split_byte_hashes": len(export["cross_split"]["sha256"]), "overlap_counts": {key: len(values) for key, values in overlap[name].items()}} for name, export in exports.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-links", action="store_true", help="Verify cross-export pixels and box correspondence without admitting data")
    parser.add_argument("--render-link-samples", action="store_true", help="Render a fixed training-source sample and mark its holdout exposure")
    parser.add_argument("--save-visual-review", action="store_true", help="Seal manual decisions and propagate audit exposure across exports")
    parser.add_argument("--render-geometry-details", action="store_true", help="Revisit only prior geometry candidates at source resolution")
    parser.add_argument("--screen-geometry-candidates", action="store_true", help="Compare seven exposed candidates with pinned source references without showing references")
    arguments = parser.parse_args()
    if sum((arguments.pixel_links, arguments.render_link_samples, arguments.save_visual_review, arguments.render_geometry_details, arguments.screen_geometry_candidates)) > 1:
        parser.error("Run one audit phase at a time")
    if arguments.pixel_links:
        pixel_links()
    elif arguments.render_link_samples:
        render_link_samples()
    elif arguments.save_visual_review:
        save_visual_review()
    elif arguments.render_geometry_details:
        render_geometry_details()
    elif arguments.screen_geometry_candidates:
        screen_geometry_candidates()
    else:
        main()
