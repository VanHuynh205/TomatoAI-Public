"""Audit FieldPlant v11 tomato ZIP members or derive a local conflict ledger."""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from urllib3.exceptions import HTTPError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "infra/derived/yolo26s-fieldplant-source-audit-v1"
PREVIOUS_AUDIT = ROOT / "infra/derived/yolo26s-field-source-audit-v1"
PROGRESS = ROOT / "docs/progress" / HERE.name
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "prathmeshnikam821/fieldplant?datasetVersionNumber=1"
)
ZIP_BYTES = 5_912_343_540
ZIP_ETAG = '"259399d895cbaa190eb4e441fdd4c07e"'
RANGE_LIMIT = 20_000_000
COCO_SHA = "b44d734b1b90b498d327f25702bd527b9e4fd61607da4d72c8572d2137c87d79"
SOURCE_MANIFEST_SHA = "1a32e70d0931ca0636a6b49073e6231b718e307bac74e2c54bb9e23b23ce4b53"
SCENE_LINKS = ((3216, 3323), (3531, 4443))

sys.path.insert(0, str(ROOT))
from ai.data import _pixel_hash_stream  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def save(path, data, *, newline=None):
    pending = path.with_suffix(".part")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=newline)
    os.replace(pending, path)


class RangeReader:
    """Seekable ZIP reader with a bounded, exact HTTP Range backing."""

    def __init__(self, session, url, size, etag, directory_reads=None):
        self.session = session
        self.url = url
        self.size = size
        self.etag = etag
        self.directory_reads = {} if directory_reads is None else directory_reads
        self.position = 0
        self.window = None
        self.transfer_bytes = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        origin = (0, self.position, self.size)[whence]
        if origin + offset < 0:
            raise ValueError("negative ZIP seek")
        self.position = origin + offset
        return self.position

    def seekable(self):
        return True

    def fetch(self, start, end):
        if not (0 <= start < end <= self.size and end - start <= RANGE_LIMIT):
            raise ValueError("unsafe ZIP range")
        headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
        for attempt in range(3):
            try:
                with self.session.get(self.url, headers=headers, timeout=(10, 45), stream=True) as response:
                    self.url = response.url
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise OSError(f"transient HTTP {response.status_code}")
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{self.size}"
                        or response.headers.get("ETag") != self.etag
                        or response.headers.get("Content-Encoding", "identity") != "identity"
                    ):
                        raise ValueError(f"unexpected ZIP range response (HTTP {response.status_code})")
                    raw = response.raw.read(end - start + 1)
                    self.transfer_bytes += len(raw)
                    if len(raw) != end - start:
                        raise OSError("truncated ZIP range")
                    return raw
            except (OSError, requests.RequestException, HTTPError) as error:
                if attempt == 2:
                    raise OSError(f"ZIP range {start}-{end - 1} failed ({type(error).__name__})") from None
                time.sleep(2**attempt)

    def read(self, length=-1):
        start = self.position
        end = min(self.size, start + length) if length >= 0 else self.size
        if start >= end:
            return b""
        self.position = end
        if (start, end) in self.directory_reads:
            return self.directory_reads[start, end]
        if self.window is not None:
            offset, raw = self.window
            if offset <= start < end <= offset + len(raw):
                return raw[start - offset : end - offset]
            raise ValueError("ZIP read outside the requested member span")
        raw = self.fetch(start, end)
        self.directory_reads[start, end] = raw
        return raw


def member_ranges(archive, records):
    selected = {row["archive_member"]: row for row in records}
    ordered = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
    found = set()
    jobs = []
    for index, entry in enumerate(ordered):
        if entry.filename not in selected:
            continue
        start = entry.header_offset
        end = ordered[index + 1].header_offset if index + 1 < len(ordered) else archive.start_dir
        if (
            entry.filename in found
            or entry.is_dir()
            or not 0 < entry.file_size <= RANGE_LIMIT
            or not 0 < end - start <= RANGE_LIMIT
        ):
            raise ValueError("invalid tomato ZIP member")
        found.add(entry.filename)
        row = selected[entry.filename]
        row["zip_entry"] = {
            "crc32": f"{entry.CRC:08x}",
            "uncompressed_bytes": entry.file_size,
            "compressed_bytes": entry.compress_size,
            "local_record_start": start,
            "local_record_end_exclusive": end,
        }
        if jobs and jobs[-1][1] == start and end - jobs[-1][0] <= RANGE_LIMIT:
            jobs[-1] = (jobs[-1][0], end, [*jobs[-1][2], row])
        else:
            jobs.append((start, end, [row]))
    if found != selected.keys():
        raise ValueError("tomato member missing from ZIP central directory")
    return jobs


def source_records():
    coco_path = HERE / "fieldplant-metadata/train_annotations.coco.json"
    raw = coco_path.read_bytes()
    if digest(raw) != COCO_SHA:
        raise ValueError("FieldPlant COCO checksum changed")
    coco = json.loads(raw)
    names = {row["id"]: row["name"] for row in coco["categories"]}
    boxes = defaultdict(list)
    for box in coco["annotations"]:
        boxes[box["image_id"]].append(box)
    rows = []
    for image in coco["images"]:
        annotations = boxes[image["id"]]
        if not any(names[box["category_id"]].startswith("Tomato ") for box in annotations):
            continue
        filename = image["file_name"]
        if not re.fullmatch(r"[\w .()'-]+\.jpg", filename, flags=re.IGNORECASE):
            raise ValueError("unsafe FieldPlant image basename")
        original_name, separator, _ = filename.partition(".rf.")
        if not separator:
            raise ValueError("missing Roboflow filename suffix")
        rows.append({
            "id": f"fieldplant-{image['id']}",
            "image": image,
            "archive_member": f"train/{filename}",
            "source_split": "train",
            "filename_group_hint": original_name,
            "filename_date_tokens": re.findall(r"(?<!\d)(20\d{6})(?=_|\d{6}_)", original_name),
            "annotations": [
                {
                    **box,
                    "source_label": names[box["category_id"]],
                    "target_status": (
                        "excluded_healthy"
                        if names[box["category_id"]] == "Tomato healthy"
                        else "disease_scope_pending" if names[box["category_id"]].startswith("Tomato ")
                        else "excluded_other_crop"
                    ),
                }
                for box in annotations
            ],
            "training_admitted": False,
            "final_holdout_eligible": False,
            "visual_review_status": "unreviewed",
        })
    if len(rows) != 1403 or len({row["archive_member"] for row in rows}) != len(rows):
        raise ValueError("unexpected tomato image inventory")
    return rows, coco["licenses"]


def verify_image(raw, row):
    entry = row["zip_entry"]
    if len(raw) != entry["uncompressed_bytes"] or f"{zlib.crc32(raw):08x}" != entry["crc32"]:
        raise ValueError("image size or CRC32 differs from ZIP central directory")
    expected = row["image"]["width"], row["image"]["height"]
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "JPEG" or image.size != expected:
            raise ValueError("image format or dimensions differ from COCO")
        exif = image.getexif()
        orientation = exif.get(274, 1)
        date_time = exif.get_ifd(34665).get(36867)
    pixel_sha, error = _pixel_hash_stream(io.BytesIO(raw))
    if error:
        raise ValueError(f"image decode failed ({error})")
    return {
        "status": "verified",
        "path": f"raw/{row['id']}.jpg",
        "bytes": len(raw),
        "sha256": digest(raw),
        "pixel_sha256": pixel_sha,
        "exif_orientation": orientation,
        "exif_date_time_original": date_time,
    }


def acquire(jobs, reader, previous):
    local = threading.local()

    def run(job):
        start, end, records = job
        if not hasattr(local, "reader"):
            local.reader = RangeReader(requests.Session(), reader.url, ZIP_BYTES, ZIP_ETAG, reader.directory_reads)
            local.archive = zipfile.ZipFile(local.reader)
        worker = local.reader
        worker.window = None
        before = worker.transfer_bytes
        failures = []
        for row in records:
            path = OUTPUT / "raw" / f"{row['id']}.jpg"
            try:
                cached = path.exists()
                if not cached and worker.window is None:
                    worker.window = (start, worker.fetch(start, end))
                raw = path.read_bytes() if cached else local.archive.read(row["archive_member"])
                result = verify_image(raw, row)
                old = previous.get(row["id"])
                if old and old.get("sha256") != result["sha256"]:
                    raise ValueError("previously verified image SHA-256 changed")
                if not cached:
                    pending = path.with_suffix(".part")
                    pending.write_bytes(raw)
                    os.replace(pending, path)
                row["acquisition"] = result
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                row["acquisition"] = {"status": "failed", "error": str(error)}
                failures.append(row["id"])
        worker.window = None
        return len(records), worker.transfer_bytes - before, failures

    done = 0
    transfer = reader.transfer_bytes
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, transferred, failures in pool.map(run, jobs):
            if failures:
                print("Failed", ", ".join(failures), flush=True)
            if done // 100 != (done + count) // 100:
                print(f"Processed {done + count}/1403, transferred {transfer + transferred} bytes", flush=True)
            done += count
            transfer += transferred
    return transfer


def record_groups(rows, selection, audit_downloads, baseline, protected):
    by_id = {row["id"]: row for row in rows}
    buckets = defaultdict(list)
    links = defaultdict(set)
    observed = {
        row["id"] for row in selection["records"]
        if row["source"] == "fieldplant-v11-kaggle-v1"
    }
    audit_sha = {row["id"]: row["sha256"] for row in audit_downloads}
    audit_bytes = set(audit_sha.values())
    if not observed <= by_id.keys():
        raise ValueError("prior audit IDs missing from COCO")
    baseline_sha = set(baseline["known_byte_sha256"]) | {
        row["legacy_source_sha256"] for row in baseline["source_records"] if "legacy_source_sha256" in row
    }
    baseline_pixels = set(baseline["known_pixel_sha256"]) | {
        row["legacy_source_pixel_sha256"] for row in baseline["source_records"] if "legacy_source_pixel_sha256" in row
    }
    protected_sha = set(protected["protected_byte_sha256"])
    protected_pixels = set(protected["protected_pixel_sha256"])
    for row in rows:
        key = row["id"]
        image = row["acquisition"]
        buckets["filename", row["filename_group_hint"]].append(key)
        reasons = []
        if key in observed:
            reasons.append("prior_visual_audit")
            row["visual_review_status"] = "prior_audit_sample"
        if image["status"] == "verified":
            if key in audit_sha and image["sha256"] != audit_sha[key]:
                raise ValueError(f"prior visual audit SHA-256 mismatch for {key}")
            buckets["bytes", image["sha256"]].append(key)
            buckets["pixels", image["pixel_sha256"]].append(key)
            if key not in observed and image["sha256"] in audit_bytes:
                reasons.append("prior_audit_exact")
            if image["sha256"] in baseline_sha or image["pixel_sha256"] in baseline_pixels:
                reasons.append("baseline_exposure_exact")
            if image["sha256"] in protected_sha or image["pixel_sha256"] in protected_pixels:
                reasons.append("protected_split_exact")
        row["direct_exclusion_reasons"] = reasons

    for keys in buckets.values():
        for key in keys[1:]:
            links[keys[0]].add(key)
            links[key].add(keys[0])
    for left, right in SCENE_LINKS:
        left_key, right_key = f"fieldplant-{left}", f"fieldplant-{right}"
        links[left_key].add(right_key)
        links[right_key].add(left_key)

    seen = set()
    for root in by_id:
        if root in seen:
            continue
        stack = [root]
        members = []
        while stack:
            key = stack.pop()
            if key not in seen:
                seen.add(key)
                members.append(key)
                stack.extend(links[key])
        reasons = sorted({reason for key in members for reason in by_id[key]["direct_exclusion_reasons"]})
        group = min(members, key=lambda key: by_id[key]["image"]["id"])
        for key in members:
            by_id[key]["provisional_group"] = group
            by_id[key]["group_exclusion_reasons"] = reasons
    return {
        "provisional_group_count": len({row["provisional_group"] for row in rows}),
        "duplicate_buckets": {
            kind: [keys for (name, _), keys in buckets.items() if name == kind and len(keys) > 1]
            for kind in ("filename", "bytes", "pixels")
        },
        "prior_visual_scene_links": SCENE_LINKS,
        "direct_exclusions": Counter(reason for row in rows for reason in row["direct_exclusion_reasons"]),
        "group_exclusions": Counter(reason for row in rows for reason in row["group_exclusion_reasons"]),
        "grouping_complete": False,
        "limitations": ["Filename, bytes, pixels and two prior visual links are partial clues.",
                        "Cross-source near duplicates, crops, plant and capture session remain unverified."],
    }


def annotation_conflicts(rows, pixel_duplicate_buckets):
    by_id = {row["id"]: row for row in rows}
    conflicts = []
    for ids in pixel_duplicate_buckets:
        boxes = [
            sorted((box["category_id"], tuple(box["bbox"])) for box in by_id[key]["annotations"])
            for key in ids
        ]
        if all(member == boxes[0] for member in boxes[1:]):
            continue
        box_counts = [len(member) for member in boxes]
        categories = [sorted({category for category, _ in member}) for member in boxes]
        for key in ids:
            by_id[key]["annotation_conflict_status"] = "unresolved_exact_pixel_duplicate"
        conflicts.append({
            "ids": ids,
            "box_counts": box_counts,
            "source_category_ids": categories,
            "box_counts_differ": len(set(box_counts)) > 1,
            "category_sets_differ": any(member != categories[0] for member in categories[1:]),
            "status": "unresolved",
        })
    return conflicts


def partition_summary(rows):
    pixels = defaultdict(set)
    groups = defaultdict(set)
    boxes = Counter()
    for row in rows:
        for box in row["annotations"]:
            label = box["source_label"]
            pixels[label].add(row["acquisition"]["pixel_sha256"])
            groups[label].add(row["provisional_group"])
            boxes[label] += 1
    return {
        "source_records": len(rows),
        "unique_pixel_images": len({row["acquisition"]["pixel_sha256"] for row in rows}),
        "provisional_groups": len({row["provisional_group"] for row in rows}),
        "source_boxes": sum(boxes.values()),
        "source_classes": {
            label: {"unique_pixel_images": len(pixels[label]), "provisional_groups": len(groups[label]),
                    "source_boxes": boxes[label]}
            for label in sorted(boxes)
        },
    }


def conflict_resolution(manifest):
    rows_by_group = defaultdict(list)
    for row in sorted(manifest["records"], key=lambda row: row["image"]["id"]):
        rows_by_group[row["provisional_group"]].append(row)
    conflicts = manifest["grouping"]["annotation_conflicts_on_exact_pixels"]
    conflict_ids = {key for conflict in conflicts for key in conflict["ids"]}
    prior_exposure_groups = 0
    groups = {}
    partitions = defaultdict(list)
    for group, rows in rows_by_group.items():
        group_conflicts = [row["id"] for row in rows if row["id"] in conflict_ids]
        reasons = {reason for row in rows for reason in row["group_exclusion_reasons"]}
        prior_exposure_groups += bool(reasons)
        if group_conflicts:
            reasons.add("unresolved_exact_pixel_annotation_conflict")
        statuses = {}
        for row in rows:
            if group_conflicts:
                status = "conflict_quarantine"
            elif reasons:
                status = "prior_exposure_group"
            elif all(box["target_status"] == "excluded_healthy" for box in row["annotations"]):
                status = "negative_visual_review_candidate"
            else:
                status = "disease_visual_review_candidate"
            statuses[row["id"]] = status
            partitions[status].append(row)
        groups[group] = {
            "conflicting_record_ids": group_conflicts,
            "holdout_exclusion_reasons": sorted(reasons),
            "records": statuses,
        }
    return {
        "schema_version": 1,
        "purpose": "FieldPlant disease-only whole-leaf review eligibility",
        "training_admitted": False,
        "final_holdout_established": False,
        "blocked_splits_for_conflict_groups": ["train", "development", "holdout"],
        "grouping_complete": False,
        "conflict_evidence": {
            "exact_pixel_buckets": len(conflicts),
            "source_records": len(conflict_ids),
            "box_counts_differ": sum(conflict["box_counts_differ"] for conflict in conflicts),
            "category_sets_differ": sum(conflict["category_sets_differ"] for conflict in conflicts),
            "prior_exposure_groups_including_quarantine": prior_exposure_groups,
        },
        "summary": {status: partition_summary(partitions[status]) for status in sorted(partitions)},
        "groups": groups,
    }


def resolve_conflicts():
    manifest = OUTPUT / "source-manifest.json"
    raw = manifest.read_bytes()
    if digest(raw) != SOURCE_MANIFEST_SHA:
        raise ValueError("FieldPlant source manifest SHA-256 changed")
    ledger = conflict_resolution(json.loads(raw))
    ledger["source_manifest"] = {"path": manifest.name, "sha256": SOURCE_MANIFEST_SHA}
    save(OUTPUT / "conflict-resolution.json", ledger, newline="\n")
    return 0


def main():
    rows, licenses = source_records()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw").mkdir(exist_ok=True)
    manifest = OUTPUT / "source-manifest.json"
    previous_manifest = load(manifest) if manifest.exists() else {}
    previous = {
        row["id"]: row["acquisition"] for row in previous_manifest.get("records", [])
        if row["acquisition"]["status"] == "verified"
    }
    selection = PREVIOUS_AUDIT / "selection.json"
    downloads = PREVIOUS_AUDIT / "download-results.json"
    baseline = PROGRESS / "baseline-exposure.json"
    protected = PROGRESS / "protected-splits.json"
    rights_readme = HERE / "fieldplant-metadata/coco-README.roboflow.txt"
    inputs = {path: path.read_bytes() for path in (selection, downloads, baseline, protected, rights_readme)}
    input_sha = {path.relative_to(ROOT).as_posix(): digest(raw) for path, raw in inputs.items()}
    with requests.Session() as session:
        reader = RangeReader(session, URL, ZIP_BYTES, ZIP_ETAG)
        with zipfile.ZipFile(reader) as archive:
            annotation = archive.getinfo("train/_annotations.coco.json")
            if annotation.file_size != (HERE / "fieldplant-metadata/train_annotations.coco.json").stat().st_size:
                raise ValueError("unexpected archive COCO length")
            if digest(archive.read(annotation)) != COCO_SHA:
                raise ValueError("archive COCO checksum changed")
            jobs = member_ranges(archive, rows)
        print(f"Planned {len(rows)} tomato members in {len(jobs)} ranges", flush=True)
        transfer = acquire(jobs, reader, previous)
    groups = record_groups(rows, *(json.loads(inputs[path]) for path in (selection, downloads, baseline, protected)))
    groups["annotation_conflicts_on_exact_pixels"] = annotation_conflicts(rows, groups["duplicate_buckets"]["pixels"])
    statuses = Counter(row["acquisition"]["status"] for row in rows)
    class_counts = Counter(box["source_label"] for row in rows for box in row["annotations"])
    archive_history = previous_manifest.get("archive", {})
    first_complete_transfer = archive_history.get("ranged_payload_bytes_first_complete_run")
    if first_complete_transfer is None and len(previous) == len(rows):
        first_complete_transfer = archive_history.get("ranged_payload_bytes_this_run")
    if first_complete_transfer is None and statuses["verified"] == len(rows) and not previous:
        first_complete_transfer = transfer
    output = {
        "schema_version": 1,
        "source": "FieldPlant Roboflow v11 / Kaggle COCO mirror v1",
        "purpose": "Source audit only",
        "training_admitted": False,
        "final_holdout_established": False,
        "annotation_sha256": COCO_SHA,
        "source_rights": {
            "project_url": "https://universe.roboflow.com/plant-disease-detection/fieldplant/dataset/11",
            "attribution": "Moupojou et al., FieldPlant, DOI 10.1109/ACCESS.2023.3263042",
            "licenses_in_coco": licenses,
            "kaggle_license_metadata": "Unknown",
            "mirror_equivalence": "COCO hash verified; no publisher checksum for the full archive",
        },
        "archive": {
            "url": URL,
            "bytes": ZIP_BYTES,
            "etag": ZIP_ETAG,
            "selected_member_ranges": len(jobs),
            "selected_local_record_bytes": sum(end - start for start, end, _ in jobs),
            "ranged_payload_bytes_this_run": transfer,
            "ranged_payload_bytes_first_complete_run": first_complete_transfer,
        },
        "inputs_sha256": input_sha,
        "annotation_counts": class_counts,
        "image_counts_by_label": Counter(
            label for row in rows for label in {box["source_label"] for box in row["annotations"]}
        ),
        "healthy_only_image_candidates": sum(
            all(box["target_status"] == "excluded_healthy" for box in row["annotations"]) for row in rows
        ),
        "acquisition_counts": statuses,
        "grouping": groups,
        "admission_requirements": [
            "Resolve disease scopes with source names preserved and no unsupported merges.",
            "Verify each box encloses one identifiable whole leaf or leaflet; natural occlusion or truncation is allowed.",
            "Verify every target disease leaf is annotated in each image; inspect only train/development material.",
            "Reconcile conflicting source annotations on identical pixels before admitting a representative.",
            "Verify any healthy-only candidate as a clean negative.",
            "Complete cross-source near-duplicate and scene/plant grouping before splitting.",
        ],
        "records": rows,
    }
    save(manifest, output)
    print("Acquisition counts", dict(statuses), "transferred", transfer, flush=True)
    return int(statuses["verified"] != len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve-conflicts", action="store_true", help="Write the ledger from the pinned local manifest")
    args = parser.parse_args()
    raise SystemExit(resolve_conflicts() if args.resolve_conflicts else main())
