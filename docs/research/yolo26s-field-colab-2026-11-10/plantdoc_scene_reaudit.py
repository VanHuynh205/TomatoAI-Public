"""Reserve original PlantDoc TRAIN scenes for detector review, then render source boxes."""

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent.name
sys.path.insert(0, str(ROOT))
admission = import_module(f"docs.research.{STUDY}.plantdoc_leaf_admission")
guard = import_module(f"docs.progress.{STUDY}.candidate_split_guard")
master_module = guard.master_module
audit = import_module(f"docs.progress.{STUDY}.audit_plantdoc_detection")

OUTPUT = ROOT / "infra/derived/yolo26s-plantdoc-scene-reaudit-v1"
SOURCE_CSV = admission.PROGRESS / "plantdoc-source-train-labels.csv"
FOLDER_MANIFEST = admission.REVIEW / "manifest.csv"
PROTECTED = admission.PROGRESS / "protected-splits.json"
V1 = admission.OUTPUT / "manifest.json"
V2 = "infra/derived/yolo26s-plantdoc-disease-only-v2/manifest.json"
SICHUAN = "infra/derived/yolo26s-sichuan-leaf-subset-v1/admission-disease-only-v3.json"
GUARD_RECEIPT = admission.PROGRESS / "candidate-split-guard-v1.json"
PUBLISHER_BLOBS = admission.OUTPUT / "publisher-train-blobs.json"
TREE_SHA256 = "2ec999ce34064b354a69e799568fa66b504c4f43e95acb849f823e7cc4d7bf66"
MIXED_IDS = {"PD0088", "PD0120", "PD0629"}
MIXED_REASON = "mixed_box_labels_outside_approved_image_mapping"
NON_TARGET = "Tomato leaf"
COORDS = ("xmin", "ymin", "xmax", "ymax")
CONFLICT_ID = "PD0539"
CONFLICT_XML = "TRAIN/d-to-tylv-fo005-14496A31BE70553130A.xml"
UC_IPM_REFERENCE = "https://ipm.ucanr.edu/PMG/T/D-TO-TYLV-FO.005.html"


def fetch(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def verify_blob(raw, blob, path):
    # Git hashes the header and raw bytes together.
    header = f"blob {len(raw)}\0".encode("ascii")
    if len(raw) != blob["size"] or hashlib.sha1(header + raw).hexdigest() != blob["blob_sha1"]:
        raise ValueError(f"Publisher Git blob mismatch: {path}")


def source_boxes(labels):
    size = (int(labels[0][1]["width"]), int(labels[0][1]["height"]))
    boxes = [
        {
            **audit.box_record(row, size),
            "source_box_id": ordinal,
            "disease_target_class": None if row["class"] == NON_TARGET else audit.NAMES[row["class"]],
        }
        for ordinal, row in labels
    ]
    return size, boxes


def build_selection():
    inputs = guard.load_inputs()
    master = inputs[guard.MASTER]
    index = guard.record_index(master)
    quarantine = guard.training_quarantine(master, inputs[guard.BRYAN])
    pins = {
        SOURCE_CSV: audit.CSV_SHA256,
        FOLDER_MANIFEST: admission.INPUTS[FOLDER_MANIFEST],
        PROTECTED: audit.PROTECTED_SHA256,
        V1: admission.V1_MANIFEST_SHA256,
        PUBLISHER_BLOBS: admission.PUBLISHER_BLOBS_SHA256,
        GUARD_RECEIPT: "c3ac249aef5045b8b3de54227dabb178f1b8d42ba642340b5560de00380913be",
    }
    for path, expected in pins.items():
        if admission.sha256(path) != expected:
            raise ValueError(f"Sealed metadata changed: {path}")
    receipt = json.loads(GUARD_RECEIPT.read_bytes())
    if (
        receipt["unresolved_bryan"] != quarantine
        or receipt["script_sha256"] != admission.sha256(Path(guard.__file__))
    ):
        raise ValueError("Candidate guard no longer reproduces its sealed quarantine")
    protection = json.loads(PROTECTED.read_bytes())
    legacy = {row["id"]: row for row in json.loads(V1.read_bytes())["records"]}
    v2 = {row["id"]: row for row in inputs[V2]["records"]}
    folder_rows = admission.table(FOLDER_MANIFEST)
    source_index = defaultdict(list)
    csv_rows = admission.table(SOURCE_CSV)
    for ordinal, row in enumerate(csv_rows, start=1):
        source_index[audit.key(row["filename"])].append((ordinal, row))
    scenes = {
        row["id"] for row in folder_rows
        if row["source_split"] == "train" and row["reason"] == "no_dominant_leaf"
    }
    mixed = {
        row["id"] for row in legacy.values()
        if row["split"] == "train" and row["technical_exclusions"] == [MIXED_REASON]
    }
    ids = scenes | mixed
    members = {row["id"]: row for row in folder_rows if row["id"] in ids}
    if (
        mixed != MIXED_IDS or len(scenes) != 10 or len(ids) != 13 or members.keys() != ids
        or scenes & legacy.keys() or any(v2[id_]["decision"] != "excluded" for id_ in mixed)
    ):
        raise ValueError("Supplemental PlantDoc selection membership changed")
    blocked = set(master["blocked_from_training"]) | set(quarantine["quarantined_from_training"])
    protected_paths = {
        (ROOT / row["image_path"]).resolve() for row in protection["plantdoc_protected_records"]
    }
    seeds = {f"plantdoc:{id_}" for id_ in ids}
    relatives = set(master_module.exclusion_sets(
        index, master["explicit_relations"], set(), seeds,
    )["excluded_from_final_holdout"])
    local_pairs = [
        (row["left"], row["right"])
        for relative in ("plantdoc-folder-review-v2/review-links.csv", "plantdoc-clean-v1/related-pairs.csv")
        for row in admission.table(ROOT / "infra/derived" / relative)
    ]
    local_closure = master_module.protect.closure(folder_rows, ids, local_pairs)
    if not {f"plantdoc:{id_}" for id_ in local_closure} <= relatives:
        raise ValueError("Master closure missed a known PlantDoc relation")
    if relatives & blocked or local_closure & set(protection["plantdoc_train_protected_ids"]):
        raise ValueError("Review selection reaches a protected relation")
    blocked_holdout = set(master["excluded_from_final_holdout"]) | relatives
    current = [
        {"id": f"plantdoc:{row['id']}", "sha256": row["source_sha256"], "split": row["split"]}
        for row in v2.values() if row["decision"] == "accepted"
    ]
    current.extend(
        {"id": row["id"], "sha256": row["image_sha256"], "split": "train"}
        for row in inputs[SICHUAN]["records"]
    )
    candidates = [
        {"id": f"plantdoc:{id_}", "sha256": members[id_]["sha256"], "split": "train"}
        for id_ in sorted(ids)
    ]
    counts = guard.validate_records(master, quarantine, current + candidates)
    guard.validate_records(
        {**master, "excluded_from_final_holdout": sorted(blocked_holdout)}, quarantine, current,
    )

    raw_tree = fetch(admission.TREE)
    if hashlib.sha256(raw_tree).hexdigest() != TREE_SHA256:
        raise ValueError("Unexpected publisher Git tree response")
    tree = json.loads(raw_tree)
    if tree["truncated"] or tree["sha"] != audit.COMMIT:
        raise ValueError("Unexpected publisher Git tree commit or truncation")
    blobs = {
        row["path"]: {"blob_sha1": row["sha"], "size": row["size"]}
        for row in tree["tree"] if row["type"] == "blob"
    }
    verify_blob(SOURCE_CSV.read_bytes(), blobs["train_labels.csv"], SOURCE_CSV)
    old_blobs = json.loads(PUBLISHER_BLOBS.read_bytes())["train_files"]
    records, ordinals = [], set()
    for id_, member in sorted(members.items()):
        canonical = f"plantdoc:{id_}"
        path = audit.validate_member(member, "train", protection).resolve()
        if (
            path in protected_paths or index[canonical]["sha256"] != member["sha256"]
            or ("pixel", member["pixel_sha256"]) not in index[canonical]["keys"]
        ):
            raise ValueError(f"Invalid or protected source membership: {id_}")
        labels = source_index[audit.key(path.name)]
        filenames = {row["filename"] for _, row in labels}
        source_ids = {ordinal for ordinal, _ in labels}
        if len(filenames) != 1 or ordinals & source_ids:
            raise ValueError(f"Ambiguous publisher CSV join: {id_}")
        ordinals.update(source_ids)
        publisher_filename = next(iter(filenames))
        publisher_path = f"TRAIN/{publisher_filename}"
        blob = blobs[publisher_path]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != member["sha256"]:
            raise ValueError(f"Local source bytes changed: {id_}")
        verify_blob(raw, blob, publisher_path)
        size, boxes = source_boxes(labels)
        if id_ in mixed:
            original = legacy[id_]
            if (
                blob != old_blobs[publisher_filename]
                or original["source_image"] != member["image_path"]
                or original["source_sha256"] != member["sha256"]
                or original["source_pixel_sha256"] != member["pixel_sha256"]
                or [audit.box_record(row, size) for _, row in labels] != original["boxes"]
            ):
                raise ValueError(f"Sealed mixed-image evidence changed: {id_}")
        records.append({
            "id": canonical,
            "role": "visual_review_only",
            "review_status": "source_label_conflict" if id_ == CONFLICT_ID else "pending",
            "training_ready": False,
            "final_holdout_eligible": False,
            "selection_reason": MIXED_REASON if id_ in mixed else member["reason"],
            "source_split": member["source_split"],
            "source_folder_label": member["source_label"],
            "source_image": member["image_path"],
            "source_sha256": member["sha256"],
            "source_pixel_sha256": member["pixel_sha256"],
            "publisher_path": publisher_path,
            "publisher_blob": blob,
            "width": size[0],
            "height": size[1],
            "boxes": boxes,
        })

    conflict = next(row for row in records if row["id"] == f"plantdoc:{CONFLICT_ID}")
    xml_url = audit.SOURCE.replace("train_labels.csv", CONFLICT_XML)
    raw_xml = fetch(xml_url)
    verify_blob(raw_xml, blobs[CONFLICT_XML], CONFLICT_XML)
    xml = ElementTree.fromstring(raw_xml)
    xml_boxes = Counter(
        (obj.findtext("name"), tuple(float(obj.findtext(f"bndbox/{field}")) for field in COORDS))
        for obj in xml.findall("object")
    )
    if (
        xml.findtext("filename") != Path(conflict["publisher_path"]).name
        or xml_boxes != Counter((box["source_class"], tuple(box["xyxy"])) for box in conflict["boxes"])
    ):
        raise ValueError("PD0539 XML and CSV disagree")
    conflict["source_label_conflict"] = {
        "publisher_xml_url": xml_url,
        "publisher_xml_blob": blobs[CONFLICT_XML],
        "publisher_xml_sha256": hashlib.sha256(raw_xml).hexdigest(),
        "publisher_xml_matches_csv": True,
        "reference_url": UC_IPM_REFERENCE,
        "reference_image_code": "D-TO-TYLV-FO.005",
        "reference_caption_diagnosis": "Tomato yellow leaf curl virus",
        "reference_identity_verified": False,
        "disposition": "quarantine_source_class_pending_identity_and_label_verification",
    }
    disease = Counter(
        box["disease_target_class"] for row in records for box in row["boxes"]
        if box["disease_target_class"] is not None
    )
    if sum(disease.values()) != 100 or len(ordinals) != 112 or not all(
        any(box["disease_target_class"] for box in row["boxes"]) for row in records
    ):
        raise ValueError("Supplemental source box membership changed")
    return {
        "schema_version": 1,
        "status": "review_reserved_training_pending",
        "training_ready": False,
        "final_holdout_established": False,
        "publisher_commit": audit.COMMIT,
        "publisher_tree_url": admission.TREE,
        "publisher_tree_sha256": TREE_SHA256,
        "publisher_csv_url": audit.SOURCE,
        "publisher_csv_blob": blobs["train_labels.csv"],
        "inputs_sha256": {
            **guard.PINS,
            **{path.relative_to(ROOT).as_posix(): digest for path, digest in pins.items()},
            Path(__file__).relative_to(ROOT).as_posix(): admission.sha256(Path(__file__)),
        },
        "guard_prospective_counts_not_admission": counts,
        "protected_training_count": len(blocked),
        "final_holdout_excluded_ids": sorted(relatives),
        "added_final_holdout_excluded_ids": sorted(relatives - set(master["excluded_from_final_holdout"])),
        "source_box_id": "1-based data-row ordinal in the publisher CSV, excluding its header",
        "source_box_count": len(ordinals),
        "source_disease_counts": dict(sorted(disease.items())),
        "source_non_target_count": len(ordinals) - sum(disease.values()),
        "records": records,
    }


def render_overlays(selection, selection_sha256):
    verified = []
    for row in selection["records"]:
        path = (ROOT / row["source_image"]).resolve()
        if not path.is_relative_to((ROOT / "infra/plant_doc/train").resolve()):
            raise ValueError(f"Attempted pixel access outside PlantDoc TRAIN: {row['id']}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Source changed after reservation: {row['id']}")
        pixel, error = admission._pixel_hash_stream(io.BytesIO(raw))
        if error or pixel != row["source_pixel_sha256"]:
            raise ValueError(f"Pixel verification failed: {row['id']}")
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (row["width"], row["height"]) or image.getexif().get(274, 1) != 1:
                raise ValueError(f"Source geometry requires separate review: {row['id']}")
        verified.append((row, raw))

    overlays = {}
    for row, raw in verified:
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in row["boxes"]:
            color = "#ff4267" if box["disease_target_class"] else "#eeeeee"
            draw.rectangle(box["xyxy"], outline=color, width=2)
        for box in row["boxes"]:
            x, y, _, _ = box["xyxy"]
            draw.text((x + 2, y + 2), str(box["source_box_id"]), fill="white", stroke_width=1, stroke_fill="black")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data = stream.getvalue()
        path = OUTPUT / f"{row['id'].removeprefix('plantdoc:')}.png"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Overlay evidence changed: {path}")
        else:
            with path.open("xb") as output:
                output.write(data)
        overlays[path.name] = hashlib.sha256(data).hexdigest()

    return master_module.write_immutable(OUTPUT / "overlays.json", {
        "selection_sha256": selection_sha256,
        "status": "source_pixels_verified_visual_review_pending",
        "training_ready": False,
        "pixel_verified_ids": [row["id"] for row, _ in verified],
        "overlay_sha256": overlays,
    })


def main():
    selection = build_selection()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    digest = master_module.write_immutable(OUTPUT / "selection.json", selection)
    overlay_digest = render_overlays(selection, digest)
    print(json.dumps({
        "records": len(selection["records"]),
        "known_holdout_exclusions": len(selection["final_holdout_excluded_ids"]),
        "source_boxes": selection["source_box_count"],
        "selection_sha256": digest,
        "overlays_sha256": overlay_digest,
    }))


if __name__ == "__main__":
    main()
