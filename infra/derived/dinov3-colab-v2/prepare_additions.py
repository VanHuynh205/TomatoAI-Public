"""Stage train-only additions and the source-held-out SLIF Bacterial Spot pilot."""

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from PIL import Image

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2]))

from ai.prepare_dinov2_external import decoded, fingerprint
from ai.prepare_dinov2_field import (
    CLASSES,
    difference_hash,
    digest,
    hash_blocks,
    read_csv,
)

V1 = OUT.parent / "dinov3-colab-v1"
INTAKE = OUT.parent / "dinov2-next-test-v1"
INPUT = V1 / "dinov3-colab-input.zip"
HOLDOUT = V1 / "dinov3-colab-holdout.zip"
PILOT_CSV = "slif-pilot.csv"
PILOT_ZIP = "dinov3-slif-pilot.zip"
PILOT_SOURCE = "slif_tomato_v1"
HASH_FIELDS = ("sha256", "original_sha256", "normalized_sha256")


class DuplicateIndex:
    def __init__(self):
        self.exact = defaultdict(set)
        self.buckets = defaultdict(set)

    def add(self, row, image, key):
        for value in {fingerprint(image)} | {row[k] for k in HASH_FIELDS if row.get(k)}:
            self.exact[value].add(key)
        value = difference_hash(image)
        for block in hash_blocks(value):
            self.buckets[block].add((value, key))

    def matches(self, row, image, *, check_original_size=False):
        matches = set().union(
            *(
                self.exact.get(value, ())
                for value in (row["sha256"], fingerprint(image))
            )
        )
        sizes = [image.copy()] if check_original_size else []
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        sizes.append(image)
        for size in sizes:
            for view in (size, *(size.transpose(method) for method in Image.Transpose)):
                value = difference_hash(view)
                for block in hash_blocks(value):
                    matches.update(
                        key
                        for other, key in self.buckets.get(block, ())
                        if (value ^ other).bit_count() <= 4
                    )
        return matches


def checked_image(data, expected):
    if digest(data) != expected:
        raise ValueError(
            f"Unexpected image SHA-256 {digest(data)}. Expected {expected}"
        )
    return decoded(data)


def dataset_rows():
    chen = read_csv(INTAKE / "chen-test-intake.csv")
    plantseg = read_csv(INTAKE / "plantseg-test-intake.csv")
    protected_members = {
        row["archive_member"]
        for name in ("bacterial-test-reserve.csv", "five-missing-classes-findings.csv")
        for row in read_csv(INTAKE / name)
    }
    quarantined = {
        row["new_member"]
        for row in read_csv(INTAKE / "taiwan-plantseg-scene-matches.csv")
    }
    wrong_host_markers = (
        "potat",
        "solanum_tuberosum",
        "pepper",
        "eggplant",
        "soybean",
        "sunflower",
    )
    bacterial = {
        row["archive_member"]: row
        for row in read_csv(INTAKE / "plantseg-bacterial-visual.csv")
    }
    for row in bacterial.values():
        if row["contact_id"] in {"P16", "P65", "P72", "P80"}:
            quarantined.add(row["archive_member"])
    protected_urls = {
        row["photo_source_url"]
        for row in plantseg
        if row["archive_member"] in protected_members
    } - {""}
    protected_groups = {
        row["group_hint"]
        for row in chen
        if row["archive_member"] in protected_members
        or row["publisher_split"] == "test"
    }
    photo_labels = defaultdict(set)
    for row in plantseg + chen:
        group = row.get("photo_source_url") or row.get("group_hint")
        if group:
            photo_labels[row["source"], group].add(row["class_id"])
    for row in plantseg + chen:
        member = row["archive_member"]
        source = row["source"]
        plantseg_row = source == "plantseg_v7"
        group = row.get("photo_source_url") or row.get("group_hint") or member
        reserved = member in protected_members or (
            group in (protected_urls if plantseg_row else protected_groups)
        )
        domain = "mixed"
        if plantseg_row:
            eligible = row["review_status"] == "photo_and_label_review_pending"
            label = row["publisher_class"]
            marker_text = row["photo_source_url"].lower()
            if row["class_id"] == "bacterial_spot":
                context = bacterial[member]["visual_context"]
                eligible &= context in {
                    "isolated_leaf_or_controlled_background",
                    "leaf_in_situ_appears_plausible",
                }
                if context == "isolated_leaf_or_controlled_background":
                    domain = "controlled"
        else:
            marker_text = group.lower()
            eligible = (
                row["review_status"] == "photo_provenance_pending"
                and "tomato" in marker_text
            )
            label = f"Chen class {row['source_class_id']} ({row['class_id']})"
        if (
            reserved
            or member in quarantined
            or not eligible
            or len(photo_labels[source, group]) > 1
            or any(marker in marker_text for marker in wrong_host_markers)
        ):
            yield row, None
            continue
        attribution = (
            f"{row['source_url']}; archive {row['archive_license']}; resized/re-encoded"
        )
        if plantseg_row:
            attribution += f"; photo reference {row['photo_source_url']}; metadata {row['photo_license_metadata']}"
        yield row, (group, label, domain, attribution)

    for row in read_csv(INTAKE / "hossain-missing-classes-samples.csv"):
        label = {"Early Blight": "early_blight", "Leaf Mold": "leaf_mold"}.get(
            row["publisher_label"]
        )
        if label is None:
            continue
        # Both phones were used on the same date. Keep those sessions together.
        group = row["captured_at"].split(" ")[0]
        yield (
            {
                **row,
                "archive_member": row["path"],
                "archive": "",
                "class_id": label,
                "source": "hossain_v1",
            },
            (
                group,
                row["publisher_label"],
                "controlled",
                "https://data.mendeley.com/datasets/93h9p62kg4/1; CC BY 4.0; resized/re-encoded",
            ),
        )


def pilot_rows(protected, pilot_index):
    receipt = json.loads((INTAKE / "slif-bacterial-intake-receipt.json").read_bytes())
    for name in (
        "slif-raw-bacterial-index.csv",
        "slif-bacterial-photo-intake.csv",
        "slif-bacterial-group-review.csv",
    ):
        if digest((INTAKE / name).read_bytes()) != receipt["sha256"][name]:
            raise ValueError(f"SLIF intake changed: {name}")
    review = read_csv(INTAKE / "slif-bacterial-group-review.csv")
    groups = {
        stem: row["group_hint"]
        for row in review
        for stem in row["source_stems"].split("|")
    }
    photos = read_csv(INTAKE / "slif-bacterial-photo-intake.csv")
    expected = {row["group_hint"]: int(row["photo_count"]) for row in review}
    if (
        len(photos) != receipt["downloaded_files"]
        or len({row["sha256"] for row in photos}) != len(photos)
        or Counter(groups[row["group_hint"]] for row in photos) != expected
    ):
        raise ValueError("SLIF intake coverage or scene grouping changed")
    for row in photos:
        path = "images/" + Path(row["path"]).name
        data = (INTAKE / row["path"]).read_bytes()
        image = checked_image(data, row["sha256"])
        if (
            fingerprint(image) != row["normalized_sha256"]
            or difference_hash(image) != int(row["dhash128"], 16)
            or len(data) != int(row["bytes"])
            or row["class_id"] != "bacterial_spot"
        ):
            raise ValueError(f"SLIF intake metadata changed: {row['path']}")
        overlaps = protected.matches(row, image.copy(), check_original_size=True)
        if overlaps:
            raise ValueError(
                f"SLIF image overlaps an existing reference: {path}: {overlaps}"
            )
        pilot_index.add(row, image, path)
        yield (
            {
                "path": path,
                "class_id": row["class_id"],
                "group": groups[row["group_hint"]],
                "sha256": row["sha256"],
                "split": "test",
                "source": PILOT_SOURCE,
                "domain": "field",
                "original_sha256": row["sha256"],
                "normalized_sha256": row["normalized_sha256"],
                "dhash128": row["dhash128"],
                "source_file": row["archive_member"],
                "source_label": "Bacterial_Spot",
                "attribution": f"{row['source_url']}; {row['publisher_dataset_license']}",
                "label_basis": "publisher_annotation",
                "quality": "blurry" if row["group_hint"] == "t0618" else "clear",
            },
            data,
        )


def csv_bytes(records):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def main():
    rows = read_csv(V1 / "train-val.csv")
    lock_bytes = (V1 / "holdout-lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    previous = json.loads((V1 / "input-receipt.json").read_bytes())
    train_bytes = (V1 / "train-val.csv").read_bytes()
    if (
        digest(train_bytes) != previous["train_val_sha256"]
        or digest(lock_bytes) != previous["holdout_lock_sha256"]
    ):
        raise ValueError("DINOv3 v1 manifest or holdout lock changed")
    selection = list(dataset_rows())
    visual_path = OUT / "visual-exclusions.csv"
    visual_exclusions = {row["archive_member"] for row in read_csv(visual_path)}
    geometry_path = OUT / "holdout-geometry-exclusions.csv"
    geometry = defaultdict(set)
    for row in read_csv(geometry_path):
        geometry[row["group"]].add(row["holdout_path"])
    reviewed = {
        row["source_file"].split("!")[-1]
        for row in read_csv(OUT / "audit-photos/rows.csv")
    }
    if not visual_exclusions <= reviewed:
        raise ValueError("Visual exclusions contain images outside the review roster")
    protected = DuplicateIndex()
    pilot_index = DuplicateIndex()
    selected = DuplicateIndex()
    holdout_groups = set()
    audit = []
    additions = []

    with ExitStack() as stack:
        train_zip = stack.enter_context(ZipFile(INPUT))
        holdout_zip = stack.enter_context(ZipFile(HOLDOUT))
        manifests = [(train_zip, "data/", rows)]
        for split, entry in lock["manifests"].items():
            raw_manifest = holdout_zip.read(f"{split}/test.csv")
            if digest(raw_manifest) != entry["sha256"]:
                raise ValueError(f"Unexpected holdout manifest SHA-256 for {split}")
            manifest = list(
                csv.DictReader(io.StringIO(raw_manifest.decode("utf-8-sig")))
            )
            if len(manifest) != entry["rows"]:
                raise ValueError(f"Unexpected holdout count for {split}")
            holdout_groups.update((row["source"], row["group"]) for row in manifest)
            manifests.append((holdout_zip, f"{split}/", manifest))
        for archive, prefix, manifest in manifests:
            for row in manifest:
                if row["source"] == PILOT_SOURCE:
                    raise ValueError("SLIF must be held out as a source")
                image = checked_image(archive.read(prefix + row["path"]), row["sha256"])
                protected.add(row, image, prefix + row["path"])
        if not holdout_groups or holdout_zip.read("holdout-lock.json") != lock_bytes:
            raise ValueError("Holdout lock mismatch")

        raw = {
            name: stack.enter_context(ZipFile(INTAKE / name))
            for name in {row["archive"] for row, _ in selection if row["archive"]}
        }
        reserved_members = {
            row["archive_member"]
            for name in (
                "bacterial-test-reserve.csv",
                "five-missing-classes-findings.csv",
            )
            for row in read_csv(INTAKE / name)
        }
        disputed_members = {
            row["new_member"]
            for row in read_csv(INTAKE / "taiwan-plantseg-scene-matches.csv")
        }
        for row, _ in selection:
            member = row["archive_member"]
            if member in reserved_members or member in disputed_members:
                image = checked_image(raw[row["archive"]].read(member), row["sha256"])
                protected.add(row, image, member)
        pilot = list(pilot_rows(protected, pilot_index))
        candidates = []
        for row, admission in selection:
            member, archive = row["archive_member"], row["archive"]
            geometry_references = (
                geometry.get(f"{row['source']}:{admission[0]}") if admission else None
            )
            if (
                admission is None
                or member in visual_exclusions
                or member not in reviewed
                or geometry_references
            ):
                audit.append(
                    {
                        "archive_member": member,
                        "source": row["source"],
                        "class_id": row["class_id"],
                        "result": "visual_exclusion"
                        if member in visual_exclusions
                        else "holdout_geometry_overlap"
                        if geometry_references
                        else "intake_or_review_exclusion",
                        "reference": "|".join(sorted(geometry_references or ())),
                    }
                )
                continue
            data = (
                raw[archive].read(member)
                if archive
                else (INTAKE / row.get("path", member)).read_bytes()
            )
            image = checked_image(data, row["sha256"])
            candidates.append((row, admission, image.width * image.height))
            selected.add(row, image, (row["class_id"], member))

        # Keep the largest available frame from a filename or original-photo group.
        admitted_groups = set()
        admitted_members = set()
        with ZipFile(OUT / "dinov3-additions.zip", "w") as output:
            for row, (hint, label, domain, attribution), _ in sorted(
                candidates, key=lambda item: (-item[2], item[0]["archive_member"])
            ):
                member, archive = row["archive_member"], row["archive"]
                source, class_id = row["source"], row["class_id"]
                group = f"{source}:{hint}"
                if class_id not in CLASSES or (source, group) in holdout_groups:
                    raise ValueError(f"Unexpected label or held-out group for {member}")
                data = (
                    raw[archive].read(member)
                    if archive
                    else (INTAKE / row.get("path", member)).read_bytes()
                )
                image = decoded(data)
                normalized = fingerprint(image)
                overlaps = protected.matches(row, image.copy())
                pilot_overlaps = pilot_index.matches(
                    row, image.copy(), check_original_size=True
                )
                nearby = selected.matches(row, image)
                conflict = any(other != class_id for other, _ in nearby)
                duplicate = any(
                    other_member in admitted_members for _, other_member in nearby
                )
                grouped = archive and (source, hint) in admitted_groups
                reason = (
                    "locked_reference_overlap"
                    if overlaps
                    else "slif_pilot_overlap"
                    if pilot_overlaps
                    else "candidate_label_conflict"
                    if conflict
                    else "candidate_duplicate"
                    if duplicate
                    else "same_photo_group"
                    if grouped
                    else ""
                )
                audit.append(
                    {
                        "archive_member": member,
                        "source": source,
                        "class_id": class_id,
                        "result": reason or "added",
                        "reference": "|".join(sorted(overlaps | pilot_overlaps)),
                    }
                )
                if reason:
                    continue
                jpeg = io.BytesIO()
                image.save(jpeg, format="JPEG", quality=73, optimize=True)
                staged = jpeg.getvalue()
                staged_sha = digest(staged)
                staged_image = decoded(staged)
                sha = row["sha256"]
                path = f"images/{sha[:2]}/{sha}.jpg"
                if (
                    staged_sha in protected.exact
                    or fingerprint(staged_image) in protected.exact
                ):
                    audit[-1]["result"] = "locked_staged_overlap"
                    continue
                pilot_overlaps = pilot_index.matches(
                    {"sha256": staged_sha}, staged_image
                )
                if pilot_overlaps:
                    audit[-1]["result"] = "slif_pilot_staged_overlap"
                    audit[-1]["reference"] = "|".join(sorted(pilot_overlaps))
                    continue
                output.writestr(ZipInfo(f"data/{path}"), staged)
                additions.append(
                    {
                        "path": path,
                        "class_id": class_id,
                        "group": group,
                        "sha256": staged_sha,
                        "split": "train",
                        "source": source,
                        "domain": domain,
                        "original_sha256": sha,
                        "normalized_sha256": normalized,
                        "source_file": f"{archive}!{member}" if archive else member,
                        "source_id": row.get("mendeley_id")
                        or row.get("photo_source_url")
                        or member,
                        "source_label": label,
                        "attribution": attribution,
                        "label_basis": "publisher_annotation",
                        "split_block": group,
                    }
                )
                admitted_groups.add((source, hint))
                admitted_members.add(member)

    assert additions and len({row["path"] for row in rows + additions}) == len(
        rows
    ) + len(additions)
    assert all(row["split"] == "train" for row in additions)
    assert (
        not {(row["source"], row["group"]) for row in rows + additions} & holdout_groups
    )
    assert all(row["source"] != PILOT_SOURCE for row in rows + additions)
    pilot_manifest = [row for row, _ in pilot]
    pilot_csv = csv_bytes(pilot_manifest)
    with ZipFile(OUT / PILOT_ZIP, "w") as output:
        output.writestr(ZipInfo("slif/test.csv"), pilot_csv)
        for row, data in pilot:
            output.writestr(ZipInfo(f"slif/{row['path']}"), data)
    (OUT / PILOT_CSV).write_bytes(pilot_csv)
    for name, records in (
        ("train-val.csv", rows + additions),
        ("addition-audit.csv", audit),
    ):
        (OUT / name).write_bytes(csv_bytes(records))
    report = {
        "reference_rows": len(rows)
        + sum(item["rows"] for item in lock["manifests"].values()),
        "holdout_lock_sha256": digest(lock_bytes),
        "train_val_sha256": digest((OUT / "train-val.csv").read_bytes()),
        "addition_zip_sha256": digest((OUT / "dinov3-additions.zip").read_bytes()),
        "pilot": {
            "source": PILOT_SOURCE,
            "domain": "field",
            "manifest_sha256": digest(pilot_csv),
            "zip_sha256": digest((OUT / PILOT_ZIP).read_bytes()),
            "rows": len(pilot_manifest),
            "clear_rows": sum(row["quality"] == "clear" for row in pilot_manifest),
            "provisional_scene_groups": dict(
                Counter(row["group"] for row in pilot_manifest)
            ),
        },
        "visual_exclusions_sha256": digest(visual_path.read_bytes()),
        "holdout_geometry_exclusions_sha256": digest(geometry_path.read_bytes()),
        "split_count": dict(Counter(row["split"] for row in rows + additions)),
        "additions": {
            source: dict(
                Counter(row["class_id"] for row in additions if row["source"] == source)
            )
            for source in sorted({row["source"] for row in additions})
        },
        "screen_results": dict(Counter(row["result"] for row in audit)),
    }
    (OUT / "addition-receipt.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
