"""Stage the locked, single-label field classifier dataset without modifying source data."""

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError

from ai.dinov3_field import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "infra/derived"
RAW = DERIVED / "dinov3-field-v1/raw"
OUTPUT = DERIVED / "dinov2-field-v1"
IMTIAZ_AUDIT = DERIVED / "yolo26s-imtiaz-official-export-v1"
NEGATIVE = CLASSES[-1]
SPLITS = ("test", "val", "train")
RACHIN_LABELS = {
    "Bacteiral Spot": "bacterial_spot",
    "Early Blight": "early_blight",
    "Late Blight": "late_blight",
    "Healthy": "healthy",
    "Curl": "yellow_leaf_curl",
}
SOURCES = {
    "rachin": "https://data.mendeley.com/datasets/jttrv2w27r/2",
    "ghana_ccmt": "https://data.mendeley.com/datasets/bwh3zbpkpv/1",
    "imtiaz": "https://data.mendeley.com/datasets/bpfd9cns5g/2",
    "plantdoc": "https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8",
    "plantvillage": "https://github.com/spMohanty/PlantVillage-Dataset/tree/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a",
    "openimages_hands": "https://storage.googleapis.com/openimages/web/download_v7.html",
    "soilclass": "https://data.mendeley.com/datasets/6mx34nshm4/1",
    "plantdoc_other_leaves": "https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8",
    "plantcity": "https://data.mendeley.com/datasets/w8kh2xkspx/2",
}


@dataclass(frozen=True)
class Sample:
    source: str
    domain: str
    split: str
    class_id: str
    group: str
    path: str
    archive: str = ""
    expected_sha256: str = ""
    source_id: str = ""
    source_label: str = ""
    attribution: str = ""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def assigned_split(group: str) -> str:
    bucket = int(digest(group.encode())[:8], 16) % 100
    return "test" if bucket < 20 else "val" if bucket < 35 else "train"


def locked_samples() -> Iterator[Sample]:
    pv_root = Path("infra/tomato_plantvillage/TomatoAI/data/raw/plantvillage")
    pv = read_csv(DERIVED / "pv8-v1/manifest.csv")
    # Round-robin across leaf groups to keep the compact upload representative.
    families: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pv:
        families[row["group_id"]].append(row)
    ranked = sorted(
        (offset, digest(row["source_path"].encode()), row)
        for family in families.values()
        for offset, row in enumerate(family)
    )
    counts: Counter[tuple[str, str]] = Counter()
    for _, _, row in ranked:
        key = row["split"], row["class_id"]
        if counts[key] >= (350 if row["split"] == "train" else 75):
            continue
        counts[key] += 1
        yield Sample(
            "plantvillage",
            "controlled",
            *key,
            row["group_id"],
            (pv_root / row["source_path"]).as_posix(),
            expected_sha256=row["source_sha256"],
            source_id=row["sample_id"],
            source_label=row["label_original"],
        )
    for split in SPLITS:
        name = "reserved_test" if split == "test" else split
        for row in read_csv(DERIVED / f"plantdoc-split-v1/{name}.csv"):
            yield Sample(
                "plantdoc",
                "web",
                split,
                row["class_id"],
                row["component_id"],
                row["image_path"],
                expected_sha256=row["sha256"],
                source_id=row["id"],
                source_label=row["source_label"],
            )


def imtiaz_samples(
    archive_name: str, archive: ZipFile, omitted: Counter[str]
) -> Iterator[Sample]:
    metadata = read_json(IMTIAZ_AUDIT / "mendeley-annotated-metadata.json")
    audit = read_json(IMTIAZ_AUDIT / "metadata-audit.json")
    if (
        digest((IMTIAZ_AUDIT / "mendeley-annotated-metadata.json").read_bytes())
        != audit["mendeley_metadata_sha256"]
    ):
        raise ValueError("Imtiaz metadata differs from verified export audit")
    labels = dict(
        zip(
            audit["source_names"],
            ("bacterial_spot", "early_blight", "healthy", "late_blight", "leaf_mold"),
        )
    )
    records = {
        f"{folder['source_path']}/{record['filename']}": record
        for folder in metadata["folders"]
        for record in folder["files"]
    }
    families: defaultdict[str, list[Sample]] = defaultdict(list)
    for path in archive.namelist():
        _, _, relative = path.partition("/TomatoLeafMulticlass (Annotated)/")
        if "/images/" not in relative:
            continue
        annotation = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        raw_labels = archive.read(annotation)
        record = records[annotation.split("/", 1)[1]]
        if digest(raw_labels) != record["sha256"]:
            raise ValueError(f"Imtiaz label differs from verified export: {annotation}")
        ids = {int(line.split()[0]) for line in raw_labels.splitlines() if line.strip()}
        label = audit["source_names"][next(iter(ids))] if len(ids) == 1 else ""
        split = {"train": "train", "valid": "val", "test": "test"}[
            relative.split("/")[0]
        ]
        group = PurePosixPath(path).name.split(".rf.", 1)[0]
        record = records[path.split("/", 1)[1]]
        families[group].append(
            Sample(
                "imtiaz",
                "field",
                split,
                labels.get(label, ""),
                group,
                path,
                archive_name,
                record["sha256"],
                record["id"],
                label,
            )
        )
    for family in families.values():
        labels_and_splits = {(sample.class_id, sample.split) for sample in family}
        if len(labels_and_splits) == 1 and family[0].class_id:
            yield from family
        else:
            omitted["imtiaz: mixed, empty, unsupported, or cross-split family"] += len(
                family
            )


def archive_samples(
    archives: dict[str, ZipFile], omitted: Counter[str]
) -> Iterator[Sample]:
    name = "rachin-v2-verified.zip"
    rachin = archives[name]
    records = json.loads(rachin.read("publisher-records.json"))
    if {r["filename"] for r in records} | {"publisher-records.json"} != set(
        rachin.namelist()
    ):
        raise ValueError("Rachin files differ from publisher index")
    folder_labels: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        path = record["filename"]
        label = PurePosixPath(path).stem.rsplit("_", 1)[0]
        folder_labels[record["folder_id"]].add(label)
        if rachin.getinfo(path).file_size != record["size"]:
            raise ValueError(f"Rachin member size mismatch: {path}")
        yield Sample(
            "rachin",
            "field",
            "test",
            RACHIN_LABELS[label],
            path,
            path,
            name,
            record["sha256"],
            record["id"],
            label,
        )
    if len(folder_labels) != len(RACHIN_LABELS) or any(
        len(labels) != 1 for labels in folder_labels.values()
    ):
        raise ValueError("Rachin publisher folders do not match filename labels")

    name = "ghana-raw-tomato-v1.zip"
    groups: defaultdict[tuple[str, str], list[tuple[int, str, str, str]]] = defaultdict(
        list
    )
    for path in archives[name].namelist():
        file = PurePosixPath(path)
        label = file.parent.name
        class_id = {"healthy": "healthy", "septoria leaf spot": "septoria_leaf_spot"}[
            label
        ]
        match = re.match(r".*?(\d+)_", file.stem)
        if match is None:
            raise ValueError(f"Unexpected Ghana filename: {path}")
        number = int(match[1])
        group = f"{label}:{number // 50:03d}"
        groups[class_id, assigned_split(group)].append((number, path, group, label))
    for (class_id, split), members in groups.items():
        cap = {"train": 650, "val": 200, "test": 320}[split]
        # A suffixed copy shares a stem with its original and adds no capture.
        by_number: dict[int, tuple[str, str, str]] = {}
        for number, path, group, label in sorted(members):
            by_number.setdefault(number, (path, group, label))
        picked = sorted(by_number.values(), key=lambda item: digest(item[0].encode()))[
            :cap
        ]
        omitted["ghana_ccmt: repeated numeric filename or compact upload cap"] += len(
            members
        ) - len(picked)
        for path, group, label in picked:
            yield Sample(
                "ghana_ccmt",
                "field",
                split,
                class_id,
                group,
                path,
                name,
                source_label=label,
            )

    name = "imtiaz-v2.zip"
    yield from imtiaz_samples(name, archives[name], omitted)

    name = "openimages-hands.zip"
    for record in json.loads(archives[name].read("attribution.json")):
        image_id = record["id"]
        yield Sample(
            "openimages_hands",
            "web",
            assigned_split(image_id),
            NEGATIVE,
            image_id,
            f"{image_id}.jpg",
            name,
            source_id=image_id,
            source_label="Human hand",
            attribution=json.dumps(record, sort_keys=True),
        )

    name = "soilclass-v1.zip"
    for path in archives[name].namelist():
        if path.endswith("/"):
            continue
        file = PurePosixPath(path)
        timestamp = re.search(r"(20\d{6})_?(\d{6})", file.stem)
        if timestamp is None:
            raise ValueError(f"Missing SoilClass timestamp: {path}")
        date, time = timestamp.groups()
        minutes = int(time[:2]) * 60 + int(time[2:4])
        group = f"{date}:{minutes // 2:03d}"
        yield Sample(
            "soilclass",
            "soil_capture",
            assigned_split(group),
            NEGATIVE,
            group,
            path,
            name,
            source_label=file.parent.name,
        )

    name = "plantdoc-other-leaves.zip"
    for record in json.loads(archives[name].read("publisher-records.json")):
        path = record["path"]
        file = PurePosixPath(path)
        group = file.stem.casefold()
        split = "test" if file.parts[0] == "test" else assigned_split(group)
        if file.parts[0] == "train" and split == "test":
            split = "train"
        yield Sample(
            "plantdoc_other_leaves",
            "web",
            split,
            NEGATIVE,
            group,
            path,
            name,
            source_id=record["sha"],
            source_label=file.parent.name,
        )


def difference_hash(image: Image.Image) -> int:
    horizontal = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    vertical = image.convert("L").resize((8, 9), Image.Resampling.LANCZOS).tobytes()
    bits = [horizontal[i] > horizontal[i + 1] for i in range(72) if i % 9 != 8]
    bits.extend(vertical[i] > vertical[i + 8] for i in range(64))
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hash_blocks(value: int) -> list[tuple[int, int]]:
    # Eight disjoint 16-bit blocks guarantee a shared block for distances at most four.
    return [(offset, (value >> offset) & 0xFFFF) for offset in range(0, 128, 16)]


def stage(
    samples: list[Sample],
    archives: dict[str, ZipFile],
    target: Path,
    omitted: Counter[str],
) -> list[dict[str, str]]:
    rows = []
    exact: set[str] = set()
    buckets: defaultdict[tuple[int, int], list[tuple[int, str, str]]] = defaultdict(
        list
    )
    group_splits: dict[str, str] = {}
    source_priority = {source: index for index, source in enumerate(SOURCES)}
    samples.sort(
        key=lambda s: (SPLITS.index(s.split), source_priority[s.source], s.path)
    )
    for index, sample in enumerate(samples, 1):
        group = f"{sample.source}:{sample.group}"
        if group in group_splits and group_splits[group] != sample.split:
            omitted[
                f"{sample.source}: group already assigned to a higher-priority split"
            ] += 1
            continue
        data = (
            archives[sample.archive].read(sample.path)
            if sample.archive
            else (ROOT / sample.path).read_bytes()
        )
        source_hash = digest(data)
        if sample.expected_sha256 and source_hash != sample.expected_sha256:
            raise ValueError(f"Source integrity mismatch: {sample.path}")
        if sample.source == "plantdoc_other_leaves":
            git_blob = f"blob {len(data)}\0".encode() + data
            if (
                hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()
                != sample.source_id
            ):
                raise ValueError(f"PlantDoc Git blob mismatch: {sample.path}")
        try:
            with Image.open(io.BytesIO(data)) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            omitted[f"{sample.source}: unreadable image"] += 1
            continue
        normalized = digest(
            f"{image.width}x{image.height}\0".encode() + image.tobytes()
        )
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        perceptual = difference_hash(image)
        blocks = hash_blocks(perceptual)
        near = any(
            (sample.split != split or sample.source != source)
            and (perceptual ^ earlier).bit_count() <= 4
            for block in blocks
            for earlier, split, source in buckets[block]
        )
        staged = io.BytesIO()
        image.save(staged, format="JPEG", quality=73, optimize=True)
        jpeg = staged.getvalue()
        staged_hash = digest(jpeg)
        hashes = {source_hash, normalized, staged_hash}
        if hashes & exact or near:
            reason = (
                "exact or normalized duplicate"
                if hashes & exact
                else "perceptual duplicate"
            )
            omitted[f"{sample.source}: {reason}"] += 1
            continue
        exact.update(hashes)
        for block in blocks:
            buckets[block].append((perceptual, sample.split, sample.source))
        group_splits[group] = sample.split
        relative = f"images/{source_hash[:2]}/{source_hash}.jpg"
        output = target / relative
        output.parent.mkdir(exist_ok=True, parents=True)
        output.write_bytes(jpeg)
        source_file = (
            f"{sample.archive}!{sample.path}" if sample.archive else sample.path
        )
        rows.append(
            {
                "path": relative,
                "class_id": sample.class_id,
                "group": group,
                "sha256": staged_hash,
                "split": sample.split,
                "source": sample.source,
                "domain": sample.domain,
                "original_sha256": source_hash,
                "normalized_sha256": normalized,
                "source_file": source_file,
                "source_id": sample.source_id,
                "source_label": sample.source_label,
                "attribution": sample.attribution,
            }
        )
        if index % 1000 == 0:
            print(
                f"Checked {index:,}/{len(samples):,} candidates, staged {len(rows):,}",
                flush=True,
            )
    return rows


def verify(rows: list[dict[str, str]]) -> None:
    for split in SPLITS:
        assert {row["class_id"] for row in rows if row["split"] == split} == set(
            CLASSES
        ), split
    for key in ("path", "sha256", "original_sha256", "normalized_sha256"):
        assert len({row[key] for row in rows}) == len(rows), key
    for split in SPLITS:
        groups = {row["group"] for row in rows if row["split"] == split}
        assert groups.isdisjoint(
            row["group"] for row in rows if row["split"] != split
        ), split
    assert {row["split"] for row in rows if row["source"] == "rachin"} == {"test"}
    assert {row["source"] for row in rows if row["class_id"] == NEGATIVE} == {
        "openimages_hands",
        "soilclass",
        "plantdoc_other_leaves",
    }
    for source in ("ghana_ccmt", "imtiaz"):
        assert {row["split"] for row in rows if row["source"] == source} == set(
            SPLITS
        ), source


def plantcity_samples(archive: ZipFile, name: str) -> Iterator[Sample]:
    allowed = {
        "bacterial_spot",
        "early_blight",
        "late_blight",
        "leaf_mold",
        "septoria_leaf_spot",
        "healthy",
    }
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for member in archive.namelist():
        path = PurePosixPath(member)
        if path.suffix.lower() != ".jpg":
            continue
        match = re.fullmatch(r"(tomato_)?(\d+)", path.stem)
        label = path.parent.name
        if label not in allowed or match is None:
            raise ValueError(f"Unexpected PlantCity image: {member}")
        family = match[1] or "number"
        group = f"{label}:{family}:{(int(match[2]) - 1) // 30:03d}"
        groups[label, group].append(member)
    for label in sorted(allowed):
        ranked = sorted(group for category, group in groups if category == label)
        ranked.sort(key=lambda group: digest(group.encode()))
        if len(ranked) < 4:
            raise ValueError(f"Not enough PlantCity filename blocks: {label}")
        validation = set(ranked[: max(2, round(len(ranked) * 0.18))])
        for group in ranked:
            for member in groups[label, group]:
                yield Sample(
                    "plantcity",
                    "field",
                    "val" if group in validation else "train",
                    label,
                    group,
                    member,
                    name,
                    source_label=label,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantcity", type=Path)
    args = parser.parse_args()
    output = OUTPUT if args.plantcity is None else DERIVED / "dinov2-field-v2"
    zip_name = "dataset.zip" if args.plantcity is None else "dinov2-field-v2.zip"
    if output.exists():
        raise FileExistsError(output)
    omitted: Counter[str] = Counter()
    target = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    print(f"Staging in {target}", flush=True)
    archive_names: tuple[str, ...] = (
        "rachin-v2-verified.zip",
        "ghana-raw-tomato-v1.zip",
        "imtiaz-v2.zip",
        "openimages-hands.zip",
        "soilclass-v1.zip",
        "plantdoc-other-leaves.zip",
    )
    with ExitStack() as stack:
        archives = {
            name: stack.enter_context(ZipFile(RAW / name)) for name in archive_names
        }
        samples = [*locked_samples(), *archive_samples(archives, omitted)]
        if args.plantcity:
            name = args.plantcity.name
            archives[name] = stack.enter_context(ZipFile(args.plantcity))
            samples.extend(plantcity_samples(archives[name], name))
            archive_names += (name,)
        print(f"Staging {len(samples):,} candidates", flush=True)
        rows = stage(samples, archives, target, omitted)
    verify(rows)
    for name, splits in (("train-val.csv", {"train", "val"}), ("test.csv", {"test"})):
        with (target / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] in splits)
    counts = {
        split: dict(
            sorted(
                Counter(row["source"] for row in rows if row["split"] == split).items()
            )
        )
        for split in SPLITS
    }
    class_counts = {
        split: dict(
            sorted(
                Counter(
                    row["class_id"] for row in rows if row["split"] == split
                ).items()
            )
        )
        for split in SPLITS
    }
    receipt = {
        "class_ids": CLASSES,
        "counts_by_source": counts,
        "counts_by_class": class_counts,
        "omitted": dict(sorted(omitted.items())),
        "source_urls": {
            name: SOURCES[name]
            for name in sorted(set().union(*(set(c) for c in counts.values())))
        },
        "archives": archive_names,
        "preprocessing": "EXIF orientation, RGB, longest edge at most 384, JPEG quality 73",
        "selection": [
            "PlantVillage leaf groups and PlantDoc component splits follow locked manifests. PlantVillage cap per class is 350 train, 75 val and 75 test.",
            "Rachin is test-only. Its publisher file IDs, sizes and SHA-256 were verified. Curl is mapped from the publisher's Yellow Leaf Curl Virus folder.",
            "Ghana groups contain 50 consecutive filename numbers per class. Suffixed copies use the original number. Caps are 650 train, 200 val and 320 test per class.",
            "Imtiaz uses the annotated export. Families use filename before .rf.; all variants must have only boxes of the same admitted class and share the official split.",
            "SoilClass groups all classes by two-minute capture block. Open Images uses source image ID; PlantDoc negatives use publisher paths and filename stems.",
            "Negative labels are weak source labels. Open Images only confirms that a hand appears; individual full frames have not been audited for tomato leaves.",
            "Pakistan and Netrakona archives were excluded because of provenance overlap and staged white-paper captures, respectively.",
            *(
                [
                    (
                        "PlantCity uses six unambiguous leaf classes from the original archive, never the pre-augmented archive. "
                        "Per-class filename sequence blocks of 30 are assigned to train or validation; tree and farm IDs are unavailable."
                    )
                ]
                if args.plantcity
                else []
            ),
        ],
        "duplicate_policy": [
            "Test wins over validation, and validation over training. Within each split Rachin wins before other sources.",
            "Exact source bytes, full decoded oriented pixels and staged JPEG bytes are globally unique.",
            "A combined horizontal/vertical 128-bit dHash screens across splits and sources at Hamming distance <= 4.",
            "The dHash can miss crops, rotations and repeat captures. Ghana filename blocks and SoilClass time blocks approximate capture groups. Independence by plant or farm is unverified.",
            "Imtiaz frame labels are inferred from source boxes. Unannotated leaves or symptoms can remain in a frame.",
        ],
    }
    (target / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    if (
        args.plantcity
        and (target / "test.csv").read_bytes() != (OUTPUT / "test.csv").read_bytes()
    ):
        raise ValueError("Adding PlantCity changed the sealed test manifest")
    with ZipFile(target / zip_name, "x", compression=ZIP_STORED) as archive:
        for name in (
            "train-val.csv",
            "test.csv",
            "receipt.json",
            *(row["path"] for row in rows),
        ):
            archive.write(target / name, name)
    with ZipFile(target / zip_name) as archive:
        assert len(archive.namelist()) == len(rows) + 3
        assert archive.testzip() is None
    target.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "zip_bytes": (output / zip_name).stat().st_size,
                "rows": len(rows),
                "counts_by_source": counts,
                "counts_by_class": class_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
