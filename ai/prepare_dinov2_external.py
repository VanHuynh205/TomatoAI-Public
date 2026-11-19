"""Acquire and freeze independent, non-augmented tomato leaf evaluation images."""

import argparse
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from csv import DictWriter
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import UUID
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from PIL import Image, ImageOps

from ai.prepare_dinov2_field import (
    DERIVED,
    difference_hash,
    digest,
    hash_blocks,
    read_csv,
)

OUTPUT = DERIVED / "dinov2-external-v1"
BASELINE = DERIVED / "dinov2-field-v2"
BASELINE_MANIFESTS = ("train-val.csv", "test.csv")
TAIWAN_SHA = "f623d13a7f0388dada08ca5b5f22716557defce3832f50ff6af7153e0eab676d"
TAIWAN_URL = "https://data.mendeley.com/public-files/datasets/ngdgg79rzb/files/255e82b6-2b3a-41d2-bd07-7bfaf284a533/file_downloaded"
SOURCES = {
    "taiwan_tomato": "Mei-Ling Huang and Ya-Han Chang, Dataset of Tomato Leaves, v1, https://doi.org/10.17632/ngdgg79rzb.1",
    "tom2024": "Obed Appiah et al., TOM2024, v1, https://doi.org/10.17632/3d4yg89rtr.1",
}
LICENSE = "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/"
TAIWAN = {
    "Bacterial spot": "bacterial_spot",
    "Late blight": "late_blight",
    "health": "healthy",
}
TOM2024 = {
    "c886ca43-7c62-4a7d-a2fe-63274e0fdfd5": (
        "tomato_diseases/feuille_saine",
        "healthy",
        80,
    ),
    "0994eb6f-9454-447d-8fe3-947c915cb55d": (
        "tomato_diseases/mildiou_de_la_tomate_d",
        "late_blight",
        74,
    ),
    "dc5d280c-192f-4e90-adb8-3321a8fc29ad": (
        "onion_diseases/Feuille saine",
        "not_tomato_leaf",
        35,
    ),
    "287e4fec-7f6e-4759-a6f7-747d4376261b": (
        "maize_diseases/Feuille saine",
        "not_tomato_leaf",
        35,
    ),
}
HASH_FIELDS = ("sha256", "original_sha256", "normalized_sha256")
TOM_MANIFEST = "source_tom2024.csv"
REVIEW = "visual_review.csv"
TAIWAN_MAX_BYTES = 64 * 1024 * 1024
TOM2024_MAX_BYTES = 16 * 1024 * 1024


def fetch(url: str, *, listing: bool = False) -> bytes:
    limit = TAIWAN_MAX_BYTES if url == TAIWAN_URL else TOM2024_MAX_BYTES
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=35) as response:
                data: bytes = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Mendeley response exceeds size limit")
            if listing and not data.lstrip().startswith(b"["):
                raise URLError(f"Invalid Mendeley listing: {data[:100]!r}")
            return data
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
            sleep(2**attempt)
    raise AssertionError("unreachable")


def tom2024_image_path(directory: Path, row: dict[str, str]) -> Path:
    source_id = row["source_id"]
    try:
        file_id = UUID(source_id)
    except ValueError as error:
        raise ValueError("Invalid TOM2024 file ID") from error
    url = f"https://data.mendeley.com/public-files/datasets/3d4yg89rtr/files/{file_id}/file_downloaded"
    if row["download_url"] != url or not re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"]):
        raise ValueError("Invalid TOM2024 download URL or SHA-256")
    return directory / f"{row['original_sha256']}.jpg"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def fetch_sources(target: Path) -> None:
    taiwan_archive = target / "taiwan.7z"
    if not taiwan_archive.exists():
        data = fetch(TAIWAN_URL)
        if digest(data) != TAIWAN_SHA:
            raise ValueError("Taiwan archive SHA-256 mismatch")
        taiwan_archive.write_bytes(data)
    if digest(taiwan_archive.read_bytes()) != TAIWAN_SHA:
        raise ValueError("Taiwan archive SHA-256 mismatch")
    raw = target / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xf", str(taiwan_archive), "-C", str(raw), "taiwan/Preprocessed data"],
        check=True,
    )

    manifest = target / TOM_MANIFEST
    if manifest.exists():
        candidates = read_csv(manifest)
    else:
        candidates = []
        for folder, (name, label, cap) in TOM2024.items():
            url = f"https://data.mendeley.com/public-api/datasets/3d4yg89rtr/files?folder_id={folder}&version=1"
            records = json.loads(fetch(url, listing=True))
            groups = set()
            for record in sorted(records, key=lambda item: digest(item["id"].encode())):
                match = re.fullmatch(
                    r"\d+_(\d{13})(?:_\d+x\d+)?\.jpg", record["filename"], re.IGNORECASE
                )
                if not match or match[1] in groups:
                    continue
                groups.add(match[1])
                content = record["content_details"]
                candidates.append(
                    {
                        "class_id": label,
                        "group": f"tom2024:{match[1]}",
                        "source_file": f"CATEGORY A/CATA-French/{name}/{record['filename']}",
                        "source_id": record["id"],
                        "source_label": name,
                        "original_sha256": content["sha256_hash"],
                        "download_url": content["download_url"],
                    }
                )
                if len(groups) == cap:
                    break
            print(
                f"TOM2024 {name}: selected {len(groups)} groups from {len(records)} files",
                flush=True,
            )
        write_csv(manifest, candidates)

    tom_raw = raw / "tom2024"
    tom_raw.mkdir(exist_ok=True)

    def acquire(row: dict[str, str]) -> None:
        expected = row["original_sha256"]
        path = tom2024_image_path(tom_raw, row)
        data = path.read_bytes() if path.exists() else fetch(row["download_url"])
        if digest(data) != expected:
            raise ValueError(f"TOM2024 SHA-256 mismatch: {row['source_file']}")
        if not path.exists():
            path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for _ in pool.map(acquire, candidates):
            pass
    print(f"Verified {len(candidates)} TOM2024 downloads", flush=True)


def candidates(target: Path) -> list[tuple[dict[str, str], Path]]:
    raw = target / "raw"
    taiwan = raw / "taiwan/Preprocessed data"
    taiwan_files = sorted(taiwan.glob("*/*/*.JPG"))
    if len(taiwan_files) != 622:
        raise ValueError(
            f"Expected 622 non-augmented Taiwan images, found {len(taiwan_files)}"
        )
    result = []
    review_rows = read_csv(target / REVIEW)
    review = {row["source_id"]: row["decision"] for row in review_rows}
    tom_rows = read_csv(target / TOM_MANIFEST)
    reviewed_ids = {row["source_id"] for row in tom_rows}
    reviewed_ids.update(
        path.relative_to(raw).as_posix()
        for path in taiwan_files
        if path.parent.name in TAIWAN
    )
    if len(review) != len(review_rows) or set(review) != reviewed_ids:
        raise ValueError("Visual review must cover every candidate exactly once")
    if set(review.values()) - {"field", "field_mixed", "reject"}:
        raise ValueError("Invalid visual review decision")
    for path in taiwan_files:
        label = path.parent.name
        if label not in TAIWAN:
            continue
        serial = re.search(r"\d+$", path.stem)
        if serial is None:
            raise ValueError(f"Unexpected Taiwan filename: {path.name}")
        member = path.relative_to(raw).as_posix()
        decision = review[member]
        if decision == "reject":
            continue
        result.append(
            (
                {
                    "class_id": TAIWAN[label],
                    "group": f"taiwan_tomato:{TAIWAN[label]}:{(int(serial[0]) - 1) // 10:03}",
                    "source": "taiwan_tomato",
                    "domain": decision,
                    "original_sha256": digest(path.read_bytes()),
                    "source_file": f"taiwan.7z!{member}",
                    "source_id": member,
                    "source_label": label,
                },
                path,
            )
        )

    for row in tom_rows:
        decision = review[row["source_id"]]
        if decision == "reject":
            continue
        path = tom2024_image_path(raw / "tom2024", row)
        # Images within a half-hour are potential repeat views from the same visit.
        capture_time = int(row["group"].removeprefix("tom2024:"))
        group = f"tom2024:30min:{capture_time // (30 * 60 * 1000)}"
        result.append(
            ({**row, "source": "tom2024", "domain": decision, "group": group}, path)
        )
    return result


def fingerprint(image: Image.Image) -> str:
    return digest(f"{image.width}x{image.height}\0".encode() + image.tobytes())


def decoded(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def freeze(target: Path, baseline: Path) -> None:
    if (target / "test.csv").exists():
        raise FileExistsError(target / "test.csv")
    selected = candidates(target)
    exact: set[str] = set()
    buckets: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    def index(value: int) -> None:
        for block in hash_blocks(value):
            buckets[block].add(value)

    def nearby(value: int) -> bool:
        return any(
            (value ^ other).bit_count() <= 4
            for block in hash_blocks(value)
            for other in buckets.get(block, ())
        )

    baseline_count = 0
    for name in BASELINE_MANIFESTS:
        for row in read_csv(baseline / name):
            data = (baseline / row["path"]).read_bytes()
            if digest(data) != row["sha256"]:
                raise ValueError(f"Baseline image integrity mismatch: {row['path']}")
            image = decoded(data)
            exact.update(row[key] for key in HASH_FIELDS)
            exact.add(fingerprint(image))
            index(difference_hash(image))
            baseline_count += 1
    print(
        f"Indexed {baseline_count:,} verified baseline train, validation and test images",
        flush=True,
    )

    rows = []
    omitted: Counter[str] = Counter()
    seen_candidates: set[str] = set()
    # Index reviewed field images first so a repeat cannot replace a confirmed field view.
    for source, path in sorted(
        selected, key=lambda sample: sample[0]["domain"] != "field"
    ):
        data = path.read_bytes()
        original = digest(data)
        if original != source["original_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        image = decoded(data)
        pixels = fingerprint(image)
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=73, optimize=True)
        jpeg = stream.getvalue()
        staged = decoded(jpeg)
        staged_hash = digest(jpeg)
        all_hashes = {original, pixels, staged_hash, fingerprint(staged)}
        dhashes = {difference_hash(image), difference_hash(staged)}
        dhashes.update(
            difference_hash(image.transpose(method)) for method in Image.Transpose
        )
        if all_hashes & exact:
            omitted[f"{source['source']}: baseline exact or pixel duplicate"] += 1
        elif all_hashes & seen_candidates:
            omitted[f"{source['source']}: external exact or pixel duplicate"] += 1
        elif any(nearby(value) for value in dhashes):
            omitted[f"{source['source']}: baseline or external dHash duplicate"] += 1
        else:
            relative = f"images/{original[:2]}/{original}.jpg"
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(jpeg)
            rows.append(
                {
                    "path": relative,
                    "class_id": source["class_id"],
                    "group": source["group"],
                    "sha256": staged_hash,
                    "split": "test",
                    "source": source["source"],
                    "domain": source["domain"],
                    "original_sha256": original,
                    "normalized_sha256": pixels,
                    "source_file": source["source_file"],
                    "source_id": source["source_id"],
                    "source_label": source["source_label"],
                    "attribution": f"{SOURCES[source['source']]}; {LICENSE}; resized/re-encoded",
                }
            )
            seen_candidates.update(all_hashes)
            index(difference_hash(image))
            index(difference_hash(staged))

    rows.sort(
        key=lambda row: (
            row["source"],
            row["domain"],
            row["class_id"],
            row["source_file"],
        )
    )
    for key in (*HASH_FIELDS, "path"):
        assert len(rows) == len({row[key] for row in rows}), key
    assert {row["source"] for row in rows} == set(SOURCES)
    assert {row["class_id"] for row in rows if row["domain"] == "field"} == {
        "bacterial_spot",
        "healthy",
        "late_blight",
        "not_tomato_leaf",
    }
    write_csv(target / "test.csv", rows)
    counts = Counter(
        f"{row['source']}/{row['domain']}/{row['class_id']}" for row in rows
    )
    receipt = {
        "rows": len(rows),
        "counts_by_source_domain_class": dict(sorted(counts.items())),
        "omitted": dict(sorted(omitted.items())),
        "visual_review_counts": dict(
            sorted(
                Counter(
                    f"{row['source']}/{row['decision']}"
                    for row in read_csv(target / REVIEW)
                ).items()
            )
        ),
        "baseline_rows_checked": baseline_count,
        "baseline_manifest_sha256": {
            name: digest((baseline / name).read_bytes()) for name in BASELINE_MANIFESTS
        },
        "external_manifest_sha256": digest((target / "test.csv").read_bytes()),
        "source_manifest_sha256": digest((target / TOM_MANIFEST).read_bytes()),
        "visual_review_sha256": digest((target / REVIEW).read_bytes()),
        "taiwan_archive_sha256": TAIWAN_SHA,
        "license": LICENSE,
        "sources": SOURCES,
        "selection": "Taiwan preprocessed non-augmented; TOM2024 Category A original-capture leaf crops. Visual review separates likely in-situ views from staged or uncertain views and excludes non-leaf frames. No model predictions used.",
        "duplicates": "Baseline original, oriented RGB pixel, staged byte and decoded staged RGB hashes; external equivalents and 128-bit dHash <= 4 with mirrored/rotated query, checked against baseline staging and earlier external admissions.",
    }
    (target / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    members = [
        "test.csv",
        "receipt.json",
        TOM_MANIFEST,
        REVIEW,
        *(row["path"] for row in rows),
    ]
    package = target / "dinov2-external-v1.zip"
    with ZipFile(package, "x", compression=ZIP_STORED) as archive:
        for name in members:
            archive.writestr(ZipInfo(name), (target / name).read_bytes())
    with ZipFile(package) as archive:
        assert archive.namelist() == members and archive.testzip() is None
        for row in rows:
            assert digest(archive.read(row["path"])) == row["sha256"]
    print(
        json.dumps(
            {**receipt, "package_sha256": digest(package.read_bytes())}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "freeze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "fetch":
        fetch_sources(args.output.resolve())
    else:
        freeze(args.output, args.baseline)


if __name__ == "__main__":
    main()
