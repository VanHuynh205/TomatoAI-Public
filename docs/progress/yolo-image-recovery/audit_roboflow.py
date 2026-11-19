"""Audit local Roboflow train against sealed metadata without creating crops."""

import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote

from ai.data import _pixel_hash, _sha256
from ai.train import _atomic_save

HERE = Path(__file__).resolve().parent
DATA = Path("infra/tomato_vn")
PV = Path("infra/derived/pv8-v1/manifest.csv")
COMBINED = Path("infra/derived/pv8-plantdoc-v1/manifest.csv")
PD = Path("infra/derived/plantdoc-folder-review-v2/manifest.csv")
CLASSES = {
    0: "bacterial_spot",
    1: "early_blight",
    2: "healthy",
    3: "late_blight",
    5: "leaf_mold",
    7: "septoria_leaf_spot",
    9: "yellow_leaf_curl",
}
SOURCE_NAMES = (
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Miner",
    "Leaf Mold",
    "Mosaic Virus",
    "Septoria",
    "Spider Mites",
    "Yellow Leaf Curl Virus",
)


class Variant(TypedDict):
    image: str
    image_sha256: str
    pixel_sha256: str
    decode_error: str
    label_sha256: str
    class_indices: list[int]
    invalid_boxes: bool
    byte_matches: list[str]
    pixel_matches: list[str]


def filename_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", unquote(name)).casefold()
    text = re.sub(r"(?:[._-](?:jpeg|jpg|png|webp))+$", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def parse_classes(text: str) -> set[int]:
    classes = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        label, *coords = line.split()
        x, y, w, h = map(float, coords)
        label_id = int(label)
        if (
            not 0 <= label_id < len(SOURCE_NAMES)
            or not all(math.isfinite(n) and 0 <= n <= 1 for n in (x, y, w, h))
            or not (0 < w and 0 < h)
            or min(x - w / 2, y - h / 2) < -1e-6
            or max(x + w / 2, y + h / 2) > 1 + 1e-6
        ):
            raise ValueError("invalid YOLO box")
        classes.add(label_id)
    return classes


def table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(path: Path) -> tuple[str, str, str]:
    digest = _sha256(path)
    pixels, error = _pixel_hash(path)
    return digest, pixels, error


def main() -> None:
    import ast
    import json

    output = HERE / "p06-roboflow-audit.json"
    if output.exists():
        raise FileExistsError(output)
    source_config = (DATA / "data.yaml").read_text(encoding="utf-8")
    names = ast.literal_eval(
        next(
            line[7:]
            for line in source_config.splitlines()
            if line.startswith("names: ")
        )
    )
    if tuple(names) != SOURCE_NAMES:
        raise ValueError("Roboflow class order changed")
    paths = [
        PV,
        COMBINED,
        PD,
        DATA / "data.yaml",
        DATA / "README.roboflow.txt",
        Path(__file__).resolve().relative_to(Path.cwd()),
    ]
    hashes = {p.as_posix(): _sha256(p) for p in paths}
    p05 = json.loads((HERE / "p05-lock.json").read_text(encoding="utf-8"))
    if hashes[COMBINED.as_posix()] != p05["files_sha256"][COMBINED.as_posix()]:
        raise ValueError("sealed P02 manifest changed")
    names_index: dict[str, set[str]] = defaultdict(set)
    byte_index: dict[str, set[str]] = defaultdict(set)
    pixel_index: dict[str, set[str]] = defaultdict(set)
    for path in (PV, COMBINED, PD):
        for row in table(path):
            source = Path(row.get("source_path") or row["image_path"]).name
            split = row.get("split") or row["source_split"]
            tag = f"{path.parent.name}:{split}"
            names_index[filename_key(source)].add(tag)
            byte_index[row.get("source_sha256") or row["sha256"]].add(tag)
            if row["pixel_sha256"]:
                pixel_index[row["pixel_sha256"]].add(tag)

    images = sorted((DATA / "train/images").glob("*.jpg"))
    labels = {p.stem: p for p in (DATA / "train/labels").glob("*.txt")}
    if len(images) != 8661 or {p.stem for p in images} != labels.keys():
        raise ValueError("Roboflow train image/label pairing mismatch")
    if any(p.resolve() != p.absolute() for p in [*images, *labels.values()]):
        raise ValueError("Roboflow train paths must stay in the source directory")
    groups: dict[str, list[Variant]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for count, (image, (sha, pixels, error)) in enumerate(
            zip(images, workers.map(fingerprint, images), strict=True), 1
        ):
            group, sep, augmentation = image.stem.rpartition(".rf.")
            if not sep or re.fullmatch(r"[a-f0-9]{32}", augmentation) is None:
                raise ValueError(f"unexpected export name: {image.name}")
            label_path = labels[image.stem]
            label_hash = _sha256(label_path)
            try:
                classes = sorted(
                    parse_classes(label_path.read_text(encoding="utf-8-sig"))
                )
                invalid = False
            except ValueError:
                classes = []
                invalid = True
            groups[group].append(
                {
                    "image": image.name,
                    "image_sha256": sha,
                    "pixel_sha256": pixels,
                    "decode_error": error,
                    "label_sha256": label_hash,
                    "class_indices": classes,
                    "invalid_boxes": invalid,
                    "byte_matches": sorted(byte_index.get(sha, set())),
                    "pixel_matches": sorted(pixel_index.get(pixels, set())),
                }
            )
            if count % 2000 == 0:
                print(f"Fingerprinted {count}/{len(images)} train images", flush=True)

    totals: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    results = []
    for group, variants in groups.items():
        filename_matches = sorted(names_index.get(filename_key(group), set()))
        matches = set(filename_matches)
        ids: set[int] = set()
        has_invalid = False
        has_exact = False
        has_empty = False
        for variant in variants:
            source_matches = set(variant["byte_matches"]) | set(
                variant["pixel_matches"]
            )
            matches.update(source_matches)
            has_exact |= bool(source_matches)
            ids.update(variant["class_indices"])
            has_invalid |= bool(variant["decode_error"] or variant["invalid_boxes"])
            has_empty |= not variant["class_indices"]
        if matches:
            totals["overlap_groups"] += 1
            totals["validation_name_or_hash_groups"] += any(
                tag.endswith(":val") for tag in matches
            )
            totals["test_name_or_hash_groups"] += any(
                tag.endswith(":test") for tag in matches
            )
        totals["exact_overlap_groups"] += has_exact
        totals["invalid_group_count"] += has_invalid
        totals["mixed_class_group_count"] += len(ids) > 1
        candidate = None
        if not matches and not has_invalid and not has_empty and len(ids) == 1:
            candidate = CLASSES.get(next(iter(ids)))
            if candidate is not None:
                eligible_counts[candidate] += 1
        results.append(
            dict(
                group_hint=group,
                filename_matches=filename_matches,
                variants=variants,
                label_review_hint=candidate,
                training_ready=False,
            )
        )
    if any(_sha256(Path(name)) != digest for name, digest in hashes.items()):
        raise ValueError("metadata changed during audit")
    _atomic_save(
        output,
        dict(
            schema_version=1,
            scope="Roboflow original train export only",
            source_files_sha256=hashes,
            training_ready=False,
            test_images_opened=0,
            images=len(images),
            group_hints=len(groups),
            group_sizes=dict(Counter(map(len, groups.values()))),
            totals=dict(totals),
            label_review_hint_counts=dict(eligible_counts),
            groups=results,
        ),
    )
    print(dict(totals), dict(eligible_counts), flush=True)


if __name__ == "__main__":
    main()
