"""Build development-only splits for field adaptation of the existing classifier."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from ai.dinov2_field import NEGATIVE
from ai.dinov3_field import load_manifest
from ai.prepare_dinov2_field import difference_hash, hash_blocks

RACHIN = "rachin"
BLOCK_SIZE = 20
NEAR_DISTANCE = 5
SEED = 42


def capture_block(row: dict[str, str]) -> str:
    member = row["source_file"].split("!", 1)[-1]
    _, _, number = PurePosixPath(member).stem.rpartition("_")
    if not number.isdecimal() or int(number) < 1:
        raise ValueError(f"Missing Rachin sequence number: {member}")
    return f"{row['class_id']}:{(int(number) - 1) // BLOCK_SIZE:03}"


def group_rachin(
    root: Path, rows: list[dict[str, str]]
) -> tuple[dict[str, list[dict[str, str]]], list[tuple[str, str]]]:
    parent: dict[str, str] = {}
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    conflicts = set()
    blocks = [capture_block(row) for row in rows]

    def representative(block: str) -> str:
        parent.setdefault(block, block)
        while parent[block] != block:
            parent[block] = parent[parent[block]]
            block = parent[block]
        return block

    for index, row in enumerate(rows):
        payload = (root / row["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed Rachin image: {row['path']}")
        with Image.open(io.BytesIO(payload)) as image:
            fingerprints = (
                difference_hash(image),
                difference_hash(ImageOps.mirror(image)),
            )
        for fingerprint in set(fingerprints):
            for key in hash_blocks(fingerprint):
                for earlier, value in buckets[key]:
                    if (
                        earlier == index
                        or (fingerprint ^ value).bit_count() > NEAR_DISTANCE
                    ):
                        continue
                    if rows[earlier]["class_id"] != row["class_id"]:
                        conflicts.add((earlier, index))
                    else:
                        a, b = sorted(
                            (
                                representative(blocks[earlier]),
                                representative(blocks[index]),
                            )
                        )
                        parent[b] = a
                buckets[key].append((index, fingerprint))
    excluded = {index for pair in conflicts for index in pair}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, (block, row) in enumerate(zip(blocks, rows, strict=True)):
        if index not in excluded:
            groups[representative(block)].append(row)
    return dict(groups), [
        (rows[a]["path"], rows[b]["path"]) for a, b in sorted(conflicts)
    ]


def split_rachin(groups: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for group, rows in groups.items():
        by_label[rows[0]["class_id"]].append(group)
    result: list[dict[str, str]] = []
    for label, keys in sorted(by_label.items()):
        if len(keys) < 3:
            raise ValueError(f"Rachin {label} has fewer than three proxy groups")
        keys.sort(key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest())
        target = round(sum(len(groups[key]) for key in keys) * 0.25)
        if len(keys) == 3:
            keys.sort(key=lambda key: abs(len(groups[key]) - target))
        minimum_validation_groups = 1 if len(keys) == 3 else 2
        selected = 0
        for index, key in enumerate(keys):
            validation = index < minimum_validation_groups or (
                selected < target and index < len(keys) - 2
            )
            if validation:
                selected += len(groups[key])
            split = "val" if validation else "train"
            result.extend(
                {**row, "split": split, "group": f"rachin:sequence:{key}"}
                for row in groups[key]
            )
    return result


def build(root: Path, output: Path) -> None:
    prior = load_manifest(root, "train-val.csv")
    old_test = load_manifest(root, "test.csv")
    rachin = [row for row in old_test if row["source"] == RACHIN]
    if len(rachin) != 1367 or {row["split"] for row in rachin} != {"test"}:
        raise ValueError("Expected the complete previously exposed Rachin subset")
    groups, conflicts = group_rachin(root, rachin)
    adaptation = split_rachin(groups)
    rows = prior + adaptation
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Only development rows are permitted")
    for key in ("path", "sha256", "original_sha256", "normalized_sha256", "group"):
        train_keys = {row[key] for row in rows if row["split"] == "train"}
        val_keys = {row[key] for row in rows if row["split"] == "val"}
        if train_keys & val_keys:
            raise ValueError(f"Train/validation overlap on {key}")
    expected = {row["class_id"] for row in rachin}
    for split in ("train", "val"):
        if {row["class_id"] for row in adaptation if row["split"] == split} != expected:
            raise ValueError(f"Rachin {split} lost class coverage")
    controls = [row for row in old_test if row["source"] != RACHIN]
    if {row["path"] for row in rows} & {row["path"] for row in controls}:
        raise ValueError("Development overlaps the retained regression controls")

    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "development.csv"
    control_manifest = output / "regression-controls.csv"
    regression = [
        row
        for row in controls
        if row["source"] == "plantvillage" or row["class_id"] == NEGATIVE
    ]
    for file, selection in ((manifest, rows), (control_manifest, regression)):
        with file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(selection)
    coverage = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        selected_rachin = [row for row in adaptation if row["split"] == split]
        coverage[split] = {
            "n": len(selected),
            "sources": dict(sorted(Counter(row["source"] for row in selected).items())),
            "rachin_images": dict(
                sorted(Counter(row["class_id"] for row in selected_rachin).items())
            ),
            "rachin_groups": dict(
                sorted(
                    Counter(
                        label
                        for label, _ in {
                            (row["class_id"], row["group"]) for row in selected_rachin
                        }
                    ).items()
                )
            ),
        }
    receipt = {
        "development_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "regression_controls_sha256": hashlib.sha256(
            control_manifest.read_bytes()
        ).hexdigest(),
        "regression_controls_n": len(regression),
        "input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("train-val.csv", "test.csv")
        },
        "rachin_source_count": len(rachin),
        "source_note": "Previously exposed Rachin images are development-only.",
        "rachin_conflicting_pairs_quarantined": conflicts,
        "rachin_sequence_block": BLOCK_SIZE,
        "rachin_mirrored_dhash_distance": NEAR_DISTANCE,
        "rachin_group_limit": "Sequence blocks and image fingerprints are proxies. Plant and farm IDs are unavailable.",
        "seed": SEED,
        "coverage": coverage,
    }
    (output / "development-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("infra/derived/dinov2-field-v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("infra/derived/dinov2-improvement-v1")
    )
    args = parser.parse_args()
    build(args.data, args.output)


if __name__ == "__main__":
    main()
