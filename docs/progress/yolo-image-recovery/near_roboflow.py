"""Find Roboflow near-duplicates using PlantDoc fingerprints already on disk."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ai.data import _decode_rgb, _sha256
from ai.train import _atomic_save

from .audit_roboflow import COMBINED, DATA, HERE, table  # type: ignore[import-not-found]

REFERENCE = Path("infra/derived/plantdoc-clean-v1/manifest.csv")
REVIEW = Path("infra/derived/plantdoc-folder-review-v2/summary.json")
BASIS = np.cos(np.pi / 64 * np.arange(8)[:, None] * (2 * np.arange(32) + 1))


def phash(rgb: Image.Image) -> int:
    # Match the stored PlantDoc v1 63-bit DCT signature.
    thumb = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    coeff = (BASIS @ thumb @ BASIS.T).ravel()[1:]
    return int.from_bytes(np.packbits(coeff > np.median(coeff)).tobytes(), "big")


def crop_signatures(rgb: Image.Image) -> set[int]:
    small = rgb.convert("L").resize((144, 144), Image.Resampling.LANCZOS)
    return {
        phash(small.crop((x, y, x + width, y + height)))
        for width, height in (
            (72, 72),
            (104, 104),
            (72, 144),
            (104, 144),
            (144, 72),
            (144, 104),
        )
        for x in {0, (144 - width) // 2, 144 - width}
        for y in {0, (144 - height) // 2, 144 - height}
    }


def near_distances(
    full: set[int], crops: set[int], reference: tuple[int, int]
) -> tuple[int, int] | None:
    full_delta = min((h ^ r).bit_count() for h in full for r in reference)
    crop_delta = min((h ^ r).bit_count() for h in crops for r in reference)
    return (full_delta, crop_delta) if full_delta <= 8 or crop_delta <= 4 else None


def main() -> None:
    audit_path = HERE / "p06-roboflow-audit.json"
    priority_path = HERE / "p06-roboflow-review-priority.json"
    output = HERE / "p06-roboflow-near-overlap.json"
    if output.exists():
        raise FileExistsError(output)
    paths = (audit_path, priority_path, REFERENCE, REVIEW, COMBINED, Path(__file__))
    hashes = {p.resolve().relative_to(Path.cwd()).as_posix(): _sha256(p) for p in paths}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    priority = json.loads(priority_path.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if (
        priority["audit_sha256"] != _sha256(audit_path)
        or review["v1_manifest_sha256"] != hashes[REFERENCE.as_posix()]
    ):
        raise ValueError("P06 provenance hash mismatch")
    if audit["source_files_sha256"][COMBINED.as_posix()] != hashes[COMBINED.as_posix()]:
        raise ValueError("sealed P02 manifest changed")

    reference_rows = table(REFERENCE)
    sealed = {
        r["sample_id"].removeprefix("plantdoc:"): r
        for r in table(COMBINED)
        if r["sample_id"].startswith("plantdoc:")
    }
    references = []
    for row in reference_rows:
        locked = sealed.get(row["id"])
        if locked is not None and (
            locked["source_sha256"] != row["sha256"]
            or locked["pixel_sha256"] != row["pixel_sha256"]
        ):
            raise ValueError("PlantDoc reference fingerprint provenance mismatch")
        if row["phash"] and row["phash_mirror"]:
            references.append(
                (
                    row["id"],
                    row["source_split"],
                    locked["split"] if locked else None,
                    (int(row["phash"], 16), int(row["phash_mirror"], 16)),
                )
            )

    image_dir = (DATA / "train/images").resolve(strict=True)
    variants_checked = 0
    findings = []
    groups = [g for g in audit["groups"] if g["label_review_hint"] is not None]
    for n, group in enumerate(groups, 1):
        best: dict[str, tuple[int, int, str, str, str | None]] = {}
        for variant in group["variants"]:
            path = (image_dir / variant["image"]).resolve(strict=True)
            if path.parent != image_dir:
                raise ValueError("Roboflow audit path escaped train images")
            with path.open("rb") as stream:
                import hashlib

                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != variant["image_sha256"]
                ):
                    raise ValueError("Roboflow train source changed after audit")
                stream.seek(0)
                with _decode_rgb(stream) as rgb:
                    full = {phash(rgb), phash(ImageOps.mirror(rgb))}
                    crops = crop_signatures(rgb)
            variants_checked += 1
            for image_id, source_split, sealed_split, ref in references:
                distances = near_distances(full, crops, ref)
                if distances is not None:
                    a, b = distances
                    record = (a, b, variant["image"], source_split, sealed_split)
                    previous = best.get(image_id)
                    if previous is None or (min(a, b), a, b) < (
                        min(previous[:2]),
                        *previous[:2],
                    ):
                        best[image_id] = record
        if best:
            findings.append(
                dict(
                    export_group=group["group_hint"],
                    source_label_hint=group["label_review_hint"],
                    matches=[
                        dict(
                            plantdoc_id=image_id,
                            full_distance=a,
                            crop_distance=b,
                            export_image=name,
                            source_split=source_split,
                            sealed_split=sealed_split,
                        )
                        for image_id, (
                            a,
                            b,
                            name,
                            source_split,
                            sealed_split,
                        ) in sorted(best.items())
                    ],
                )
            )
        if n % 100 == 0:
            print(f"Compared {n}/{len(groups)} Roboflow groups", flush=True)
    if any(_sha256(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("near-overlap audit input changed")
    suspects = {r["export_group"] for r in findings}
    impacted = [
        row
        for row in priority["pending"]
        if suspects.intersection(row["export_groups"])
    ]
    _atomic_save(
        output,
        dict(
            schema_version=1,
            source_files_sha256=hashes,
            training_ready=False,
            thresholds={"full_or_mirror": 8, "candidate_crop": 4},
            cached_reference_count=len(references),
            variants_checked=variants_checked,
            groups_checked=len(groups),
            suspect_groups=len(findings),
            findings=findings,
            priority_impacted=len(impacted),
            priority_impacted_counts=dict(
                Counter(r["source_label_hint"] for r in impacted)
            ),
            original_plantdoc_images_opened=0,
        ),
    )
    print(
        f"{len(findings)} suspect groups among {len(groups)}. Priority families impacted: {len(impacted)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
