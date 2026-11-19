"""Screen FieldPlant for related whole-frame images without exporting pixels."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import import_module
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai.data import _decode_rgb  # noqa: E402

PHASH_MODULE = "docs.progress.yolo-image-recovery.near_roboflow"
phash = import_module(PHASH_MODULE).phash
DERIVED = ROOT / "infra/derived"
FIELD = DERIVED / "yolo26s-fieldplant-source-audit-v1"
PROGRESS = ROOT / "docs/progress/yolo26s-field-colab-2026-11-10"
PD = ROOT / "infra/plant_doc"
PV = ROOT / "infra/tomato_plantvillage/TomatoAI/data/raw/plantvillage"
OUTPUT = DERIVED / "yolo26s-fieldplant-near-duplicates-v1" / "candidate-audit.json"
MANIFEST_SHA = "1a32e70d0931ca0636a6b49073e6231b718e307bac74e2c54bb9e23b23ce4b53"
CONFLICT_SHA = "a132853b321e81f7b14bb724218a193b21c35e5b2fb9dc04afac875db1416fe1"
STRICT_BITS, MAX_BITS, MAX_RGB_MAE = 4, 8, 24
TRANSFORMS = (None, *Image.Transpose)


@dataclass
class Source:
    path: Path
    sha256: str
    keys: set[tuple[str, str]] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)
    flags: set[str] = field(default_factory=set)


def audit_inputs():
    digests = {}
    sources: dict[str, Source] = {}

    def read(path, expected=None):
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"Metadata hash changed: {path.name}")
        digests[path.relative_to(ROOT).as_posix()] = actual
        return data

    def add(id_, base, relative, sha, ref, *, keys=(), pixel="", flags=()):
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or Path(relative).is_absolute():
            raise ValueError(f"Image path escaped its source directory: {id_}")
        source = sources.setdefault(id_, Source(path, sha))
        if (source.path, source.sha256) != (path, sha):
            raise ValueError(f"Conflicting path or byte hash for {id_}")
        source.keys.update((kind, value) for kind, value in keys if value)
        source.keys.add(("byte", sha))
        if pixel:
            source.keys.add(("pixel", pixel))
        source.refs.add(ref)
        source.flags.update(flags)

    manifest_path = FIELD / "source-manifest.json"
    conflict_path = FIELD / "conflict-resolution.json"
    manifest = json.loads(read(manifest_path, MANIFEST_SHA))
    conflict = json.loads(read(conflict_path, CONFLICT_SHA))
    if conflict["source_manifest"]["sha256"] != MANIFEST_SHA:
        raise ValueError("Conflict ledger does not reference the pinned source manifest")
    baseline = json.loads(read(PROGRESS / "baseline-exposure.json", manifest["inputs_sha256"][
        (PROGRESS / "baseline-exposure.json").relative_to(ROOT).as_posix()
    ]))
    protected = json.loads(read(PROGRESS / "protected-splits.json", manifest["inputs_sha256"][
        (PROGRESS / "protected-splits.json").relative_to(ROOT).as_posix()
    ]))
    protected_test = set(protected["plantdoc_test_protected_ids"])
    field_groups = {}
    for row in manifest["records"]:
        id_, group = row["id"], row["provisional_group"]
        ledger_group = conflict["groups"][group]
        if id_ not in ledger_group["records"]:
            raise ValueError(f"Conflict ledger is missing {id_}")
        field_groups[id_] = group
        acquired = row["acquisition"]
        add(id_, FIELD / "raw", Path(acquired["path"]).relative_to("raw"), acquired["sha256"],
            "source-manifest.json/records", keys=(("fieldplant_group", group),),
            pixel=acquired["pixel_sha256"], flags=ledger_group["holdout_exclusion_reasons"])
    ledger_groups = {id_: group for group, entry in conflict["groups"].items() for id_ in entry["records"]}
    if ledger_groups != field_groups or sum(len(g["records"]) for g in conflict["groups"].values()) != len(field_groups):
        raise ValueError("FieldPlant group coverage disagrees with conflict ledger")

    for row in baseline["source_records"]:
        id_ = row["sample_id"]
        base = PD if id_.startswith("plantdoc:") else PV
        relative = (ROOT / "infra" / row["source_path"]).relative_to(base)
        add(id_, base, relative, row["source_sha256"], "baseline-exposure.json/source_records",
            keys=(("baseline_group", row["group_id"]), ("pv_leaf_group", row.get("leaf_group_id")),
                  ("pv_component", row.get("split_component_id")), ("pv_uuid", row.get("source_uuid"))),
            pixel=row["pixel_sha256"], flags=("baseline_exposure",))
    for row in protected["plantvillage_protected_records"]:
        add(row["sample_id"], PV, row["source_file"], row["sha256"],
            "protected-splits.json/plantvillage_protected_records",
            keys=(("pv_leaf_group", row["leaf_group_id"]), ("pv_component", row["split_component_id"]),
                  ("pv_uuid", row["source_uuid"])), flags=("protected_split",))

    def plantdoc(row, ref, flags=()):
        add(f"plantdoc:{row['id']}", PD, (ROOT / row["image_path"]).relative_to(PD), row["sha256"],
            ref, keys=(("pd_component", row["component_id"]), ("pd_v1_component", row["v1_component_id"])),
            pixel=row["pixel_sha256"], flags=flags)

    for section in ("plantdoc_protected_records", "plantdoc_allowed_train", "plantdoc_allowed_development"):
        for row in protected[section]:
            flags = {"protected_split"} if section == "plantdoc_protected_records" else set()
            if row["id"] in protected_test:
                flags.add("protected_test")
            plantdoc(row, f"protected-splits.json/{section}", flags)

    def locked_table(relative):
        path = DERIVED / relative
        expected = protected["inputs_sha256"][path.relative_to(ROOT).as_posix()]
        return csv.DictReader(StringIO(read(path, expected).decode("utf-8-sig"), newline=""))

    pd_folder = "plantdoc-folder-review-v2/manifest.csv"
    for row in locked_table(pd_folder):
        flags = ("plantdoc_prior_review",) if row["label_basis"] != "source_folder_preserved_see_reason" else ()
        plantdoc(row, pd_folder, flags)
    for version in ("yolo26s-plantdoc-leaf-admission-v1", "yolo26s-plantdoc-disease-only-v2"):
        ref = f"{version}/manifest.json"
        for row in json.loads(read(DERIVED / ref))["records"]:
            add(f"plantdoc:{row['id']}", PD, (ROOT / row["source_image"]).relative_to(PD),
                row["source_sha256"], ref, pixel=row["source_pixel_sha256"], flags=("plantdoc_prior_review",))

    relations = []
    for ref in ("plantdoc-clean-v1/related-pairs.csv", "plantdoc-folder-review-v2/review-links.csv"):
        for row in locked_table(ref):
            relations.append((f"plantdoc:{row['left']}", f"plantdoc:{row['right']}"))
    for row in locked_table("plantdoc-clean-v1/pv-pairs.csv"):
        if row["pv_sample_id"] in sources:
            relations.append((f"plantdoc:{row['id']}", row["pv_sample_id"]))
    read(ROOT / Path(*PHASH_MODULE.split(".")).with_suffix(".py"))
    read(Path(__file__).resolve())
    return sources, field_groups, relations, digests


def signatures(rgb, transforms):
    small = rgb.resize((32, 32), Image.Resampling.LANCZOS)
    result = []
    for transform in transforms:
        view = small if transform is None else small.transpose(transform)
        result.append((phash(view), np.asarray(view, dtype=np.int16)))
    return result


def compare(query, right_hashes, right_pixels):
    matches = {}
    for transform, (left_hash, left_pixels) in zip(TRANSFORMS, query, strict=True):
        distances = np.bitwise_count(np.uint64(left_hash) ^ right_hashes)
        for index in np.flatnonzero(distances <= MAX_BITS).tolist():
            bits = int(distances[index])
            mae = float(np.abs(left_pixels - right_pixels[index]).mean())
            if bits <= STRICT_BITS or mae <= MAX_RGB_MAE:
                score = (bits, mae)
                if index not in matches or score < matches[index][:2]:
                    matches[index] = (*score, transform.name if transform is not None else "identity")
    return matches


def bounded_jpeg_preview(data):
    """Decode a bounded reference preview only; never change training decode limits."""
    with Image.open(BytesIO(data)) as original:
        if original.format != "JPEG" or original.width * original.height > 30_000_000:
            raise ValueError("bounded_jpeg_preview_not_supported")
        original.draft("RGB", (512, 512))
        if original.width * original.height > 2_000_000:
            raise ValueError("bounded_jpeg_preview_pixel_limit")
        with ImageOps.exif_transpose(original) as oriented:
            return oriented.convert("RGB")


def collect_signatures(sources, field_groups, *, jpeg_reference_previews=False):
    fingerprint_cache = {}
    coverage = Counter()
    unavailable = {}
    by_id = {}
    for id_, source in sources.items():
        try:
            data = source.path.read_bytes()
        except (FileNotFoundError, PermissionError) as error:
            if id_ in field_groups:
                raise
            unavailable[id_] = type(error).__name__
            continue
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise ValueError(f"Source byte hash changed: {id_}")
        coverage["byte_verified"] += 1
        if source.sha256 not in fingerprint_cache:
            try:
                with _decode_rgb(BytesIO(data)) as rgb:
                    transforms = TRANSFORMS if id_ in field_groups else (None,)
                    fingerprint_cache[source.sha256] = signatures(rgb, transforms)
            except ValueError as error:
                if id_ in field_groups:
                    raise
                if jpeg_reference_previews and str(error) == "pixel_limit":
                    try:
                        with bounded_jpeg_preview(data) as rgb:
                            fingerprint_cache[source.sha256] = signatures(rgb, (None,))
                        coverage["jpeg_reference_previews"] += 1
                    except ValueError as preview_error:
                        fingerprint_cache[source.sha256] = str(preview_error)
                else:
                    fingerprint_cache[source.sha256] = str(error)
        signature = fingerprint_cache[source.sha256]
        if isinstance(signature, str):
            unavailable[id_] = signature
        else:
            by_id[id_] = signature
    return by_id, coverage, unavailable


def candidate_edges(sources, field_groups, fingerprints):
    field_ids = list(dict.fromkeys((field_groups[id_], sources[id_].sha256) for id_ in field_groups))
    representatives = {(group, sources[id_].sha256): id_ for id_, group in reversed(field_groups.items())}
    queries = [representatives[key] for key in field_ids]
    references = queries + [id_ for id_ in fingerprints if id_ not in field_groups]
    hashes = np.array([fingerprints[id_][0][0] for id_ in references], dtype=np.uint64)
    pixels = np.stack([fingerprints[id_][0][1] for id_ in references])
    exact = defaultdict(set)
    for id_ in sources:
        for key in sources[id_].keys:
            if key[0] in {"byte", "pixel"}:
                exact[key].add(id_)
    edges = []
    for start, left in enumerate(queries, 1):
        findings = compare(fingerprints[left], hashes[start:], pixels[start:])
        for index, (bits, mae, transform) in sorted(findings.items()):
            right = references[index + start]
            kind = "fieldplant_internal" if right in field_groups else "cross_source"
            edges.append({"left": left, "right": right, "kind": kind, "phash_bits": bits,
                          "rgb_mae": round(mae, 3), "left_transform": transform})
        for key in sorted(sources[left].keys):
            if key[0] in {"byte", "pixel"}:
                for right in sorted(exact[key]):
                    if right not in field_groups:
                        edges.append({"left": left, "right": right, "kind": f"cross_source_exact_{key[0]}"})
    return edges, len(queries), len(references) - len(queries)


def candidate_groups(sources, field_groups, relations, edges):
    parent = {id_: id_ for id_ in sources}

    def root(id_):
        while parent[id_] != id_:
            parent[id_] = parent[parent[id_]]
            id_ = parent[id_]
        return id_

    def join(left, right):
        parent[root(left)] = root(right)

    key_owner = {}
    for id_, source in sources.items():
        for key in source.keys:
            join(id_, key_owner.setdefault(key, id_))
    for left, right in relations:
        join(left, right)
    for edge in edges:
        join(edge["left"], edge["right"])
    members = defaultdict(list)
    witnesses = defaultdict(dict)
    for id_, source in sources.items():
        component = root(id_)
        if id_ in field_groups:
            members[component].append(id_)
        for flag in source.flags:
            witnesses[component].setdefault(flag, id_)
    result = []
    for component in sorted({root(e["left"]) for e in edges}, key=lambda key: min(members[key])):
        ids = sorted(members[component])
        flags = witnesses[component]
        result.append({
            "fieldplant_records": ids,
            "prior_fieldplant_groups": sorted({field_groups[id_] for id_ in ids}),
            "exclusion_witnesses": flags,
            "new_exclusion_flags_by_record": {
                id_: sorted(flags.keys() - sources[id_].flags)
                for id_ in ids if flags.keys() - sources[id_].flags
            },
        })
    return result


def main(*, jpeg_reference_previews=False):
    output = OUTPUT if not jpeg_reference_previews else DERIVED / "yolo26s-fieldplant-near-duplicates-v2/candidate-audit.json"
    sources, field_groups, relations, digests = audit_inputs()
    print(f"Verifying and fingerprinting {len(sources)} source records", flush=True)
    fingerprints, coverage, unavailable = collect_signatures(sources, field_groups, jpeg_reference_previews=jpeg_reference_previews)
    print(f"Comparing {len(fingerprints)} readable records", flush=True)
    edges, queries, references = candidate_edges(sources, field_groups, fingerprints)
    groups = candidate_groups(sources, field_groups, relations, edges)
    external = {e["right"] for e in edges if e["right"] not in field_groups}
    external.update(id_ for g in groups for id_ in g["exclusion_witnesses"].values() if id_ not in field_groups)
    records = {
        id_: {"path": sources[id_].path.relative_to(ROOT).as_posix(), "sha256": sources[id_].sha256,
              "refs": sorted(sources[id_].refs), "flags": sorted(sources[id_].flags)}
        for id_ in sorted(external | unavailable.keys())
    }
    summary = {
        "fieldplant_records": len(field_groups), "fieldplant_queries": queries,
        "external_records": len(sources) - len(field_groups), "external_fingerprinted": references,
        "byte_verified": coverage["byte_verified"], "unavailable": len(unavailable),
        "internal_pairs_compared": queries * (queries - 1) // 2,
        "cross_source_pairs_compared": queries * references,
        "edges_by_kind": dict(Counter(e["kind"] for e in edges)), "candidate_components": len(groups),
        "fieldplant_records_in_candidate_components": sum(len(g["fieldplant_records"]) for g in groups),
        "fieldplant_records_with_new_exclusion_flags": sum(len(g["new_exclusion_flags_by_record"]) for g in groups),
    }
    result = {
        "status": "candidate_review_only", "training_admitted": False, "final_holdout_established": False,
        "method": {"strict_phash_bits": STRICT_BITS, "max_phash_bits": MAX_BITS, "max_rgb_mae": MAX_RGB_MAE,
                   "square_size": 32, "phash": "existing 63-bit DCT after Lanczos square resize and EXIF transpose",
                   "query_transforms": [t.name if t is not None else "identity" for t in TRANSFORMS]},
        "limitations": [("Whole-frame similarity can miss crops, collage constituents, non-right-angle rotations,"
                         " same-plant photos and repeated capture sessions. A candidate edge is not proof of identity."),
                        ("Only FieldPlant-internal and FieldPlant-to-reference near pairs were tested."
                         " External relations come from existing metadata, exact hashes and listed PlantDoc links.")],
        "inputs_sha256": digests, "summary": summary, "edges": edges, "candidate_groups": groups,
        "external_records": records, "unavailable_references": unavailable,
    }
    if jpeg_reference_previews:
        result["method"]["oversize_reference_preview"] = {"format": "JPEG", "native_pixel_limit": 30_000_000, "decoded_preview_pixel_limit": 2_000_000, "draft_request": [512, 512], "preview_count": coverage["jpeg_reference_previews"], "purpose": "similarity screening only; training decoder unchanged"}
        result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError("Audit changed; use a new output version")
    else:
        with output.open("xb") as file:
            file.write(encoded)
    print(json.dumps({"sha256": hashlib.sha256(encoded).hexdigest(), **summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-reference-previews", action="store_true", help="Use bounded JPEG previews for oversized comparison references and write v2")
    args = parser.parse_args()
    main(jpeg_reference_previews=args.jpeg_reference_previews)
