"""Record every fitting/selection member declared by the deployed baseline dataset."""

import json
from collections import Counter
from pathlib import Path

from protect_splits import sha256, table

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    dataset = ROOT / "infra/derived/pv8-plantdoc-v1"
    seal = json.loads((dataset / "seal.json").read_bytes())
    for name in ("manifest.csv", "split.json", "inherited-splits.json"):
        if sha256(dataset / name) != seal["hashes"][name]:
            raise ValueError("Baseline dataset metadata seal changed")
    metadata_path = ROOT / "ai/artifacts/yolo26s-cls-v1/metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    if metadata["dataset_version"] != "pv8-plantdoc-v1" or metadata["weights_sha256"] != "322ec3442a5fa54e913da155e8964205ea964117a63fdeec0780820c30f6e56f":
        raise ValueError("Deployed baseline identity changed")
    rows = table(dataset / "manifest.csv")
    if len({r["sample_id"] for r in rows}) != len(rows) or {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Unexpected baseline membership")
    exposed = [r for r in rows if r["split"] in {"train", "val"}]
    pv_path = ROOT / "infra/tomato_plantvillage/TomatoAI/data/processed/plantvillage/pv-d9e4c21eec8e9172/manifest.csv"
    pv = {r["record_id"]: r for r in table(pv_path)}
    records = []
    for row in exposed:
        record = {key: row[key] for key in ("sample_id", "source_path", "source_sha256", "pixel_sha256", "class_id", "group_id", "split")}
        if row["sample_id"].startswith("pv:"):
            source = pv[row["sample_id"]]
            if source["sha256"] != row["source_sha256"]:
                raise ValueError("PV baseline/source hash mismatch")
            record.update({key: source[key] for key in ("source_uuid", "leaf_group_id", "split_component_id")})
            record["legacy_source_pixel_sha256"] = source["pixel_sha256"]
        records.append(record)
    inputs = [metadata_path, dataset / "seal.json", dataset / "manifest.csv", dataset / "split.json", pv_path, ROOT / "docs/progress/yolo-image-recovery/P09/decision.json"]
    result = {"purpose": "Exclude all known baseline fitting and selection exposures and related groups from NEW confirmatory holdout; not a train exclusion list.", "baseline": "yolo26s-cls-v1", "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}, "source_records": records, "counts": dict(Counter(r["split"] for r in records)), "known_byte_sha256": sorted({r["source_sha256"] for r in records}), "known_pixel_sha256": sorted({r["pixel_sha256"] for r in records}), "known_groups": sorted({r["group_id"] for r in records}), "limitations": ["Known declared exposures only; near-duplicate/scene/crop links must be added when admitting new sources.", "Old reserved test is not labelled as baseline exposure; its existing no-training protections remain in force.", "No source pixels opened; no independent new holdout established by this metadata alone."]}
    encoded = (json.dumps(result, indent=2) + "\n").encode()
    output = HERE / "baseline-exposure.json"
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("Exposure evidence changed; create a new version")
    output.write_bytes(encoded)
    print(json.dumps({"counts": result["counts"], "records": len(records), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
