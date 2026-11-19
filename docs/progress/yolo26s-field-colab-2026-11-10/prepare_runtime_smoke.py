"""Package a frozen, real-label runtime fixture; never a training candidate or final test."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from package_baseline import package_bytes

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "infra/derived/yolo26s-plantdoc-leaf-admission-v1/manifest.json"
MANIFEST_SHA256 = "21af798493261a0043c9b5ea034a45730a490fb0909fce59cb60f21e39d1c868"
CLASSES = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot"]


def select_records(manifest, protection):
    selected, used_groups = [], set()
    for split, count in (("train", 2), ("development", 1)):
        for class_id in CLASSES:
            eligible = [r for r in manifest["records"] if r["decision"] == "accepted" and r["split"] == split and r["exif_orientation"] == 1 and {b["class_id"] for b in r["boxes"]} == {class_id}]
            chosen = []
            for row in sorted(eligible, key=lambda r: r["id"]):
                if row["group_id"] in used_groups:
                    continue
                if row["id"] in protection["plantdoc_test_protected_ids"] or row["final_holdout_eligible"]:
                    raise ValueError("Protected or final-holdout image in runtime fixture")
                if split == "train" and row["id"] in protection["plantdoc_train_protected_ids"]:
                    raise ValueError("Protected development image in smoke train")
                chosen.append(row)
                used_groups.add(row["group_id"])
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f"Insufficient admitted smoke records: {split}/{class_id}")
            selected.extend(chosen)
    return selected


def main():
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Pinned admission manifest changed")
    protection = json.loads((HERE / "protected-splits.json").read_bytes())
    selected = select_records(json.loads(raw), protection)
    files = {}
    for row in selected:
        source = (ROOT / row["source_image"]).resolve()
        if not source.is_relative_to((ROOT / "infra/plant_doc/train").resolve()):
            raise ValueError("Source outside approved publisher train")
        image = source.read_bytes()
        if hashlib.sha256(image).hexdigest() != row["source_sha256"]:
            raise ValueError("Source bytes changed")
        split = "train" if row["split"] == "train" else "val"
        files[f"images/{split}/{row['id']}{source.suffix.lower()}"] = image
        files[f"labels/{split}/{row['id']}.txt"] = ("\n".join(f"{CLASSES.index(b['class_id'])} " + " ".join(f"{v:.12g}" for v in b["xywhn"]) for b in row["boxes"]) + "\n").encode()
    fixture = {"purpose": "runtime_smoke_only_no_accuracy_or_acceptance_claim", "candidate_eligible": False, "final_holdout_eligible": False, "admission_sha256": MANIFEST_SHA256, "names": CLASSES, "records": selected}
    files["fixture.json"] = (json.dumps(fixture, indent=2) + "\n").encode()
    payload, hashes = package_bytes(files)
    target = HERE / "runtime-smoke-data.zip"
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("Runtime fixture changed; create another version")
    target.write_bytes(payload)
    receipt = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "member_manifest_sha256": hashlib.sha256(json.dumps(hashes, indent=2).encode()).hexdigest(), "source_admission_sha256": MANIFEST_SHA256, "counts": dict(Counter(r["split"] for r in selected)), "names": CLASSES, "source_ids": [r["id"] for r in selected], "candidate_eligible": False}
    (HERE / "runtime-smoke-data.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
