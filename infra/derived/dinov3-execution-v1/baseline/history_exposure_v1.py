"""Extend existing exposure closure with model-bound manifests; never open images."""

import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
master = import_module("docs.progress.yolo26s-field-colab-2026-11-10.master_exposure")
D = "infra/derived/"
P = "docs/progress/yolo26s-field-colab-2026-11-10/"
IDENTITY = ("path", "sha256", "class_id", "source", "domain", "group")


def require_identity(expected, actual, fields=IDENTITY):
    left = [tuple(row[key] for key in fields) for row in expected]
    right = [tuple(row[key] for key in fields) for row in actual]
    if len(set(left)) != len(left) or len(set(right)) != len(right) or set(left) != set(right):
        raise ValueError("Prediction identities are incomplete, duplicated or changed")


def build():
    pins, manifests, bindings, predictions = {}, {}, {}, []

    def raw(relative):
        path = ROOT / relative
        content = path.read_bytes()
        pins[relative] = hashlib.sha256(content).hexdigest()
        return content

    def read(relative):
        return json.loads(raw(relative))

    def table(relative):
        rows = list(csv.DictReader(io.StringIO(raw(relative).decode("utf-8-sig"))))
        manifests[relative] = {"rows": len(rows), "splits": dict(Counter(r.get("split", "") for r in rows))}
        return rows

    previous = read(P + "master-exposure-v1.json")
    records = {r["id"]: {**r, "keys": {tuple(k) for k in r["keys"]}} for r in previous["records"]}
    training = set(previous["blocked_from_training"])
    final = set(previous["excluded_from_final_holdout"])
    roles = defaultdict(set)
    evidence = defaultdict(set)

    def add(row, role, origin, *, protect_training=False, exclude_final=False):
        digest = row.get("sha256") or row["source_sha256"]
        id_ = row.get("sample_id") or "dino:" + digest
        keys = []
        for field, kind in (("original_sha256", "byte"), ("source_sha256", "byte"),
                            ("pixel_sha256", "pixel"), ("normalized_sha256", "dino_normalized")):
            value = row.get(field)
            if value:
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(f"Invalid {field} in {origin}")
                keys.append((kind, value))
        source = row.get("source") or row.get("source_dataset", "")
        group = row.get("group") or row.get("group_id") or row.get("capture_group_candidate")
        if source and group:
            keys.append(("declared_source_group", source + ":" + group))
        master.add_record(records, id_, digest, keys)
        roles[role].add(id_)
        evidence[id_].add(origin + "::" + role)
        if protect_training:
            training.add(id_)
        if exclude_final:
            final.add(id_)
        return id_

    p07_path = D + "pv8-plantdoc-v1/manifest.csv"
    p07 = table(p07_path)
    lock = read("docs/progress/yolo-image-recovery/p07-lock.json")
    if pins[p07_path] != lock["files_sha256"][p07_path]:
        raise ValueError("P07 manifest no longer matches its run lock")
    inner_train, inner_dev = set(lock["inner_train_ids"]), set(lock["inner_dev_ids"])
    assert len(inner_train) == 6169 and len(inner_dev) == 2069 and not inner_train & inner_dev
    assert inner_train | inner_dev == {r["sample_id"] for r in p07 if r["split"] == "train"}
    bindings["p07"] = {"manifest": p07_path, "binding": "docs/progress/yolo-image-recovery/p07-lock.json", "inner_train": 6169, "inner_development": 2069, "refit_train": 8238, "official_validation": 1785, "threshold_selection": "official validation; not a separate calibration set"}
    for row in p07:
        split = row["split"]
        role = {"train": "training_eligibility", "val": "checkpoint_and_threshold_selection", "test": "historical_reserved_test"}[split]
        add(row, role, p07_path, protect_training=split != "train", exclude_final=split != "test")
        if row["sample_id"] in inner_dev:
            roles["checkpoint_selection"].add(row["sample_id"])

    baseline = read(P + "baseline-exposure.json")
    assert len(baseline["source_records"]) == 10023
    assert set(baseline["known_byte_sha256"]) == {r["source_sha256"] for r in p07 if r["split"] != "test"}
    v2_metadata_path = D + "dinov2-colab-v1/results/leaf-classifier.json"
    v2_meta = read(v2_metadata_path)
    model_specs = [
        ("dinov2_v1", D + "dinov2-field-v2/train-val.csv", v2_meta["manifests"]["train-val.csv"], 6969, 1642, v2_metadata_path),
        ("dinov3_v1", D + "dinov3-colab-v1/train-val.csv", read(D + "dinov3-colab-v1/model/leaf-classifier.json")["manifest_sha256"], 7781, 1642, D + "dinov3-colab-v1/model/leaf-classifier.json"),
        ("dinov3_v2", D + "dinov3-colab-v2/train-val.csv", read(D + "dinov3-colab-v2/frozen-model.json")["train_val_sha256"], 8069, 1642, D + "dinov3-colab-v2/frozen-model.json"),
    ]
    archive = D + "dinov2-improvement-v1/results/dinov2-improvement-model-20261111.zip"
    member = "model/leaf-classifier.json"
    with zipfile.ZipFile(ROOT / archive) as zipped:
        content = zipped.read(member)
    pins[archive + "!/" + member] = hashlib.sha256(content).hexdigest()
    improvement = json.loads(content)
    assert improvement["initial_sha256"]["leaf-classifier.json"] == pins[v2_metadata_path]
    model_specs.append(("dinov2_improvement", D + "dinov2-improvement-v1/development.csv", improvement["development_sha256"], 7963, 1997, archive + "!/" + member))
    for model, path, expected, n_train, n_val, binding in model_specs:
        rows = table(path)
        if pins[path] != expected or Counter(r["split"] for r in rows) != {"train": n_train, "val": n_val}:
            raise ValueError(f"Incorrect model-to-manifest binding: {model}")
        bindings[model] = {"manifest": path, "binding": binding, "manifest_sha256": expected, "train_eligible": n_train, "checkpoint_selection": n_val}
        for row in rows:
            add(row, "training_eligibility" if row["split"] == "train" else "checkpoint_selection", path, protect_training=row["split"] == "val", exclude_final=True)

    legacy_path, external_path = D + "dinov2-field-v2/test.csv", D + "dinov2-external-v1/test.csv"
    legacy, external = table(legacy_path), table(external_path)
    holdout = read(D + "dinov3-colab-v1/holdout-lock.json")
    for key, path, rows in (("legacy", legacy_path, legacy), ("external", external_path, external)):
        assert pins[path] == holdout["manifests"][key]["sha256"]
        assert len(rows) == holdout["manifests"][key]["rows"]
    for path, rows in ((legacy_path, legacy), (external_path, external)):
        for row in rows:
            add(row, "observed_benchmark", path, protect_training=True, exclude_final=True)

    def prediction(model, path, rows, fields=IDENTITY):
        found = table(path)
        require_identity(rows, found, fields)
        predictions.append({"model": model, "path": path, "rows": len(rows), "matched_identity_fields": list(fields)})

    for model, legacy_file, external_file in (
        ("p07", "dinov2-colab-v1/yolo26s-cls-p07-predictions.csv", "dinov3-colab-v1/p07-external.csv"),
        ("dinov2_v1", "dinov2-colab-v1/results/test-predictions.csv", "dinov2-improvement-v1/results/external-prior.csv"),
        ("dinov3_v1", "dinov3-colab-v1/dinov3-legacy-predictions.csv", "dinov3-colab-v1/dinov3-external-predictions.csv"),
        ("dinov3_v2", "dinov3-colab-v2/dinov3-v2-legacy-predictions.csv", "dinov3-colab-v2/dinov3-v2-external-predictions.csv"),
    ):
        prediction(model, D + legacy_file, legacy)
        prediction(model, D + external_file, external)
    prediction("dinov2_improvement", D + "dinov2-improvement-v1/results/external-candidate.csv", external)
    prediction("dinov2_v1_size280", D + "dinov2-improvement-v1/results/external-prior-size280.csv", external)
    controls_path = D + "dinov2-improvement-v1/regression-controls.csv"
    controls = table(controls_path)
    for name in ("prior", "candidate"):
        prediction("dinov2_improvement_" + name, D + f"dinov2-improvement-v1/results/controls-{name}.csv", controls)
    assert {r["sha256"] for r in controls} <= {r["sha256"] for r in legacy}

    slif_path = D + "dinov3-colab-v2/slif-pilot.csv"
    slif = table(slif_path)
    assert len(slif) == 21
    for version in (1, 2):
        prediction(f"dinov3_v{version}", D + f"dinov3-colab-v2/dinov3-v{version}-slif-predictions.csv", slif)
    for row in slif:
        add(row, "observed_benchmark", slif_path, protect_training=True, exclude_final=True)
    bapari_path = D + "dinov2-next-test-v1/dev.csv"
    bapari = table(bapari_path)
    prediction("dinov2_v1_224_and_280", D + "dinov2-bapari-dev-colab-v1/bapari-dev-paired-predictions.csv", bapari, ("path", "sha256", "class_id", "source", "capture_group_candidate"))
    for row in bapari:
        add(row, "observed_diagnostic_then_training", bapari_path, exclude_final=True)

    intake_root = D + "dinov2-next-test-v1/"
    intake = read(intake_root + "intake-receipt.json")
    unresolved = []
    for name in intake["test_intake"]:
        path = intake_root + name
        rows = table(path)
        assert pins[path] == intake["sha256"][name]
        for index, row in enumerate(rows, 2):
            if row.get("sha256"):
                add(row, "historical_candidate_intake_not_admitted", path, protect_training=True)
            else:
                unresolved.append({"path": path, "line": index, "source": row.get("source"), "member": row.get("archive_member"), "note": "No digest in old intake; SLIF's scored subset is separately bound to slif-pilot.csv."})

    # Prepared v1 is inventoried without attributing its folder suffix to a trained model.
    prepared_path = D + "dinov2-field-v1/train-val.csv"
    prepared = table(prepared_path)
    for row in prepared:
        add(row, "prepared_manifest_no_run_binding", prepared_path)

    closed = master.exclusion_sets(records, previous["explicit_relations"], training, final)
    closed_final = set(closed["excluded_from_final_holdout"])
    closed_training = set(closed["blocked_from_training"])
    original_test = {r["source_sha256"] for r in p07 if r["split"] == "test"}
    observed_legacy = {r["original_sha256"] for r in legacy}
    assert len(original_test & observed_legacy) == 621
    assert len(original_test - observed_legacy) == 1114
    key_sets = {"final_exclusion": set(), "training_exclusion": set(), "protected_union": set()}
    for id_, row in records.items():
        for kind, digest in row["keys"]:
            if kind not in {"byte", "pixel", "dino_normalized"}:
                continue
            if id_ in closed_final:
                key_sets["final_exclusion"].add(digest)
            if id_ in closed_training:
                key_sets["training_exclusion"].add(digest)
            if id_ in closed_final or id_ in closed_training:
                key_sets["protected_union"].add(digest)
    summary = {
        "status": "known_history_screening_only_not_data_admission", "models": bindings,
        "inputs_sha256": dict(sorted(pins.items())), "manifests": manifests, "verified_prediction_files": predictions,
        "counts": {"records": len(records), "roles": {k: len(v) for k, v in sorted(roles.items())}, "excluded_from_final": len(closed_final), "blocked_from_training": len(closed_training), "digest_sets": {k: len(v) for k, v in key_sets.items()}},
        "p07_reserved_test": {"total": len(original_test), "observed_later_in_legacy": 621, "outside_verified_legacy_predictions": 1114, "eligible_for_new_final": False},
        "unresolved_intake_identities": unresolved,
        "limitations": ["Training manifests prove eligibility, not actual consumed draws in historical runs.", "Protected union includes reserved records as well as exposed records; membership does not prove model exposure.", "Final-exclusion closure includes prior review/tuning/adaptation/selection and observed benchmarks; it does not grant eligibility to nonmembers.", "Training-exclusion closure preserves historical validation/test/intake declarations. It reports historical crossings, not a retrospective claim that all runs obeyed this union.", "Declared groups and existing relation closure are conservative exclusions, not certificates of plant/session independence. New-source near-duplicate and provenance review remain required.", "The 1,114 P07 TEST records outside the verified legacy predictions remain same-source reserved data; absence of all possible exposure is not proved.", "Intake documents predate SLIF scoring and PlantSeg/Chen training; current manifest and prediction bindings supersede stale not-scored claims.", "No independent development, calibration or final split is qualified by this inventory. Pretraining corpora are not fully auditable here."],
    }
    closure = {**closed, "direct_roles": {k: sorted(v) for k, v in sorted(roles.items())}, "evidence": {k: sorted(v) for k, v in sorted(evidence.items())}, "records": [{**r, "keys": sorted(r["keys"])} for _, r in sorted(records.items())], "explicit_relations": previous["explicit_relations"]}
    master.write_immutable(HERE / "history-exposure-closure.json", closure)
    for key, name in (("final_exclusion", "final-exclusion-hashes.txt"), ("training_exclusion", "training-exclusion-hashes.txt"), ("protected_union", "protected-history-hashes.txt")):
        content = ("\n".join(sorted(key_sets[key])) + "\n").encode()
        path = HERE / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("History changed; create a reviewed new inventory version")
        if not path.exists():
            path.write_bytes(content)
        summary.setdefault("outputs_sha256", {})[name] = hashlib.sha256(content).hexdigest()
    summary["outputs_sha256"]["history-exposure-closure.json"] = hashlib.sha256((HERE / "history-exposure-closure.json").read_bytes()).hexdigest()
    summary["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    master.write_immutable(HERE / "history-exposure-inventory.json", summary)
    return summary


if __name__ == "__main__":
    result = build()
    print(json.dumps({"counts": result["counts"], "p07_reserved_test": result["p07_reserved_test"], "prediction_files": len(result["verified_prediction_files"])}, indent=2))
