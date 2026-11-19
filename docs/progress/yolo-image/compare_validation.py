"""Compare the locked YOLO runs and shipped baseline on shared validation IDs."""

import argparse
import json
import math
import runpy
from pathlib import Path

from ai.data import _sha256
from ai.evaluate import compute_metrics
from ai.experiments import _completed_run, _prediction_metrics, select_threshold
from ai.train import _atomic_save

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL = HERE.parent / "2026-11-07-retrain"
LEGACY = runpy.run_path(str(HISTORICAL / "compare_validation.py"))
csv_rows = LEGACY["csv_rows"]
json_file = LEGACY["json_file"]


def assert_lineage(metadata, *, config, source_hashes, environment):
    for field, expected in {
        "config": config,
        "source_hashes": source_hashes,
        "environment": environment,
    }.items():
        if metadata.get(field) != expected:
            raise ValueError(f"P02 run lineage mismatch: {field}")


def verify_run_lineage(run, protocol, revision):
    hashes = protocol["files_sha256"] | revision["files_sha256"]
    config_name = f"ai/configs/{run}.json"
    config_path = ROOT / config_name
    if _sha256(config_path) != hashes[config_name]:
        raise ValueError(f"P02 canonical config hash mismatch: {run}")
    packages = {
        row["name"].lower(): row["version"]
        for row in protocol["environment"]["packages"]
    }
    source_hashes = {
        Path(name).name: digest
        for name, digest in hashes.items()
        if Path(name).parent == Path("ai") and Path(name).suffix == ".py"
    }
    directory, best, metadata, metadata_hash = _completed_run(ROOT / "ai/runs" / run)
    assert_lineage(
        metadata,
        config=json_file(config_path),
        source_hashes=source_hashes,
        environment={
            "python": protocol["environment"]["python"].split()[0],
            "torch": packages["torch"],
            "numpy": packages["numpy"],
            "device": "cuda",
        },
    )
    if _sha256(directory / "last.pth") != metadata["checkpoint_hashes"]["last.pth"]:
        raise ValueError(f"P02 last checkpoint hash mismatch: {run}")
    return {
        "run_id": run,
        "completed_epochs": metadata["epoch"],
        "best_epoch": best["epoch"],
        "best_sha256": metadata["checkpoint_hashes"]["best.pth"],
        "parent_best_sha256": best["identity"]["parent_best_sha256"],
        "metadata_sha256": metadata_hash,
    }


def validation_rows(manifest):
    rows = [row for row in csv_rows(manifest) if row["split"] == "val"]
    ids = [row["sample_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("invalid validation membership")
    return rows


def saved_predictions(path, rows, classes):
    value = json_file(path)
    if (
        value["split"] != "val"
        or value["class_ids"] != classes
        or value["sample_count"] != len(rows)
    ):
        raise ValueError("validation export split/classes/count mismatch")
    entries = value["predictions"]
    _, _, scores, _ = _prediction_metrics(entries, rows, classes)
    return {
        entry["sample_id"]: (entry["prediction_index"], score)
        for entry, score in zip(entries, scores, strict=True)
    }


def shipped_predictions(pv, plantdoc, classes, pv_path, plantdoc_path):
    predictions = saved_predictions(pv_path, pv, classes)
    expected = {"infra/" + row["source_path"]: row for row in plantdoc}
    matched = [row for row in csv_rows(plantdoc_path) if row["path"] in expected]
    if len(matched) != len(expected) or {row["path"] for row in matched} != set(
        expected
    ):
        raise ValueError("B0 PlantDoc path membership mismatch")
    for row in matched:
        target = expected[row["path"]]
        score = float(row["score"])
        if (
            row["model"] != "b0-v1"
            or row["error"]
            or row["label"] != target["class_id"]
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("invalid shipped B0 PlantDoc baseline")
        predictions[target["sample_id"]] = (classes.index(row["pred"]), score)
    return predictions


def compare_run(rows, predictions, classes, threshold):
    ids = {row["sample_id"] for row in rows}
    if len(ids) != len(rows) or ids != predictions.keys():
        raise ValueError("prediction/validation membership mismatch")
    groups = {
        source: [row for row in rows if row["sample_id"].startswith(source + ":")]
        for source in ("pv", "plantdoc")
    }
    if sum(map(len, groups.values())) != len(rows) or not all(groups.values()):
        raise ValueError("unexpected validation source membership")
    result = {}
    for source, subset in {"pooled": rows, **groups}.items():
        indexed = {row["sample_id"]: predictions[row["sample_id"]] for row in subset}
        fixed = LEGACY["metrics"](subset, indexed, classes)
        targets = [classes.index(row["class_id"]) for row in subset]
        pairs = [indexed[row["sample_id"]] for row in subset]
        operating = compute_metrics(
            targets,
            [index for index, _ in pairs],
            [threshold is not None and score >= threshold for _, score in pairs],
            classes,
        )
        result[source] = {
            "at_0_5": fixed,
            "at_selected": {
                "threshold": threshold,
                "accepted_count": operating["accepted_count"],
                "coverage": operating["coverage"],
                "accepted_accuracy": operating["accepted_accuracy"],
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", choices=("E09", "E10"))
    args = parser.parse_args()
    dataset = ROOT / "infra/derived/pv8-plantdoc-v1"
    summary_path = ROOT / "docs/experiments/validation-summary-yolo-v1.json"
    baseline_path = HISTORICAL / "source-comparison.json"
    previous_summary = ROOT / "docs/experiments/validation-summary-adaptation-v1.json"
    protocol_path = HERE / "protocol-lock.json"
    revision_path = HERE / "protocol-lock-v2.json"
    inputs = {}

    def track(path):
        inputs[path.relative_to(ROOT).as_posix()] = _sha256(path)
        return path

    protocol = json_file(track(protocol_path))
    revision = json_file(track(revision_path))
    if (
        inputs[protocol_path.relative_to(ROOT).as_posix()]
        != revision["base_lock_sha256"]
    ):
        raise ValueError("P02 base lock hash mismatch")
    evaluation_lock = HERE / "evaluation-lock.json"
    evaluation = json_file(track(evaluation_lock))
    for name, digest in evaluation["files_sha256"].items():
        if _sha256(ROOT / name) != digest:
            raise ValueError(f"Evaluation input hash mismatch: {name}")
    if args.verify_run:
        print(json.dumps(verify_run_lineage(args.verify_run, protocol, revision)))
        return
    rows = validation_rows(track(dataset / "manifest.csv"))
    pv = [row for row in rows if row["sample_id"].startswith("pv:")]
    plantdoc = [row for row in rows if row["sample_id"].startswith("plantdoc:")]
    if (len(rows), len(pv), len(plantdoc)) != (1785, 1716, 69):
        raise ValueError("unexpected sealed validation counts")
    classes = [
        item["class_id"]
        for item in json_file(track(dataset / "label_map.json"))["classes"]
    ]
    summary = json_file(track(summary_path))
    track(baseline_path)
    baseline_name = baseline_path.relative_to(ROOT).as_posix()
    if inputs[baseline_name] != protocol["files_sha256"][baseline_name]:
        raise ValueError("P02 historical comparison hash mismatch")
    historical = json_file(baseline_path)
    if (
        summary["split"] != "val"
        or summary["app_default_architecture"] != "efficientnet_b0"
        or {record["run_id"] for record in summary["runs"]} != {"E09", "E10"}
        or summary["selected_by_architecture"]
        != {"yolo26s_cls": summary["selected_run"]}
    ):
        raise ValueError("unexpected protocol or YOLO validation summary")

    def baseline_input(path):
        track(path)
        name = path.relative_to(ROOT).as_posix()
        if inputs[name] != historical["input_sha256"][name]:
            raise ValueError(f"historical validation input drift: {name}")
        return path

    shipped = json_file(track(ROOT / "ai/artifacts/b0-v1/metadata.json"))
    if shipped["class_ids"] != classes or shipped["model_version"] != "b0-v1":
        raise ValueError("shipped B0 metadata mismatch")
    b0_predictions = shipped_predictions(
        pv,
        plantdoc,
        classes,
        baseline_input(ROOT / "ai/runs/E02/validation-predictions.json"),
        baseline_input(ROOT / ".scratch/plantdoc-diagnosis/b0-v1.csv"),
    )
    records = {
        "b0-v1": compare_run(rows, b0_predictions, classes, shipped["threshold"])
    }
    adaptation = {
        row["run_id"]: row
        for row in json_file(baseline_input(previous_summary))["runs"]
    }
    yolo = {row["run_id"]: row for row in summary["runs"]}
    lineage = {run: verify_run_lineage(run, protocol, revision) for run in yolo}
    if lineage["E10"]["parent_best_sha256"] != lineage["E09"]["best_sha256"]:
        raise ValueError("P02 E10 did not use the verified E09 best")
    for run in ("E06", "E08", "E09", "E10"):
        path = ROOT / f"ai/runs/{run}/validation-predictions.json"
        selected = yolo.get(run)
        predictions = saved_predictions(
            track(path) if selected else baseline_input(path), rows, classes
        )
        threshold = (
            adaptation[run]["threshold"] if selected is None else selected["threshold"]
        )
        if selected:
            scores = [predictions[row["sample_id"]][1] for row in rows]
            correct = [
                predictions[row["sample_id"]][0] == classes.index(row["class_id"])
                for row in rows
            ]
            if (
                threshold != select_threshold(scores, correct)
                or inputs[path.relative_to(ROOT).as_posix()]
                != selected["validation_predictions_sha256"]
                or any(
                    selected[field] != lineage[run][field]
                    for field in ("best_epoch", "best_sha256", "metadata_sha256")
                )
            ):
                raise ValueError(
                    f"YOLO summary threshold or prediction hash mismatch: {run}"
                )
        records[run] = compare_run(rows, predictions, classes, threshold)

    gate = protocol["research_gate"]
    e08_plantdoc = records["E08"]["plantdoc"]["at_0_5"]["macro_f1_supported_classes"]
    e08_pv = records["E08"]["pv"]["at_0_5"]["macro_f1"]
    if not (
        math.isclose(
            e08_plantdoc,
            gate["plantdoc_supported_macro_f1_strictly_greater_than"],
            abs_tol=1e-14,
        )
        and math.isclose(e08_pv - 0.01, gate["pv_macro_f1_minimum"], abs_tol=1e-14)
    ):
        raise ValueError("recomputed E08 metrics disagree with the locked gate")

    results = {}
    for run in ("E09", "E10"):
        record = records[run]
        plantdoc_f1 = record["plantdoc"]["at_0_5"]["macro_f1_supported_classes"]
        pv_f1 = record["pv"]["at_0_5"]["macro_f1"]
        operating = record["pooled"]["at_selected"]
        research_pass = (
            plantdoc_f1 > gate["plantdoc_supported_macro_f1_strictly_greater_than"]
            and pv_f1 >= gate["pv_macro_f1_minimum"]
        )
        results[run] = {
            "plantdoc_macro_f1_delta_vs_e08": plantdoc_f1 - e08_plantdoc,
            "pv_macro_f1_delta_vs_e08": pv_f1 - e08_pv,
            "research_gate_pass": research_pass,
            "operating_threshold_pass": operating["threshold"] is not None,
            "eligible_for_task6_review": research_pass
            and operating["threshold"] is not None,
        }
    winner = summary["selected_run"]
    result = {
        "schema_version": 1,
        "protocol_id": revision["protocol_id"],
        "split": "validation_only",
        "complex_background_scope": "PlantDoc validation is a source proxy; no complexity labels.",
        "threshold_sources": {
            "b0-v1": "shipped bundle",
            "E06": "adaptation validation summary",
            "E08": "adaptation validation summary",
            "E09": "P02 validation",
            "E10": "P02 validation",
        },
        "records": records,
        "research_gate": gate,
        "results_by_run": results,
        "run_lineage": lineage,
        "decision": {
            "selected_run": winner,
            "selection": "existing pooled macro-F1, then loss rule",
            **results[winner],
            "app_default_model_version": "b0-v1",
            "test_opened": False,
        },
        "input_sha256": inputs,
    }
    output = HERE / "comparison.json"
    if output.exists():
        if json_file(output) != result:
            raise FileExistsError(f"Refusing to overwrite changed comparison: {output}")
    else:
        _atomic_save(output, result)
    print(json.dumps(result["decision"], allow_nan=False))


if __name__ == "__main__":
    main()
