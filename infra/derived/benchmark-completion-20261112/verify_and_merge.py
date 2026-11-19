"""Recalculate every new metric independently before extending the scorecard."""

import contextlib
import csv
import hashlib
import io
import json
import math
import runpy
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TEMP = Path("C:/Users/admin/AppData/Local/Temp/codex-tomato-scorecard-7575a58477f7467599ebbe0559058519")
IDENTITY = ["path", "sha256", "class_id", "source", "domain", "group"]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independent(rows):
    assert rows
    true = Counter(row["class_id"] for row in rows)
    predicted = Counter(row["prediction"] for row in rows)
    correct = Counter(row["class_id"] for row in rows if row["prediction"] == row["class_id"])
    return len(rows), sum(correct.values()), sum(2 * correct[key] / (n + predicted[key]) for key, n in true.items()) / len(true)


def verify_metric(rows, metric):
    n, correct, f1 = independent(rows)
    assert n == metric["n"]
    assert math.isclose(correct / n, metric["accuracy"], abs_tol=1e-12)
    assert math.isclose(f1, metric["macro_f1"], abs_tol=1e-12)
    labels = metric["confusion_labels"]
    matrix = Counter((row["class_id"], row["prediction"]) for row in rows)
    assert [[matrix[a, b] for b in labels] for a in labels] == metric["confusion"]


def main():
    assert independent([{"class_id": "a", "prediction": "a"}]) == (1, 1, 1)
    assert independent([{"class_id": "a", "prediction": "b"}]) == (1, 0, 0)
    assert independent([{"class_id": "a", "prediction": "a"}, {"class_id": "a", "prediction": "b"}]) == (2, 1, 2 / 3)
    old = json.loads((OUT / "metrics-before.json").read_text(encoding="utf-8"))
    sys.argv = [str(TEMP / "collect.py"), str(ROOT), str(OUT / "archive-metrics.json")]
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(TEMP / "collect.py"))
    baseline = json.loads((OUT / "archive-metrics.json").read_text(encoding="utf-8"))
    assert baseline == old, "Archived source values changed since workbook delivery"
    source = ROOT / "infra/derived/dinov2-field-v2"
    datasets = {
        "historic": read(source / "test.csv"),
        "validation": [r for r in read(source / "train-val.csv") if r["split"] == "val"],
        "taiwan": read(ROOT / "infra/derived/dinov2-external-v1/test.csv"),
    }
    names = {"b0-e06": ns["B0_ADAPT"], "resnet-e08": ns["R50_ADAPT"], "yolo-p08": ns["P08"], "dinov2-v1-280": ns["D2_280"], "dinov2-improvement-280": ns["D2_IMP"]}
    epochs = {"b0-e06": 17, "resnet-e08": 19, "yolo-p08": ns["refit_epochs"][ns["P08"]], "dinov2-v1-280": 6, "dinov2-improvement-280": 6}
    ds_names = {"historic": ns["LEGACY"], "validation": ns["DINO_VAL"], "taiwan": ns["EXT"]}
    local = json.loads((OUT / "local-results.json").read_text())
    colab_root = OUT / "colab-results"
    colab = json.loads((colab_root / "results/results.json").read_text())
    # ZIP records use LF; the local write_text copy uses Windows CRLF.
    with zipfile.ZipFile(OUT / "benchmark-inputs.zip") as sent:
        assert (colab_root / "inputs.json").read_bytes() == sent.read("inputs.json")
    assert json.loads((colab_root / "inputs.json").read_text()) == json.loads((OUT / "colab-inputs.json").read_text())
    assert colab["input_registration_sha256"] == digest(colab_root / "inputs.json")
    assert digest(colab_root / "colab_predict.py") == digest(OUT / "colab_predict.py")
    assert local["input_hashes_unchanged"] and colab["input_hashes_unchanged"]
    assert len(local["results"]) == 8 and len(colab["results"]) == 6
    assert colab["device"] == "cuda" and colab["size"] == 280 and colab["test_time_horizontal_flip"]
    development = read(ROOT / "infra/derived/dinov2-improvement-v1/development.csv")
    dev_hashes = {split: {r["original_sha256"] for r in development if r["split"] == split} for split in ["train", "val"]}
    overlap = {name: {split: sum(row["original_sha256"] in hashes for row in rows) for split, hashes in dev_hashes.items()} for name, rows in datasets.items()}
    assert overlap == {"historic": {"train": 994, "val": 355}, "validation": {"train": 0, "val": 1642}, "taiwan": {"train": 0, "val": 0}}
    checked = []
    for record, folder in [(r, OUT) for r in local["results"]] + [(r, colab_root / "results") for r in colab["results"]]:
        model_id, ds = record["model_id"], record["dataset"]
        file = folder / record["prediction_file"]
        assert digest(file) == record["prediction_sha256"]
        predictions = read(file)
        expected = datasets[ds]
        assert len(predictions) == len(expected)
        assert all(all(p[k] == r[k] for k in IDENTITY) for p, r in zip(predictions, expected, strict=True))
        assert all(0 <= float(p["confidence"]) <= 1 for p in predictions)
        report = record["metrics"]
        subsets = [(ns["ALL"], predictions, report["all"]), (ns["FIELD"], [r for r in predictions if r["domain"] == "field" and r["class_id"] != "not_tomato_leaf"], report["field_tomato"]), (ns["NEG"], [r for r in predictions if r["class_id"] == "not_tomato_leaf"], report["not_tomato"])]
        subsets += [(ns["source_name"](key), [r for r in predictions if r["source"] == key], metric) for key, metric in report["by_source"].items()]
        for cut, rows, metric in subsets:
            verify_metric(rows, metric)
        checked.append({"model": model_id, "dataset": ds, "n": len(predictions), "prediction_sha256": digest(file), "field": independent(subsets[1][1])})
        if model_id.startswith("dinov2") and ds == "taiwan":
            assert record["archived_top1_mismatches"] == 0
            for cut, _, metric in subsets:
                previous = [r for r in old["scores"] if r[:3] == [names[model_id], ds_names[ds], cut]]
                if previous:
                    assert math.isclose(previous[0][6], metric["accuracy"], abs_tol=1e-12)
                    assert math.isclose(previous[0][7], metric["macro_f1"], abs_tol=1e-12)
            continue
        note = "Chấm mới từ checkpoint đã khóa; top-1 trước ngưỡng. "
        note += "Colab T4, 280 px, trung bình logits ảnh gốc/lật ngang." if model_id.startswith("dinov2") else "CPU, tiền xử lý và ngưỡng của bundle 8 lớp; không có lớp ảnh âm."
        if model_id == "dinov2-improvement-280" and ds == "historic":
            note += " TRÙNG DỮ LIỆU PHÁT TRIỂN: 994 ảnh đã train, 355 ảnh đã validation; không phải test độc lập."
        if ds == "validation":
            note += " Validation chung, không phải test độc lập."
        for cut, rows, metric in subsets:
            enhanced = dict(metric)
            if not model_id.startswith("dinov2"):
                accepted = [r for r in rows if r["status"] == "ok"]
                enhanced.update(accepted_count=len(accepted), coverage=len(accepted) / len(rows), accepted_accuracy=sum(r["prediction"] == r["class_id"] for r in accepted) / len(accepted) if accepted else None)
            metrics_file = folder / f"{model_id}-{ds}-metrics.json"
            ns["score"](ds_names[ds], names[model_id], cut, epochs[model_id], enhanced, str(metrics_file.relative_to(ROOT)).replace('\\', '/'), threshold=None if model_id.startswith("dinov2") else 0.5, note=note)
    assert len(checked) == 14
    updated = baseline | {key: ns[key] for key in ["scores", "classes", "epochs", "extras"]}
    for key in ["scores", "classes", "epochs", "extras"]:
        assert updated[key][:len(old[key])] == old[key]
    updated["common_models"] = [names["b0-e06"], names["resnet-e08"], names["yolo-p08"], ns["D2"], ns["D2_280"], ns["D2_IMP"], ns["V1"], ns["V2"], ns["A"], ns["B"]]
    assert len(set(tuple(r[:4]) for r in updated["scores"])) == len(updated["scores"])
    (OUT / "metrics-complete.json").write_text(json.dumps(updated, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    verification = {"prediction_sets_verified": checked, "missing_model_dataset_pairs_completed": 12, "new_prediction_rows": sum(x["n"] for x in checked if not (x["model"].startswith("dinov2") and x["dataset"] == "taiwan")), "parity_rows": 920, "independent_metric_recalculation": True, "all_original_score_rows_preserved": True, "original_workbook_sha256": digest(OUT / "workbook-before.xlsx"), "improvement_exact_original_byte_overlap": overlap, "p08_export_epoch": epochs["yolo-p08"], "new_scores": len(updated["scores"]) - len(old["scores"]), "new_class_rows": len(updated["classes"]) - len(old["classes"])}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in verification.items() if key != "prediction_sets_verified"}, indent=2))


if __name__ == "__main__":
    main()
