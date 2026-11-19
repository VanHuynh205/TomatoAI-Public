import csv
import json
import shutil
from importlib import import_module
from pathlib import Path

import pytest

comparison = import_module("infra.derived.dinov3-colab-v2.compare_results")
PREDICTION_FIELDS = ("path", "sha256", "class_id", "source", "domain", "group")


def test_report_checks_frozen_exports_and_scores_locked_rows(tmp_path: Path) -> None:
    train_export = tmp_path / "train"
    model = train_export / "model"
    model.mkdir(parents=True)
    export = tmp_path / "test"
    export.mkdir()
    receipt_path = comparison.RUN / "addition-receipt.json"
    shutil.copyfile(receipt_path, train_export / receipt_path.name)
    receipt = comparison.read_json(receipt_path)
    for name in comparison.MODEL_FILES:
        (model / name).write_text(name, encoding="utf-8")
    (model / "leaf-classifier.json").write_text(
        json.dumps(
            {
                "model_version": comparison.MODEL_VERSION,
                "class_ids": comparison.CLASSES,
                "size": 224,
                "test_time_horizontal_flip": False,
            }
        ),
        encoding="utf-8",
    )
    frozen = train_export / "frozen-model.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "frozen_before_holdout_prediction",
                "model_version": comparison.MODEL_VERSION,
                "selected_epoch": 0,
                "holdout_lock_sha256": receipt["holdout_lock_sha256"],
                "pilot_manifest_sha256": receipt["pilot"]["manifest_sha256"],
                "train_val_sha256": receipt["train_val_sha256"],
                "sha256": {
                    path.name: comparison.digest(path) for path in model.iterdir()
                },
            }
        ),
        encoding="utf-8",
    )
    shutil.copyfile(frozen, export / frozen.name)
    for name in comparison.v1.BENCHMARKS:
        shutil.copyfile(
            comparison.v1.RUN / f"dinov3-{name}-predictions.csv",
            export / f"dinov3-v2-{name}-predictions.csv",
        )

    pilot = receipt["pilot"]
    rows = comparison.v1.locked_rows(
        comparison.RUN / "slif-pilot.csv",
        {
            "sha256": pilot["manifest_sha256"],
            "rows": pilot["rows"],
        },
    )
    paths = {
        version: export / f"dinov3-{version}-slif-predictions.csv"
        for version in ("v1", "v2")
    }
    for version, path in paths.items():
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, (*PREDICTION_FIELDS, "prediction", "confidence", "status")
            )
            writer.writeheader()
            for row in rows:
                label = (
                    "bacterial_spot"
                    if version == "v2" and row["quality"] == "clear"
                    else "healthy"
                )
                writer.writerow(
                    {
                        **{field: row[field] for field in PREDICTION_FIELDS},
                        "prediction": label,
                        "confidence": 0.5,
                        "status": "ok",
                    }
                )

    report = comparison.build_report(export, train_export)
    for benchmark in report["historical_benchmarks"].values():
        assert set(benchmark["models"]) == {"dinov3_v2", "dinov3_v1", "p07", "dinov2"}
        assert benchmark["dinov3_v2_minus_baseline_positive_field"]["dinov3_v1"] == {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_f1_ci95_group_bootstrap": [0.0, 0.0],
        }
        metrics = benchmark["models"]["dinov3_v2"]["all_images"]
        assert metrics["negative_rejection"]["total"] > 0
        assert metrics["acceptance_by_class"]["healthy"]["acceptance"]["total"] > 0
        assert (
            sum(group["acceptance"]["total"] for group in metrics["by_group"])
            == metrics["top1"]["n"]
        )

    results = report["slif_pilot"]["models"]
    assert set(results) == {"dinov3_v2", "dinov3_v1"}
    assert results["dinov3_v1"]["all_images"]["bacterial_spot_recall"]["count"] == 0
    all_images = results["dinov3_v2"]["all_images"]
    assert all_images["bacterial_spot_recall"] == {
        "count": 20,
        "total": 21,
        "rate": 20 / 21,
    }
    assert all_images["bacterial_spot_mean_group_recall"] == 0.75
    assert results["dinov3_v2"]["clear_images"]["bacterial_spot_recall"]["rate"] == 1.0
    assert "macro_f1" not in all_images

    contents = paths["v2"].read_text(encoding="utf-8")
    paths["v2"].write_text(
        contents.replace(rows[0]["sha256"], "0" * 64, 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match sealed test row"):
        comparison.pilot_comparison(rows, pilot, {"dinov3_v2": paths["v2"]})

    (model / "classifier.safetensors").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen model SHA-256 mismatch"):
        comparison.build_report(export, train_export)
