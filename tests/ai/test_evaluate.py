"""M7 evaluator contracts on synthetic samples and sealed, local fixture images."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from ai.evaluate import compute_metrics, evaluate, main
from tests.ai.test_evaluation_protocol import inputs as inputs  # noqa: PLC0414
from tests.ai.test_inference import bundle_template as bundle_template  # noqa: PLC0414
from tests.ai.test_inference import class_hash, read_meta, sha, write_meta
from tests.ai.test_inference import valid_bundle as valid_bundle  # noqa: PLC0414
from tests.ai.test_loader import manifest_rows
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414


def test_metrics_include_abstentions_and_zero_support() -> None:
    m = compute_metrics([0, 0, 1], [0, 1, 1], [True, False, True], ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(4 / 9)
    assert m["coverage"] == pytest.approx(2 / 3)
    assert m["accepted_accuracy"] == 1
    assert m["accepted_error"] == 0
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert m["per_class"]["a"]["precision"] == 1
    assert m["per_class"]["a"]["recall"] == 0.5
    assert m["per_class"]["c"] == {
        "support": 0,
        "evaluable": False,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    assert m["complete_class_support"] is False


def test_zero_accepted_accuracy_is_null() -> None:
    m = compute_metrics([0, 1], [0, 1], [False, False], ["z", "a"])
    assert m["coverage"] == 0
    assert m["accepted_accuracy"] is None
    assert m["accepted_error"] is None
    assert m["accuracy"] == m["macro_f1"] == 1
    assert m["class_ids"] == ["z", "a"]
    assert m["complete_class_support"] is True


@pytest.mark.parametrize(
    "args",
    [
        ([], [], [], ["a"]),
        ([0], [], [False], ["a"]),
        ([0], [0], [1], ["a"]),
        ([False], [0], [True], ["a"]),
        ([0], [-1], [True], ["a"]),
        ([1], [0], [True], ["a"]),
        ([0], [0], [True], ["a", "a"]),
        ([0], [0], [True], []),
    ],
)
def test_invalid_metric_inputs(args: Any) -> None:
    with pytest.raises(ValueError):
        compute_metrics(*args)


@pytest.fixture
def evaluation_case(
    valid_bundle: Path, sealed_images: tuple[Path, Path]
) -> tuple[Path, Path]:
    dataset, _ = sealed_images
    meta = read_meta(valid_bundle)
    meta.update(
        dataset_version=dataset.name,
        dataset_sha256=sha(dataset / "seal.json"),
        split_hash=sha(dataset / "split.json"),
        threshold=None,
    )
    write_meta(valid_bundle, meta)
    return valid_bundle, dataset


def test_fixture_report_reproducible_from_csv(
    evaluation_case: tuple[Path, Path], tmp_path: Path
) -> None:
    bundle, dataset = evaluation_case
    out = tmp_path / "report"
    assert (
        main(
            [
                "--bundle",
                str(bundle),
                "--dataset",
                str(dataset),
                "--split",
                "val",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    ids = read_meta(bundle)["class_ids"]
    with (out / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    source = manifest_rows(dataset, "val")
    assert [(r["sample_id"], r["group_id"], r["true_class_id"]) for r in rows] == [
        (r["sample_id"], r["group_id"], r["class_id"]) for r in source
    ]
    assignments = json.loads((dataset / "split.json").read_text())["assignments"]
    components = {r["sample_id"]: r["component_id"] for r in assignments}
    assert [r["component_id"] for r in rows] == [
        components[r["sample_id"]] for r in rows
    ]
    assert len(rows) == report["sample_count"] == 8
    # Independent counts from the serialized rows, with every fixture prediction abstained.
    assert all(r["status"] == "uncertain" and 0 < float(r["score"]) <= 1 for r in rows)
    counts = Counter((r["true_class_id"], r["predicted_class_id"]) for r in rows)
    assert report["confusion_matrix"] == [[counts[a, b] for b in ids] for a in ids]
    assert report["coverage"] == 0 and report["accepted_error"] is None
    assert report["accuracy"] == pytest.approx(
        sum(counts[c, c] for c in ids) / len(rows), abs=1e-8
    )
    f1 = []
    for c in ids:
        support = sum(counts[c, b] for b in ids)
        predicted = sum(counts[a, c] for a in ids)
        f1.append(2 * counts[c, c] / (support + predicted))
        assert report["per_class"][c]["support"] == report["component_support"][c] == 1
    assert report["macro_f1"] == pytest.approx(sum(f1) / len(ids), abs=1e-8)
    assert report["low_support_class_ids"] == ids
    assert report["model_version"] == read_meta(bundle)["model_version"]
    assert report["split"] == "val" and report["class_ids"] == ids
    with pytest.raises(FileExistsError):
        evaluate(bundle, dataset, "val", out)


def test_test_split_requires_locked_protocol(tmp_path: Path) -> None:
    # Even the model and the dataset paths are absent. Fail on the protocol first.
    with pytest.raises(ValueError, match="protocol"):
        evaluate(
            tmp_path / "missing-model",
            tmp_path / "missing-data",
            "test",
            tmp_path / "report",
        )


def test_mismatched_class_order_rejected(
    evaluation_case: tuple[Path, Path], tmp_path: Path
) -> None:
    bundle, dataset = evaluation_case
    meta = read_meta(bundle)
    meta["class_ids"].reverse()
    meta["class_map_sha256"] = class_hash(meta["class_ids"])
    write_meta(bundle, meta)
    with pytest.raises(ValueError, match="class"):
        evaluate(bundle, dataset, "val", tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_corrupt_last_image_leaves_no_partial_report(
    evaluation_case: tuple[Path, Path], tmp_path: Path
) -> None:
    bundle, dataset = evaluation_case
    raw = (
        dataset / json.loads((dataset / "provenance.json").read_text())["source_root"]
    ).resolve()
    (raw / manifest_rows(dataset, "val")[-1]["source_path"]).write_bytes(b"changed")
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="source_sha256"):
        evaluate(bundle, dataset, "val", out)
    assert not out.exists() and not list(tmp_path.glob(".report-*"))


def test_output_must_stay_outside_input_roots(
    evaluation_case: tuple[Path, Path], tmp_path: Path
) -> None:
    bundle, dataset = evaluation_case
    for out in (bundle / "out", dataset / "out", tmp_path / "source" / "out"):
        with pytest.raises(ValueError, match="output"):
            evaluate(bundle, dataset, "val", out)
        assert not out.exists()


def test_changed_bundle_rejected(
    inputs: tuple[list[Path], Path, Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai.evaluation_protocol import create_protocol

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    with (bundles[1] / "model.pth").open("ab") as stream:
        stream.write(b"peer changed")
    monkeypatch.setattr(
        "ai.evaluate.ManifestDataset", lambda *a: pytest.fail("TEST loader opened")
    )
    out = protocol.parent / "result"
    with pytest.raises(ValueError, match="weights"):
        evaluate(bundles[0], dataset, "test", out, protocol)
    assert not out.exists()


def test_test_output_conflicts_rejected_before_stage_or_reservation(
    inputs: tuple[list[Path], Path, Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai.evaluation_protocol import create_protocol

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    ledger = protocol.parent / "test-open-ledger"
    monkeypatch.setattr(
        "ai.evaluate.tempfile.TemporaryDirectory",
        lambda **kwargs: pytest.fail("staged an output conflicting with the protocol"),
    )
    for output in (
        ledger / "resnet50.json",
        ledger / "efficientnet_b0.json",
        ledger / "nested/report",
        dataset.parent / ".test-protocols/nested/report",
        protocol / "report",
        dataset.parent / "ai/new_metric.py/result",
        *(bundle / "nested/report" for bundle in bundles),
        *(decision.parent / "nested/report" for decision in decisions),
    ):
        parent_existed = output.parent.exists()
        with pytest.raises(ValueError, match="output"):
            evaluate(bundles[0], dataset, "test", output, protocol)
        assert output.parent.exists() == parent_existed
        assert not output.exists()
    assert {entry.name for entry in ledger.iterdir()} == {"lock.json"}


def test_benchmark_cannot_publish_into_peer_bundle_or_run(
    inputs: tuple[list[Path], Path, Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundles, dataset, _, decisions = inputs
    manifest = dataset.parent / "images.json"
    manifest.write_text(json.dumps({"dataset_dir": dataset.name}))
    monkeypatch.setattr("ai.evaluate.read_benchmark_images", lambda _: [])
    monkeypatch.setattr(
        "ai.evaluate.benchmark",
        lambda *args: pytest.fail("measured a conflicting output"),
    )
    for root in (bundles[1], *(d.parent for d in decisions)):
        output = root / "nested/benchmark.json"
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "benchmark",
                    "--bundle",
                    str(bundles[0]),
                    "--images",
                    str(manifest),
                    "--out",
                    str(output),
                ]
            )
        assert exc.value.code == 2
        assert (
            "output must be outside model bundles and training runs"
            in capsys.readouterr().err
        )
        assert not output.parent.exists()


def test_fixture_test_gate_before_first_prediction_and_no_second_open(
    inputs: tuple[list[Path], Path, Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai import artifacts
    from ai.evaluation_protocol import create_protocol

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    receipt = protocol.parent / "test-open-ledger/efficientnet_b0.json"

    class FixtureHead(torch.nn.Linear):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            assert receipt.is_file()
            return torch.nn.functional.linear(
                tensor[:, :2].mean((2, 3)), self.weight, self.bias
            )

    monkeypatch.setattr(artifacts, "build_model", lambda arch, k: FixtureHead(2, k))
    out = protocol.parent / "result"
    report = evaluate(bundles[0], dataset, "test", out, protocol)
    assert report == json.loads((out / "metrics.json").read_bytes())
    assert report["sample_count"] == len(manifest_rows(dataset, "test")) == 8
    assert report["protocol_sha256"] == sha(protocol)
    with pytest.raises(ValueError, match="already opened"):
        evaluate(bundles[0], dataset, "test", out.with_name("retry"), protocol)


def test_benchmark_cli_pins_image_manifest_and_creates_output_once(
    evaluation_case: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai import evaluate as timing
    from ai.benchmark import benchmark

    assert timing.benchmark is benchmark

    bundle, dataset = evaluation_case
    row = manifest_rows(dataset, "val")[0]
    manifest = tmp_path / "images.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "val",
                "dataset_dir": dataset.name,
                "seal_sha256": sha(dataset / "seal.json"),
                "sample_ids": [row["sample_id"]],
            }
        )
    )
    calls = []

    def measure(root: Path, images: list[Path], repeats: int) -> dict[str, Any]:
        calls.append(repeats)
        assert root == bundle and [p.name for p in images] == [row["source_path"]]
        return {"warm_p95_ms": 3.5}

    monkeypatch.setattr(timing, "benchmark", measure)
    args = [
        "benchmark",
        "--bundle",
        str(bundle),
        "--images",
        str(manifest),
        "--repeats",
        "101",
        "--out",
    ]
    out = tmp_path / "report.json"
    assert main([*args, str(out)]) == 0
    result = json.loads(out.read_bytes())
    assert result["warm_p95_ms"] == 3.5 and result["images_manifest_sha256"] == sha(
        manifest
    )
    for invalid in [
        out,
        dataset / "benchmark.json",
        tmp_path / "source" / "benchmark.json",
        tmp_path / "test-open-ledger" / "resnet50.json",
        tmp_path / "evaluation-protocol.json",
    ]:
        with pytest.raises(SystemExit) as exc:
            main([*args, str(invalid)])
        assert exc.value.code == 2
    assert calls == [101]
