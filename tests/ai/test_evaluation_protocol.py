"""Exercise the holdout gate using only synthetic runs and an M2 fixture."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
import torch

from ai import artifacts
from ai.data import _sha256
from tests.ai.conftest import AuditCase
from tests.ai.test_inference import class_hash
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414


@pytest.fixture
def inputs(
    sealed_images: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Path], Path, Path, list[Path]]:
    from ai import evaluation_protocol as gate

    dataset, _ = sealed_images
    sources = tmp_path / "ai"
    sources.mkdir()
    (sources / "evaluate.py").write_text("def accuracy(): return 1\n")
    monkeypatch.setattr(gate, "_SOURCE_DIR", sources)
    monkeypatch.setattr(artifacts, "build_model", lambda arch, k: torch.nn.Linear(2, k))
    seal = json.loads((dataset / "seal.json").read_bytes())
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "ai/configs/E01.json").read_bytes()
    )
    ids = config["class_ids"]
    bundles, decisions = [], []
    for arch, run in (("efficientnet_b0", "E02"), ("resnet50", "E04")):
        bundle = tmp_path / "bundles" / f"{arch}-v1"
        bundle.mkdir(parents=True)
        decision = tmp_path / "runs" / run / "decision.json"
        decision.parent.mkdir(parents=True)
        best = decision.parent / "best.pth"
        best.write_bytes(f"synthetic checkpoint {run}".encode())
        torch.save(torch.nn.Linear(2, len(ids)).state_dict(), bundle / "model.pth")
        decision.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": arch,
                    "run_id": run,
                    "best_epoch": 3,
                    "best_sha256": _sha256(best),
                    "threshold": 0.5,
                    "selection_rule": "max val macro_f1, then min val loss",
                    "threshold_policy": {
                        "comparison": ">=",
                        "selection": "max coverage",
                    },
                    "identity": {
                        "class_ids": ids,
                        "seed": 42,
                        "seal_sha256": _sha256(dataset / "seal.json"),
                        "split_hash": seal["hashes"]["split.json"],
                    },
                }
            )
        )
        (bundle / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": arch,
                    "model_version": bundle.name,
                    "preprocessing": config["preprocessing"],
                    "class_ids": ids,
                    "dataset_version": dataset.name,
                    "dataset_sha256": _sha256(dataset / "seal.json"),
                    "split_hash": seal["hashes"]["split.json"],
                    "threshold": 0.5,
                    "class_map_sha256": class_hash(ids),
                    "artifact_kind": "trained",
                    "validation_decision_sha256": _sha256(decision),
                    "weights_sha256": _sha256(bundle / "model.pth"),
                }
            )
        )
        bundles.append(bundle)
        decisions.append(decision)
    return bundles, dataset, tmp_path / "report/protocol.json", decisions


def test_lock_both_models_before_either_test_open(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import (
        create_protocol,
        reserve_test_open,
        verify_protocol,
    )

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    record = verify_protocol(protocol, bundles[0], dataset)
    assert record["sources_sha256"]["evaluate.py"] == _sha256(
        dataset.parent / "ai/evaluate.py"
    )
    assert record["dataset"]["hashes"]["label_map.json"] == _sha256(
        dataset / "label_map.json"
    )
    assert [b["run_id"] for b in record["bundles"]] == ["E02", "E04"]
    assert record["bundles"][0]["best_sha256"] == _sha256(
        decisions[0].with_name("best.pth")
    )
    assert record["bundles"][1]["threshold"] == 0.5
    with pytest.raises(FileExistsError):
        create_protocol(bundles, dataset, protocol, decisions=decisions)
    for bundle in bundles:
        receipt = reserve_test_open(protocol, bundle, dataset)
        assert receipt.parent.parent == protocol.parent
        assert json.loads(receipt.read_bytes())["protocol_sha256"] == _sha256(protocol)
        with pytest.raises(ValueError, match="already opened"):
            reserve_test_open(protocol, bundle, dataset)


@pytest.mark.parametrize(
    "changed",
    [
        "peer weights",
        "peer metadata",
        "decision",
        "checkpoint",
        "map",
        "manifest",
        "source",
        "protocol",
    ],
)
def test_drift_cannot_open_or_spend_either_slot(
    inputs: tuple[list[Path], Path, Path, list[Path]], changed: str
) -> None:
    from ai.evaluation_protocol import create_protocol, reserve_test_open

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    path = {
        "peer weights": bundles[1] / "model.pth",
        "peer metadata": bundles[1] / "metadata.json",
        "decision": decisions[1],
        "checkpoint": decisions[1].with_name("best.pth"),
        "map": dataset / "label_map.json",
        "manifest": dataset / "manifest.csv",
        "source": dataset.parent / "ai/evaluate.py",
        "protocol": protocol,
    }[changed]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError):
        reserve_test_open(protocol, bundles[0], dataset)
    path.write_bytes(original)
    assert reserve_test_open(protocol, bundles[0], dataset).is_file()


def test_concurrent_and_failed_evaluations_consume_only_one_open(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import create_protocol, reserve_test_open

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)

    def attempt() -> str:
        try:
            reserve_test_open(protocol, bundles[0], dataset)
        except ValueError as error:
            assert "already opened" in str(error)
            return "denied"
        return "reserved"

    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = [executor.submit(attempt) for _ in range(3)]
    assert sorted(f.result() for f in outcomes) == ["denied", "denied", "reserved"]
    assert attempt() == "denied"


def test_protocol_copy_cannot_reset_ledger(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import create_protocol, reserve_test_open

    bundles, dataset, protocol, decisions = inputs
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    reserve_test_open(protocol, bundles[0], dataset)
    copy = protocol.with_name("replacement.json")
    copy.write_bytes(protocol.read_bytes())
    with pytest.raises(ValueError):
        reserve_test_open(copy, bundles[0], dataset)


def test_another_report_cannot_reset_the_same_holdout(
    inputs: tuple[list[Path], Path, Path, list[Path]],
    sealed_images: tuple[Path, Path],
) -> None:
    from ai.evaluation_protocol import create_protocol, reserve_test_open

    bundles, dataset, protocol, decisions = inputs
    other = protocol.parent.with_name("another-report") / protocol.name

    def source_hashes() -> dict[Path, str]:
        return {
            file: _sha256(file)
            for root in sealed_images
            for file in root.rglob("*")
            if file.is_file()
        }

    original = source_hashes()
    create_protocol(bundles, dataset, protocol, decisions=decisions)
    reserve_test_open(protocol, bundles[0], dataset)
    with pytest.raises(FileExistsError):
        create_protocol(bundles, dataset, other, decisions=decisions)
    assert not other.exists()
    assert reserve_test_open(protocol, bundles[1], dataset).is_file()

    # A copied protocol and a matching report-local lock cannot authorize a new ledger.
    ledger = other.parent / "test-open-ledger"
    ledger.mkdir(parents=True)
    other.write_bytes(protocol.read_bytes())
    (ledger / "lock.json").write_text(
        json.dumps({"protocol": str(other), "protocol_sha256": _sha256(other)})
    )
    with pytest.raises(ValueError):
        reserve_test_open(other, bundles[0], dataset)
    assert not (ledger / "efficientnet_b0.json").exists()
    assert source_hashes() == original


def test_shared_registry_cannot_write_inside_the_raw_source(
    audit_case: AuditCase,
    request: pytest.FixtureRequest,
) -> None:
    from ai.evaluation_protocol import create_protocol

    raw = audit_case.root.with_name(".test-protocols")
    audit_case.root.rename(raw)
    audit_case.root = raw
    bundles, dataset, protocol, decisions = request.getfixturevalue("inputs")
    before = set(raw.iterdir())
    with pytest.raises(ValueError, match="outside"):
        create_protocol(bundles, dataset, protocol, decisions=decisions)
    assert set(raw.iterdir()) == before
    assert not protocol.exists()


def test_concurrent_report_directories_have_one_protocol(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import create_protocol, verify_protocol

    bundles, dataset, protocol, decisions = inputs
    other = protocol.parent.with_name("another-report") / protocol.name
    start = Barrier(2)

    def attempt(out: Path) -> Path | None:
        start.wait(timeout=10)
        try:
            return create_protocol(bundles, dataset, out, decisions=decisions)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = [path for path in executor.map(attempt, [protocol, other]) if path]
    assert len(winners) == 1
    assert verify_protocol(winners[0], bundles[0], dataset)["split"] == "test"


def test_interrupted_publication_closes_every_report_path(
    inputs: tuple[list[Path], Path, Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import evaluation_protocol as gate

    bundles, dataset, protocol, decisions = inputs
    write_once = gate._write_once

    def late_failure(path: Path, content: bytes) -> None:
        write_once(path, content)
        if path == protocol:
            raise OSError("fixture publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(gate, "_write_once", late_failure)
        with pytest.raises(OSError, match="publication failure"):
            gate.create_protocol(bundles, dataset, protocol, decisions=decisions)
    with pytest.raises(ValueError):
        gate.reserve_test_open(protocol, bundles[0], dataset)
    other = protocol.parent.with_name("retry") / protocol.name
    with pytest.raises(FileExistsError):
        gate.create_protocol(bundles, dataset, other, decisions=decisions)


def test_fixture_and_duplicate_architectures_cannot_lock(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import create_protocol

    bundles, dataset, protocol, decisions = inputs
    with pytest.raises(ValueError):
        create_protocol(bundles[:1] * 2, dataset, protocol, decisions=decisions[:1] * 2)
    meta = bundles[1] / "metadata.json"
    value: dict[str, Any] = json.loads(meta.read_bytes())
    value["artifact_kind"] = "fixture"
    meta.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixture"):
        create_protocol(bundles, dataset, protocol, decisions=decisions)
    assert not protocol.exists()


def test_dry_run_cli_and_failed_receipt_write(
    inputs: tuple[list[Path], Path, Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import evaluation_protocol as gate

    bundles, dataset, protocol, decisions = inputs
    cmd = ["lock", "--dataset", str(dataset), "--out", str(protocol)]
    for bundle, decision in zip(bundles, decisions, strict=True):
        cmd.extend(["--bundle", str(bundle), "--decision", str(decision)])
    assert gate.main(cmd) == 0
    assert (
        gate.main(
            [
                "verify",
                "--protocol",
                str(protocol),
                "--dataset",
                str(dataset),
                "--bundle",
                str(bundles[0]),
            ]
        )
        == 0
    )

    def disk_failure(fd: int) -> None:
        raise OSError("fixture disk error")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", disk_failure)
        with pytest.raises(OSError, match="fixture disk error"):
            gate.reserve_test_open(protocol, bundles[0], dataset)
    with pytest.raises(ValueError, match="already opened"):
        gate.reserve_test_open(protocol, bundles[0], dataset)


def test_ledger_path_cannot_be_used_as_protocol(
    inputs: tuple[list[Path], Path, Path, list[Path]],
) -> None:
    from ai.evaluation_protocol import create_protocol

    bundles, dataset, protocol, decisions = inputs
    collision = protocol.with_name("test-open-ledger")
    with pytest.raises(ValueError, match="ledger"):
        create_protocol(bundles, dataset, collision, decisions=decisions)
    assert not collision.exists()
