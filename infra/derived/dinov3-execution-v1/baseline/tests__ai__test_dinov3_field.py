import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from ai.dinov3_field import (
    _best_checkpoint,
    _save_epoch_checkpoint,
    continuation_weights,
    save_atomic,
    validation_report,
)


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_epoch_save_preserves_committed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool,
) -> None:
    last = tmp_path / "last.pt"
    committed = {
        "epoch": 0 if legacy else -1, "best": 0.4, "history": [{"epoch": 1 if legacy else 0}],
        "model": {"weight": torch.tensor([10])},
    }
    if legacy:
        selected = tmp_path / "best.pt"
        save_atomic({"model": committed["model"]}, selected)
        save_atomic(committed, last)
    else:
        committed["best_epoch"] = 0
        selected = _save_epoch_checkpoint(tmp_path, committed, tmp_path / "best.pt")

    next_epoch = 2 if legacy else 1
    candidate = {
        "epoch": next_epoch - 1, "best": 0.8, "best_epoch": next_epoch,
        "history": [{"epoch": next_epoch - 1}, {"epoch": next_epoch}],
        "model": {"weight": torch.tensor([20])},
    }
    replace = os.replace

    def interrupt_last_commit(source: Path, destination: Path) -> None:
        if destination == last:
            raise OSError("interrupted before last.pt replacement")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_last_commit)
        with pytest.raises(OSError, match="interrupted"):
            _save_epoch_checkpoint(tmp_path, candidate, selected)

    assert (tmp_path / f"best-epoch-{next_epoch}.pt").is_file()
    resumed = torch.load(last, weights_only=True)
    assert resumed["epoch"] + 1 == candidate["epoch"]
    assert resumed["history"] == committed["history"]
    assert resumed["best"] == 0.4
    resumed_best = _best_checkpoint(tmp_path, resumed.get("best_epoch"))
    assert resumed_best == selected
    assert torch.load(resumed_best, weights_only=True)["model"]["weight"].item() == 10

    new_best = _save_epoch_checkpoint(tmp_path, candidate, resumed_best)
    resumed = torch.load(last, weights_only=True)
    assert resumed["best"] == 0.8
    assert resumed["history"] == candidate["history"]
    assert _best_checkpoint(tmp_path, resumed["best_epoch"]) == new_best
    assert torch.load(new_best, weights_only=True)["model"]["weight"].item() == 20
    assert selected.exists() == legacy


def test_continuation_sampler_balances_classes_and_softens_capture_group_mass() -> None:
    rows = [
        {"class_id": "bacterial_spot", "domain": domain, "group": group}
        for domain, group, count in (
            ("field", "small", 1), ("field", "large", 4),
            ("controlled", "paper", 4), ("capture_unverified", "unverified", 4), ("web", "web", 1),
        )
        for _ in range(count)
    ]
    rows.extend([
        {"class_id": "healthy", "domain": "field", "group": "healthy"},
        {"class_id": "not_tomato_leaf", "domain": "soil_capture", "group": "soil"},
    ])
    classes: defaultdict[str, float] = defaultdict(float)
    groups: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, continuation_weights(rows, samples=3), strict=True):
        classes[row["class_id"]] += weight
        groups[row["group"]] += weight

    assert classes == pytest.approx({"bacterial_spot": 1, "healthy": 1, "not_tomato_leaf": 1})
    assert groups == pytest.approx({
        "small": 0.25, "large": 0.50, "paper": 0.15, "unverified": 0.10 * 2 / 3, "web": 0.10 / 3,
        "healthy": 1, "soil": 1,
    })


@pytest.mark.parametrize("independent_source", [False, True])
def test_continuation_field_sources_share_mass_without_repeating_tiny_sources(independent_source: bool) -> None:
    batches = [("field", "farm_a", 50), ("field", "farm_b", 2),
               ("controlled", "farm_a", 5), ("web", "web", 5)]
    if independent_source:
        batches.append(("field", "farm_c", 50))
    rows = [
        {"class_id": "bacterial_spot", "domain": domain, "source": source, "group": f"{source}-{i}"}
        for domain, source, count in batches
        for i in range(count)
    ]
    totals: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row, weight in zip(rows, continuation_weights(rows, samples=100), strict=True):
        totals[row["domain"], row["source"]] += weight
        if row["domain"] == "field":
            assert weight <= 0.01

    assert totals["field", "farm_a"] == pytest.approx(0.25 if independent_source else 0.375)
    assert totals["field", "farm_b"] == pytest.approx(0.02)
    assert totals["field", "farm_c"] == pytest.approx(0.25 if independent_source else 0)
    assert totals["controlled", "farm_a"] / totals["web", "web"] == pytest.approx(1.5)
    assert sum(totals.values()) == pytest.approx(1)


def test_validation_diagnostics_weight_groups_equally_and_use_runtime_acceptance() -> None:
    rows = [
        {"class_id": label, "domain": "field", "group": group, "source": source}
        for label, group, source in (
            ("bacterial_spot", "a", "farm1"), ("bacterial_spot", "b", "farm2"),
            ("bacterial_spot", "b", "farm2"), ("bacterial_spot", "b", "farm2"),
            ("healthy", "healthy", "farm1"),
        )
    ]
    labels = ["bacterial_spot", "septoria_leaf_spot", "septoria_leaf_spot", "not_tomato_leaf", "bacterial_spot"]
    scores = [0.5, 0.49, 0.49, 0.9, 0.55]

    report = validation_report(rows, labels, scores, diagnostics=True)

    assert report["field"]["per_class"]["bacterial_spot"]["recall"] == 0.25
    diagnostics = report["field_diagnostics"]
    assert diagnostics["bacterial_spot_groups"] == 2
    assert diagnostics["bacterial_spot_group_recall"] == 0.5
    assert diagnostics["accepted"] == 2
    assert diagnostics["accepted_correct"] == 1
    assert diagnostics["healthy_accepted_as_disease"] == 1
    assert diagnostics["field_by_source"]["farm2"]["n"] == 3
