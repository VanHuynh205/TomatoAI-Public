from importlib import import_module

import pytest
import torch


def recovery():
    return import_module("docs.progress.yolo-image-recovery.head_recovery")


def test_source_weights_preserve_source_mass_and_balance_classes() -> None:
    labels = torch.tensor([0, 0, 1, 0, 1, 1, 1, 1])
    ids = [
        "pv:a",
        "pv:b",
        "pv:c",
        "plantdoc:a",
        "plantdoc:b",
        "plantdoc:c",
        "plantdoc:d",
        "plantdoc:e",
    ]
    weights = recovery().sample_weights(ids, labels, 0.4)
    assert float(weights[:3].sum()) == pytest.approx(0.6)
    assert float(weights[3:].sum()) == pytest.approx(0.4)
    assert float(weights[3] / weights[4]) == pytest.approx(2)
    with pytest.raises(ValueError, match="source"):
        recovery().sample_weights(ids[:-1] + ["test:x"], labels, 0.4)


def test_fold_preserves_logits_on_unseen_features() -> None:
    x = torch.tensor([[2.0, 5.0], [-1.0, 4.0]], dtype=torch.float64)
    mean, scale = torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])
    weight, bias = torch.tensor([[1.0, -2.0], [0.0, 3.0]]), torch.tensor([0.2, 0.3])
    folded_w, folded_b = recovery().fold_head(
        weight.double(), bias.double(), mean.double(), scale.double()
    )
    expected = ((x - mean) / scale) @ weight.double().T + bias
    torch.testing.assert_close(x @ folded_w.T + folded_b, expected)


def test_selection_excludes_failing_gates_even_with_higher_plantdoc() -> None:
    metric = dict(
        plantdoc_f1=0.51, pv_f1=0.98, pooled_f1=0.92, min_recall=0.8, threshold=0.5
    )
    variants = {
        "poor_pv": metric | {"plantdoc_f1": 0.8, "pv_f1": 0.97},
        "no_threshold": metric | {"plantdoc_f1": 0.6, "threshold": None},
        "poor_class": metric | {"plantdoc_f1": 0.58, "min_recall": 0.69},
        "pass": metric,
    }
    gate = dict(
        plantdoc_supported_macro_f1_strictly_greater_than=0.49,
        pv_macro_f1_minimum=0.978,
    )
    assert recovery().select_candidate(variants, gate) == "pass"
    del variants["pass"]
    assert recovery().select_candidate(variants, gate) is None
