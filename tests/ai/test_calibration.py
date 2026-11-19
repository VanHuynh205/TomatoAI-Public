"""Numerical calibration fixtures; no model or experimental dataset is loaded."""

import importlib
import importlib.util
import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def core() -> ModuleType:
    assert importlib.util.find_spec("ai.calibration") is not None, (
        "The separate temperature/coverage calibration core is missing"
    )
    return importlib.import_module("ai.calibration")


def test_temperature_recovers_analytic_binary_optimum() -> None:
    logits = np.tile([2.0, 0.0], (4, 1))
    result = core().fit_temperature(logits, [0, 0, 0, 1])
    assert result.temperature == pytest.approx(2 / math.log(3), rel=1e-9)
    assert result.nll_after == pytest.approx(0.5623351446188083)
    assert result.nll_after < result.nll_before
    assert result.at_bound is False
    assert result.sample_count == 4


@pytest.mark.parametrize("targets,want", [([0, 1], 0.05), ([1, 0], 20.0)])
def test_temperature_reports_constrained_boundary(targets: list[int], want: float) -> None:
    result = core().fit_temperature([[1.0, 0.0], [0.0, 1.0]], targets)
    assert result.temperature == pytest.approx(want)
    assert result.at_bound is True
    assert result.nll_after <= result.nll_before


def test_flat_logits_keep_identity_temperature() -> None:
    result = core().fit_temperature([[9.0, 9.0, 9.0], [-2.0, -2.0, -2.0]], [0, 2])
    assert result.temperature == 1.0
    assert result.nll_before == pytest.approx(math.log(3))
    assert result.nll_after == result.nll_before
    assert result.at_bound is False


def test_small_representable_nll_is_not_rounded_to_zero_before_fitting() -> None:
    result = core().fit_temperature([[40.0, 0.0]], [0])
    assert result.nll_before == pytest.approx(4.248354255291589e-18, rel=1e-12, abs=0)
    assert result.temperature == pytest.approx(0.05)
    assert result.at_bound is True


def test_fit_is_invariant_to_row_order_and_per_row_offsets() -> None:
    rng = np.random.default_rng(2718)
    logits = rng.normal(size=(32, 9))
    targets = rng.integers(9, size=32)
    original = logits.copy()
    result = core().fit_temperature(logits, targets)
    order = rng.permutation(32)
    shifted = logits[order] + rng.normal(size=(32, 1)) * 100
    other = core().fit_temperature(shifted, targets[order])
    assert other.temperature == pytest.approx(result.temperature, rel=1e-8)
    assert other.nll_after == pytest.approx(result.nll_after)
    np.testing.assert_array_equal(logits, original)


@pytest.mark.parametrize("temperature", [0.05, 0.5, 1.0, 3.0, 20.0])
def test_scaling_keeps_top1_and_normalized_finite_probabilities(temperature: float) -> None:
    logits = np.random.default_rng(1729).normal(size=(40, 9))
    probabilities = core().scaled_probabilities(logits, temperature)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), logits.argmax(axis=1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()


def test_scaling_handles_large_offsets_without_exponential_overflow() -> None:
    result = core().scaled_probabilities([[1000.0, 1000.0], [-1000.0, -1001.0]], 1.0)
    np.testing.assert_allclose(result, [[0.5, 0.5], [0.7310585786300049, 0.2689414213699951]])


@pytest.mark.parametrize("logits", [[], [[1.0]], [1.0, 2.0], [[[1.0, 2.0]]], [[math.nan, 0.0]], [[math.inf, 0.0]], [[1e308, -1e308]]])
def test_invalid_or_unrepresentable_logits_are_rejected(logits: Any) -> None:
    with pytest.raises(ValueError):
        core().scaled_probabilities(logits, 1.0)


@pytest.mark.parametrize("temperature", [0, -1, math.nan, math.inf, True])
def test_temperature_must_be_positive_finite_number(temperature: Any) -> None:
    with pytest.raises(ValueError):
        core().scaled_probabilities([[1.0, 0.0]], temperature)


@pytest.mark.parametrize("targets", [[], [0], [0, 2], [0, -1], [0.0, 1.0], [True, False], [[0], [1]]])
def test_fit_rejects_mismatched_or_invalid_targets(targets: Any) -> None:
    with pytest.raises(ValueError):
        core().fit_temperature([[1.0, 0.0], [0.0, 1.0]], targets)


def test_threshold_keeps_all_ties_and_reports_them_via_actual_coverage() -> None:
    logits = np.log([[0.9, 0.1], [0.8, 0.2], [0.8, 0.2], [0.55, 0.45]])
    threshold = core().select_coverage_threshold(logits, 1.0, 0.5, negative_class_index=None)
    assert threshold == pytest.approx(0.8)
    mask = core().acceptance_mask(logits, 1.0, threshold, negative_class_index=None)
    assert mask.tolist() == [True, True, True, False]
    assert mask.mean() == 0.75


def test_permuted_nonwinning_logits_keep_the_same_threshold_tie() -> None:
    logits = [[0, -5, -17, -14, -13, -15, -9, -1, -8],
              [0, -14, -13, -9, -1, -15, -8, -17, -5]]
    threshold = core().select_coverage_threshold(logits, 1.0, 0.5, negative_class_index=8)
    assert threshold == pytest.approx(0.727230574469853)
    mask = core().acceptance_mask(logits, 1.0, threshold, negative_class_index=8)
    assert mask.tolist() == [True, True]


def test_negative_top1_is_ineligible_even_at_high_confidence() -> None:
    logits = np.log([[0.9, 0.05, 0.05], [0.005, 0.005, 0.99], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    threshold = core().select_coverage_threshold(logits, 1.0, 0.75, negative_class_index=2)
    assert threshold == pytest.approx(0.6)
    assert core().acceptance_mask(logits, 1.0, threshold, negative_class_index=2).tolist() == [True, False, True, True]
    with pytest.raises(ValueError, match="unattainable"):
        core().select_coverage_threshold(logits, 1.0, 1.0, negative_class_index=2)


def test_all_negative_top1_cannot_meet_positive_coverage() -> None:
    with pytest.raises(ValueError, match="unattainable"):
        core().select_coverage_threshold([[0.0, 3.0]], 1.0, 0.1, negative_class_index=1)


def test_acceptance_keeps_raw_top1_when_scaling_creates_a_numeric_tie() -> None:
    mask = core().acceptance_mask([[0.0, 1e-300]], 1e10, 0.5, negative_class_index=1)
    assert mask.tolist() == [False]


def test_decimal_coverage_does_not_round_seven_rows_up_to_eight() -> None:
    logits = np.column_stack([np.arange(100, dtype=float) / 100, np.zeros(100)])
    threshold = core().select_coverage_threshold(logits, 1.0, 0.07, negative_class_index=None)
    mask = core().acceptance_mask(logits, 1.0, threshold, negative_class_index=None)
    assert int(mask.sum()) == 7


def test_frozen_threshold_is_applied_without_retargeting_test_coverage() -> None:
    calibration = np.log([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    threshold = core().select_coverage_threshold(calibration, 1.0, 0.5, negative_class_index=None)
    final_fixture = np.log([[0.65, 0.35], [0.55, 0.45]])
    assert not core().acceptance_mask(final_fixture, 1.0, threshold, negative_class_index=None).any()


@pytest.mark.parametrize("coverage", [0, -0.1, 1.01, math.nan, math.inf, True])
def test_invalid_coverage_is_rejected(coverage: Any) -> None:
    with pytest.raises(ValueError):
        core().select_coverage_threshold([[1.0, 0.0]], 1.0, coverage, negative_class_index=None)


@pytest.mark.parametrize("threshold", [-0.1, 1.01, math.nan, math.inf, True])
def test_invalid_threshold_is_rejected(threshold: Any) -> None:
    with pytest.raises(ValueError):
        core().acceptance_mask([[1.0, 0.0]], 1.0, threshold, negative_class_index=None)


@pytest.mark.parametrize("negative_index", [-1, 2, 0.5, True])
def test_invalid_negative_class_index_is_rejected(negative_index: Any) -> None:
    with pytest.raises(ValueError):
        core().select_coverage_threshold([[1.0, 0.0]], 1.0, 0.5, negative_class_index=negative_index)
    with pytest.raises(ValueError):
        core().acceptance_mask([[1.0, 0.0]], 1.0, 0.5, negative_class_index=negative_index)
