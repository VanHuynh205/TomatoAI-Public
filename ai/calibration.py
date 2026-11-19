"""Numerical calibration only; callers must enforce qualified split and model locks."""

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    at_bound: bool
    sample_count: int


def _number(value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite real number")
    return float(value)


def _centered_logits(logits: ArrayLike) -> NDArray[np.float64]:
    try:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
            raise ValueError("Logits must be a nonempty N by K matrix with K >= 2")
        if not np.isfinite(values).all():
            raise ValueError("Logits must be finite")
        with np.errstate(over="raise", invalid="raise"):
            return values - values.max(axis=1, keepdims=True)
    except (TypeError, OverflowError, FloatingPointError) as error:
        raise ValueError("Logits cannot be represented safely as float64") from error


def _distribution(
    centered: NDArray[np.float64], inverse_temperature: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            weights = np.exp(centered * inverse_temperature)
            # One maximum weight is exactly 1; log1p preserves tiny nonzero losses.
            remainder = np.sort(weights, axis=1)[:, :-1].sum(axis=1)
            totals = 1.0 + remainder
            return weights / totals[:, None], np.log1p(remainder)
    except FloatingPointError as error:
        raise ValueError("Scaled logits cannot be represented safely as float64") from error


def scaled_probabilities(logits: ArrayLike, temperature: float) -> NDArray[np.float64]:
    """Return float64 softmax values without changing the input matrix."""
    temperature = _number(temperature)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return _distribution(_centered_logits(logits), 1 / temperature)[0]


def fit_temperature(logits: ArrayLike, targets: ArrayLike) -> TemperatureFit:
    """Minimize mean NLL over inverse temperature [0.05, 20] by convex bisection."""
    values = _centered_logits(logits)
    labels = np.asarray(targets)
    if (
        labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
        or (labels < 0).any() or (labels >= values.shape[1]).any()
    ):
        raise ValueError("Targets must be matching integer class indices")
    correct = values[np.arange(len(values)), labels.astype(np.int64)]

    def objective(inverse: float) -> tuple[float, float]:
        probabilities, normalizer = _distribution(values, inverse)
        try:
            with np.errstate(over="raise", invalid="raise"):
                loss = float(np.mean(normalizer - inverse * correct))
                derivative = float(np.mean((probabilities * values).sum(axis=1) - correct))
        except FloatingPointError as error:
            raise ValueError("Calibration objective cannot be represented safely") from error
        return loss, derivative

    before, _ = objective(1.0)
    lower, upper = 0.05, 20.0
    if objective(lower)[1] >= 0:
        inverse = lower
    elif objective(upper)[1] <= 0:
        inverse = upper
    else:
        for _ in range(64):
            midpoint = (lower + upper) / 2
            if objective(midpoint)[1] > 0:
                upper = midpoint
            else:
                lower = midpoint
        inverse = (lower + upper) / 2
    after, _ = objective(inverse)
    # Keep the uncalibrated model when the objective is flat at floating-point precision.
    if before <= after:
        inverse, after = 1.0, before
    return TemperatureFit(1 / inverse, before, after, inverse in (0.05, 20.0), len(values))


def _scores_and_eligibility(
    logits: ArrayLike, temperature: float, negative_class_index: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = _centered_logits(logits)
    if negative_class_index is not None and (
        isinstance(negative_class_index, (bool, np.bool_))
        or not isinstance(negative_class_index, Integral)
        or not 0 <= negative_class_index < values.shape[1]
    ):
        raise ValueError("Invalid negative class index")
    probabilities = scaled_probabilities(values, temperature)
    # Preserve raw top-1 even when temperature scaling rounds distinct scores into a tie.
    eligible = np.ones(len(values), dtype=np.bool_)
    if negative_class_index is not None:
        eligible = values.argmax(axis=1) != negative_class_index
    return probabilities.max(axis=1), eligible


def select_coverage_threshold(
    logits: ArrayLike, temperature: float, target_coverage: float,
    *, negative_class_index: int | None,
) -> float:
    """Calibrate on true in-domain rows; keep all confidence ties at the boundary."""
    coverage = _number(target_coverage)
    if not 0 < coverage <= 1:
        raise ValueError("Target coverage must be in (0, 1]")
    scores, eligible = _scores_and_eligibility(logits, temperature, negative_class_index)
    # Decimal coverage in a registration must not gain a row from binary rounding.
    required = math.ceil(Fraction(str(coverage)) * len(scores))
    allowed = scores[eligible]
    if len(allowed) < required:
        raise ValueError("Requested coverage is unattainable after negative-class rejection")
    return float(np.sort(allowed)[-required])


def acceptance_mask(
    logits: ArrayLike, temperature: float, threshold: float,
    *, negative_class_index: int | None,
) -> NDArray[np.bool_]:
    """Apply an already frozen threshold without fitting to evaluation rows."""
    threshold = _number(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    scores, eligible = _scores_and_eligibility(logits, temperature, negative_class_index)
    return eligible & (scores >= threshold)
