"""Calibration diagnostics.

Discrimination and calibration fail independently. A model whose AUC is
unchanged can still have drifted badly if its probabilities no longer mean what
they used to — and for anything where the *number* is used downstream (expected
loss, provisioning, pricing, a cutoff) calibration is the property that
matters. These functions measure it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

__all__ = [
    "CalibrationReport",
    "brier_decomposition",
    "calibration_report",
    "expected_calibration_error",
]


def _validate_inputs(
    y_true: pd.Series | np.ndarray | list[int],
    y_prob: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)

    if truth.shape != prob.shape:
        raise ValueError(f"y_true and y_prob must align: {truth.shape} vs {prob.shape}")
    if truth.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got shape {truth.shape}")
    if truth.size == 0:
        raise ValueError("inputs must be non-empty")
    if not np.all(np.isin(truth, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any(prob < 0.0) or np.any(prob > 1.0):
        raise ValueError("y_prob must lie in [0, 1]")

    return truth, prob


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration diagnostics for one scored sample.

    Attributes:
        brier: Mean squared error of the probabilities. Lower is better, but
            it is not comparable across samples with different base rates —
            use the decomposition for that.
        reliability: Mean squared gap between predicted probability and
            observed frequency within a bin. **Lower is better**; this is the
            calibration term.
        resolution: How much bin outcomes vary around the base rate. **Higher
            is better**; this is the discrimination term.
        uncertainty: Base-rate variance ``p(1-p)``. A property of the sample,
            not the model — it is why Brier scores move when the base rate
            moves even if nothing about the model changed.
        ece: Expected calibration error, the count-weighted mean absolute gap.
        observed_rate: Actual event rate.
        predicted_rate: Mean predicted probability. A gap between this and
            `observed_rate` is a pure bias shift, the easiest drift to fix and
            the one most often missed.
        table: Per-bin reliability table.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    observed_rate: float
    predicted_rate: float
    table: pd.DataFrame

    @property
    def bias(self) -> float:
        """Mean predicted probability minus observed rate."""
        return self.predicted_rate - self.observed_rate

    def __repr__(self) -> str:
        return (
            f"CalibrationReport(brier={self.brier:.5f}, ece={self.ece:.5f}, "
            f"bias={self.bias:+.5f}, observed_rate={self.observed_rate:.4f})"
        )


def brier_decomposition(
    y_true: pd.Series | np.ndarray | list[int],
    y_prob: pd.Series | np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """Murphy's decomposition of the Brier score.

    Returns:
        ``(reliability, resolution, uncertainty)``, which satisfy
        ``brier ≈ reliability - resolution + uncertainty``. The identity is
        exact only up to binning granularity, so it holds approximately for
        any finite `n_bins`.
    """
    report = calibration_report(y_true, y_prob, n_bins=n_bins)
    return report.reliability, report.resolution, report.uncertainty


def expected_calibration_error(
    y_true: pd.Series | np.ndarray | list[int],
    y_prob: pd.Series | np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> float:
    """Count-weighted mean absolute gap between predicted and observed rates."""
    return calibration_report(y_true, y_prob, n_bins=n_bins).ece


def calibration_report(
    y_true: pd.Series | np.ndarray | list[int],
    y_prob: pd.Series | np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration diagnostics for a scored sample.

    Bins are equal-width over ``[0, 1]``, which is the convention for
    reliability diagrams: equal-frequency bins would hide the sparsely
    populated high-probability region that usually carries the largest
    calibration error.

    Args:
        n_bins: Number of probability bins.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.random(5000)
        >>> y = (rng.random(5000) < p).astype(int)
        >>> report = calibration_report(y, p)
        >>> report.ece < 0.05  # a perfectly calibrated model
        True
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    truth, prob = _validate_inputs(y_true, y_prob)
    total = truth.size

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=False` gives half-open bins; clip folds p == 1.0 back into the
    # last bin instead of creating a spurious extra one.
    codes = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    observed_sum = np.bincount(codes, weights=truth, minlength=n_bins)
    predicted_sum = np.bincount(codes, weights=prob, minlength=n_bins)

    occupied = counts > 0
    observed_rate = np.divide(observed_sum, counts, out=np.zeros(n_bins), where=occupied)
    predicted_rate = np.divide(predicted_sum, counts, out=np.zeros(n_bins), where=occupied)

    base_rate = float(truth.mean())
    weights = counts / total

    reliability = float((weights * (predicted_rate - observed_rate) ** 2)[occupied].sum())
    resolution = float((weights * (observed_rate - base_rate) ** 2)[occupied].sum())
    uncertainty = base_rate * (1.0 - base_rate)
    ece = float((weights * np.abs(predicted_rate - observed_rate))[occupied].sum())

    table = pd.DataFrame(
        {
            "bin": [f"[{low:.2f}, {high:.2f})" for low, high in pairwise(edges)],
            "count": counts,
            "predicted_rate": predicted_rate,
            "observed_rate": observed_rate,
            "gap": predicted_rate - observed_rate,
        }
    )
    table.loc[~occupied, ["predicted_rate", "observed_rate", "gap"]] = np.nan

    return CalibrationReport(
        brier=float(np.mean((prob - truth) ** 2)),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        ece=ece,
        observed_rate=base_rate,
        predicted_rate=float(prob.mean()),
        table=table,
    )
