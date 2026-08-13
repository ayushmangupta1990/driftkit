"""Population Stability Index (PSI) and Characteristic Stability Index (CSI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .binning import BinSpec, fit_bins

__all__ = ["DriftResult", "csi", "interpret_psi", "psi", "psi_from_counts"]

# Conventional retail-credit bands. They are heuristics from scorecard
# practice, not a statistical test -- see `interpret_psi` for the caveat.
_MINOR_SHIFT = 0.10
_MAJOR_SHIFT = 0.25


@dataclass(frozen=True)
class DriftResult:
    """Outcome of a stability comparison.

    Attributes:
        value: The PSI (or CSI) statistic. Always >= 0; 0 means the two
            binned distributions are identical.
        table: Per-bin contributions, so a headline number can always be
            traced back to the bins that caused it. Columns: ``bin``,
            ``expected_count``, ``actual_count``, ``expected_pct``,
            ``actual_pct``, ``contribution``.
        n_expected: Number of reference observations.
        n_actual: Number of comparison observations.
    """

    value: float
    table: pd.DataFrame
    n_expected: int
    n_actual: int

    @property
    def interpretation(self) -> str:
        """Conventional verbal band for :attr:`value`."""
        return interpret_psi(self.value)

    @property
    def top_bins(self) -> pd.DataFrame:
        """Bins ordered by how much they contributed to the statistic."""
        return self.table.sort_values("contribution", ascending=False)

    def __repr__(self) -> str:
        return (
            f"DriftResult(value={self.value:.4f}, interpretation={self.interpretation!r}, "
            f"n_expected={self.n_expected}, n_actual={self.n_actual})"
        )


def interpret_psi(value: float) -> str:
    """Map a PSI value onto the conventional three-band scale.

    The bands (<0.10 stable, 0.10-0.25 moderate, >0.25 significant) are
    industry convention, not a hypothesis test. They carry no sample-size
    correction, so on a few hundred rows a "significant" reading is often
    noise, and on several million rows a "stable" reading can still hide a
    shift that matters. Treat them as a triage prompt, not a verdict.
    """
    if value < _MINOR_SHIFT:
        return "stable"
    if value < _MAJOR_SHIFT:
        return "moderate shift"
    return "significant shift"


def psi_from_counts(
    expected_counts: npt.NDArray[Any],
    actual_counts: npt.NDArray[Any],
    *,
    alpha: float = 0.5,
    labels: tuple[str, ...] | None = None,
) -> DriftResult:
    """Compute PSI directly from two aligned count vectors.

    Args:
        expected_counts: Per-bin counts from the reference sample.
        actual_counts: Per-bin counts from the comparison sample, in the same
            bin order.
        alpha: Additive smoothing applied to every bin before converting to
            proportions. PSI's ``ln(actual/expected)`` term is undefined when
            either side has an empty bin, and the usual workaround of
            substituting a tiny constant makes the result depend on an
            arbitrary epsilon. Additive smoothing is the well-defined version
            of the same idea; ``alpha=0.5`` is the Jeffreys prior. Set to 0 to
            disable, which will raise on empty bins.

    Raises:
        ValueError: If the vectors have different lengths, or if `alpha` is 0
            and a bin is empty on either side.
    """
    expected = np.asarray(expected_counts, dtype=np.float64)
    actual = np.asarray(actual_counts, dtype=np.float64)

    if expected.shape != actual.shape:
        raise ValueError(
            f"count vectors must align: expected {expected.shape}, actual {actual.shape}"
        )
    if expected.ndim != 1:
        raise ValueError(f"count vectors must be 1-D, got shape {expected.shape}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    n_expected = int(expected.sum())
    n_actual = int(actual.sum())
    if n_expected == 0 or n_actual == 0:
        raise ValueError("both samples must contain at least one observation")

    smoothed_expected = expected + alpha
    smoothed_actual = actual + alpha

    if alpha == 0 and (np.any(smoothed_expected == 0) or np.any(smoothed_actual == 0)):
        raise ValueError(
            "empty bin encountered with alpha=0; PSI is undefined there. "
            "Use alpha>0, or refit with fewer bins."
        )

    expected_pct = smoothed_expected / smoothed_expected.sum()
    actual_pct = smoothed_actual / smoothed_actual.sum()

    contribution = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    value = float(contribution.sum())

    if labels is None:
        labels = tuple(f"bin_{index}" for index in range(expected.size))

    table = pd.DataFrame(
        {
            "bin": list(labels),
            "expected_count": expected,
            "actual_count": actual,
            "expected_pct": expected_pct,
            "actual_pct": actual_pct,
            "contribution": contribution,
        }
    )

    return DriftResult(value=value, table=table, n_expected=n_expected, n_actual=n_actual)


def psi(
    expected: pd.Series | npt.NDArray[Any] | list[object],
    actual: pd.Series | npt.NDArray[Any] | list[object],
    *,
    n_bins: int = 10,
    alpha: float = 0.5,
    bins: BinSpec | None = None,
) -> DriftResult:
    """Population Stability Index between a reference and a comparison sample.

    Bins are learned from `expected` only, then applied unchanged to `actual`.

    Args:
        expected: Reference sample — typically the model's training or
            most-recent-revalidation population.
        actual: Comparison sample — typically the current scoring window.
        n_bins: Number of quantile bins to learn. Ignored when `bins` is given.
        alpha: Additive smoothing; see :func:`psi_from_counts`.
        bins: A pre-fitted :class:`~driftkit.binning.BinSpec`. Pass this when
            monitoring many periods against one frozen reference, so every
            period is measured on identical bins and the series is comparable
            over time.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> reference = rng.normal(700, 50, 10_000)
        >>> current = rng.normal(660, 60, 10_000)
        >>> result = psi(reference, current)
        >>> result.value > 0.25
        True
    """
    spec = bins if bins is not None else fit_bins(expected, n_bins=n_bins)
    return psi_from_counts(
        spec.counts(expected),
        spec.counts(actual),
        alpha=alpha,
        labels=spec.labels,
    )


def csi(
    expected: pd.Series | npt.NDArray[Any] | list[object],
    actual: pd.Series | npt.NDArray[Any] | list[object],
    *,
    points: npt.NDArray[Any] | pd.Series | None = None,
    n_bins: int = 10,
    alpha: float = 0.5,
    bins: BinSpec | None = None,
) -> DriftResult:
    """Characteristic Stability Index for a single input feature.

    CSI answers a narrower question than PSI. PSI is normally run on the model
    *output* and tells you the scored population moved; CSI is run on one
    *input* and tells you which feature moved it. Mechanically the statistic is
    the same, so the difference that matters is what you feed it.

    When `points` is supplied — the scorecard points assigned to each bin — the
    result additionally reports the shift in expected score attributable to
    this feature, which is what actually drives a portfolio's score
    distribution.

    Args:
        points: Score points per bin, aligned with the bin order of the spec
            (including the trailing missing bin). Optional.

    Returns:
        A :class:`DriftResult`. If `points` was given, its ``table`` carries an
        extra ``points`` column and a ``score_impact`` column whose sum is the
        expected point movement.
    """
    spec = bins if bins is not None else fit_bins(expected, n_bins=n_bins)
    result = psi_from_counts(
        spec.counts(expected),
        spec.counts(actual),
        alpha=alpha,
        labels=spec.labels,
    )

    if points is None:
        return result

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape != (spec.n_bins,):
        raise ValueError(
            f"points must have one entry per bin ({spec.n_bins}), got {point_array.shape[0]}"
        )

    table = result.table.copy()
    table["points"] = point_array
    table["score_impact"] = (table["actual_pct"] - table["expected_pct"]) * point_array

    return DriftResult(
        value=result.value,
        table=table,
        n_expected=result.n_expected,
        n_actual=result.n_actual,
    )
