"""Binning primitives shared by the drift metrics.

The single most common way to get PSI wrong is to re-bin the comparison sample.
Bins must be *learned once* on the reference population and then applied
unchanged to every later sample; otherwise each side gets its own quantiles,
both histograms come out uniform by construction, and PSI collapses toward zero
no matter how far the population has actually moved.

`BinSpec` exists to make that mistake hard: you `fit` it on the reference data
and it is frozen thereafter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["MISSING_LABEL", "BinSpec", "fit_bins"]

MISSING_LABEL = "__missing__"
"""Label of the dedicated bin that collects NaN/None values."""

Strategy = Literal["quantile", "uniform"]


def _as_1d_array(x: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    """Return `x` as a 1-D numpy array without copying when avoidable."""
    if isinstance(x, pd.Series):
        return x.to_numpy()
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D input, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class BinSpec:
    """A frozen binning scheme learned from a reference sample.

    Attributes:
        kind: ``"numeric"`` for interval bins, ``"categorical"`` for value bins.
        edges: Bin edges for numeric features. The outer edges are always
            ``-inf`` and ``+inf`` so that out-of-sample values below or above
            anything seen in the reference set still land in a bin instead of
            silently becoming NaN.
        categories: Ordered category values for categorical features.
        labels: Human-readable label per bin, aligned with the integer codes
            produced by :meth:`assign`. The final label is always
            :data:`MISSING_LABEL`.
    """

    kind: Literal["numeric", "categorical"]
    edges: np.ndarray | None = None
    categories: tuple[object, ...] | None = None
    labels: tuple[str, ...] = field(default=())

    @property
    def n_bins(self) -> int:
        """Number of bins, including the trailing missing-value bin."""
        return len(self.labels)

    @property
    def missing_code(self) -> int:
        """Integer code of the missing-value bin."""
        return self.n_bins - 1

    def assign(self, x: pd.Series | np.ndarray | list[object]) -> np.ndarray:
        """Map values to integer bin codes in ``[0, n_bins)``.

        NaN, None and — for categorical specs — any category unseen in the
        reference sample are routed to the missing bin rather than dropped.
        Dropping them would understate drift precisely when a feed breaks,
        which is the case monitoring exists to catch.
        """
        arr = _as_1d_array(x)

        if self.kind == "numeric":
            assert self.edges is not None
            values = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
            missing = np.isnan(values)
            # np.digitize with the -inf/+inf sentinels yields codes in
            # [1, len(edges)-1]; shift down so codes start at 0.
            numeric_codes: np.ndarray = np.digitize(values, self.edges[1:-1], right=False).astype(
                np.int64
            )
            numeric_codes[missing] = self.missing_code
            return numeric_codes

        assert self.categories is not None
        lookup = {value: index for index, value in enumerate(self.categories)}
        codes = np.full(len(arr), self.missing_code, dtype=np.int64)
        for position, value in enumerate(arr):
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            mapped = lookup.get(value)
            if mapped is not None:
                codes[position] = mapped
        return codes

    def counts(self, x: pd.Series | np.ndarray | list[object]) -> np.ndarray:
        """Return the per-bin count vector for `x`."""
        codes = self.assign(x)
        return np.bincount(codes, minlength=self.n_bins).astype(np.float64)


def fit_bins(
    x: pd.Series | np.ndarray | list[object],
    n_bins: int = 10,
    *,
    strategy: Strategy = "quantile",
    categorical: bool | None = None,
) -> BinSpec:
    """Learn a :class:`BinSpec` from a reference sample.

    Args:
        x: Reference values. NaNs are excluded from edge computation but get
            their own bin.
        n_bins: Requested number of bins for numeric features. Duplicate
            quantile edges are collapsed, so a heavily tied feature may yield
            fewer bins than requested — this is deliberate: emitting empty
            duplicate bins would inflate PSI with pure noise.
        strategy: ``"quantile"`` for equal-frequency bins (the default, and
            what regulators expect for scorecard monitoring) or ``"uniform"``
            for equal-width bins.
        categorical: Force categorical treatment. When ``None``, non-numeric
            dtypes are treated as categorical.

    Raises:
        ValueError: If `n_bins` is below 2 or `x` contains no usable values.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")

    arr = _as_1d_array(x)
    series = pd.Series(arr)

    if categorical is None:
        categorical = not pd.api.types.is_numeric_dtype(series)

    if categorical:
        present = series.dropna()
        if present.empty:
            raise ValueError("cannot fit bins: reference sample has no non-missing values")
        categories = tuple(pd.unique(present))
        category_labels = (*(str(value) for value in categories), MISSING_LABEL)
        return BinSpec(kind="categorical", categories=categories, labels=category_labels)

    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("cannot fit bins: reference sample has no non-missing values")

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        raw = np.quantile(values, quantiles)
    else:
        raw = np.linspace(values.min(), values.max(), n_bins + 1)

    interior = np.unique(raw[1:-1])
    edges = np.concatenate(([-np.inf], interior, [np.inf]))

    labels: list[str] = []
    for lower, upper in pairwise(edges):
        lower_text = "-inf" if np.isneginf(lower) else f"{lower:.6g}"
        upper_text = "inf" if np.isposinf(upper) else f"{upper:.6g}"
        labels.append(f"[{lower_text}, {upper_text})")
    labels.append(MISSING_LABEL)

    return BinSpec(kind="numeric", edges=edges, labels=tuple(labels))
