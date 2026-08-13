"""Weight of Evidence encoding and Information Value.

WOE replaces a binned feature with the log-odds of the target within each bin,
which linearises a feature against the log-odds of the outcome. That is why
scorecards are built on WOE-encoded inputs and a logistic regression: the
result stays monotone, explainable and directly convertible to points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .binning import BinSpec, fit_bins

__all__ = ["WOEEncoder", "information_value"]


def _validate_binary_target(y: pd.Series | npt.NDArray[Any] | list[int]) -> npt.NDArray[Any]:
    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {target.shape}")
    unique = pd.unique(pd.Series(target).dropna())
    unexpected = set(np.asarray(unique).tolist()) - {0, 1}
    if unexpected:
        found = sorted(map(str, unexpected))
        raise ValueError(f"y must be binary 0/1; found unexpected values: {found}")
    return target.astype(np.int64)


def information_value(
    x: pd.Series | npt.NDArray[Any] | list[object],
    y: pd.Series | npt.NDArray[Any] | list[int],
    *,
    n_bins: int = 10,
    alpha: float = 0.5,
) -> float:
    """Information Value of a single feature against a binary target.

    Rough convention: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium,
    0.3-0.5 strong, >0.5 suspiciously strong — a very high IV usually means
    leakage rather than a great feature, so check it before celebrating.
    """
    encoder = WOEEncoder(n_bins=n_bins, alpha=alpha)
    encoder.fit(pd.DataFrame({"feature": np.asarray(x)}), y)
    return float(encoder.information_values_["feature"])


class WOEEncoder:
    """Fit-once Weight of Evidence encoder for binary-target problems.

    Follows the scikit-learn transformer protocol (``fit`` / ``transform`` /
    ``fit_transform``) so it drops into a ``Pipeline``, but it does not import
    scikit-learn — the dependency is not needed for the arithmetic.

    Sign convention: ``WOE = ln(P(bin | y=0) / P(bin | y=1))``. A **positive**
    WOE therefore means the bin is over-represented among non-events (good
    accounts). Both conventions exist in the literature and the opposite one
    flips the sign of every coefficient, so it is stated explicitly here.

    Args:
        n_bins: Quantile bins per numeric feature.
        alpha: Additive smoothing on the per-bin event/non-event counts. Bins
            that are pure — all events or none — give an infinite WOE without
            it. ``0.5`` is the Jeffreys prior.

    Attributes:
        bin_specs_: Fitted :class:`~driftkit.binning.BinSpec` per column.
        woe_maps_: Per-column array of WOE values indexed by bin code.
        information_values_: Per-column IV.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> X = pd.DataFrame({"fico": rng.normal(700, 50, 1000)})
        >>> y = (rng.random(1000) < 1 / (1 + np.exp((X["fico"] - 700) / 40))).astype(int)
        >>> encoder = WOEEncoder(n_bins=5).fit(X, y)
        >>> encoder.transform(X).shape
        (1000, 1)
    """

    def __init__(self, *, n_bins: int = 10, alpha: float = 0.5) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_specs_: dict[str, BinSpec] = {}
        self.woe_maps_: dict[str, npt.NDArray[Any]] = {}
        self.information_values_: dict[str, float] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | npt.NDArray[Any] | list[int]) -> WOEEncoder:
        """Learn bins and WOE values from the training sample."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        target = _validate_binary_target(y)
        if len(target) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(target)}")

        n_non_events = int((target == 0).sum())
        n_events = int((target == 1).sum())
        if n_non_events == 0 or n_events == 0:
            raise ValueError("y must contain both classes to compute WOE")

        self.bin_specs_.clear()
        self.woe_maps_.clear()
        self.information_values_.clear()

        for column in X.columns:
            spec = fit_bins(X[column], n_bins=self.n_bins)
            codes = spec.assign(X[column])

            non_event_counts = np.bincount(codes[target == 0], minlength=spec.n_bins)
            event_counts = np.bincount(codes[target == 1], minlength=spec.n_bins)

            denominator = self.alpha * spec.n_bins
            non_event_pct = (non_event_counts + self.alpha) / (n_non_events + denominator)
            event_pct = (event_counts + self.alpha) / (n_events + denominator)

            woe = np.log(non_event_pct / event_pct)

            self.bin_specs_[str(column)] = spec
            self.woe_maps_[str(column)] = woe
            self.information_values_[str(column)] = float(((non_event_pct - event_pct) * woe).sum())

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace each fitted column with its WOE encoding."""
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        missing = [name for name in self.bin_specs_ if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing columns seen during fit: {missing}")

        encoded = {
            name: self.woe_maps_[name][spec.assign(X[name])]
            for name, spec in self.bin_specs_.items()
        }
        return pd.DataFrame(encoded, index=X.index)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | npt.NDArray[Any] | list[int]
    ) -> pd.DataFrame:
        """Fit on `X`, `y` and return the encoded frame."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> npt.NDArray[Any]:
        """Output column names, for scikit-learn pipeline introspection."""
        self._check_fitted()
        return np.asarray(list(self.bin_specs_), dtype=object)

    def summary(self) -> pd.DataFrame:
        """Per-bin WOE table across all fitted columns, sorted by IV."""
        self._check_fitted()
        frames = [
            pd.DataFrame(
                {
                    "feature": name,
                    "bin": list(spec.labels),
                    "woe": self.woe_maps_[name],
                    "iv": self.information_values_[name],
                }
            )
            for name, spec in self.bin_specs_.items()
        ]
        return pd.concat(frames, ignore_index=True).sort_values(
            ["iv", "feature"], ascending=[False, True]
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("WOEEncoder is not fitted yet; call fit() first")
