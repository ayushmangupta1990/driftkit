"""Whole-frame monitoring: run the per-feature metrics in one call."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .binning import BinSpec, fit_bins
from .drift import DriftResult, interpret_psi, psi_from_counts

__all__ = ["MonitoringReport", "monitor"]


@dataclass(frozen=True)
class MonitoringReport:
    """Per-feature drift across two samples of the same schema.

    Attributes:
        results: Full :class:`~driftkit.drift.DriftResult` per feature, so any
            headline number can be drilled into.
        summary: One row per feature, sorted by PSI descending.
    """

    results: dict[str, DriftResult]
    summary: pd.DataFrame

    @property
    def unstable(self) -> pd.DataFrame:
        """Features whose PSI clears the conventional 0.25 threshold."""
        return self.summary[self.summary["psi"] >= 0.25]

    def to_markdown(self, *, limit: int | None = None) -> str:
        """Render the summary as a markdown table, most-drifted first."""
        frame = self.summary if limit is None else self.summary.head(limit)
        lines = [
            "| feature | psi | interpretation |",
            "| --- | ---: | --- |",
        ]
        lines.extend(
            f"| {row.feature} | {row.psi:.4f} | {row.interpretation} |"
            for row in frame.itertuples()
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MonitoringReport(features={len(self.results)}, unstable={len(self.unstable)})"


def monitor(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    n_bins: int = 10,
    alpha: float = 0.5,
    columns: list[str] | None = None,
    bin_specs: dict[str, BinSpec] | None = None,
) -> MonitoringReport:
    """Compute PSI for every shared column of two frames.

    Args:
        reference: Baseline population. Bins are learned here.
        current: Population to compare against the baseline.
        columns: Restrict to these columns. Defaults to the intersection of
            both frames' columns.
        bin_specs: Pre-fitted specs keyed by column, for monitoring a series
            of periods against one frozen reference. Columns absent from the
            mapping get fresh bins.

    Raises:
        ValueError: If the two frames share no columns, or a requested column
            is missing from either frame.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
        >>> cur = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(2, 1, 5000)})
        >>> report = monitor(ref, cur)
        >>> report.summary.iloc[0]["feature"]
        'b'
    """
    if columns is None:
        columns = [name for name in reference.columns if name in set(current.columns)]
        if not columns:
            raise ValueError("reference and current share no columns to compare")
    else:
        for frame, label in ((reference, "reference"), (current, "current")):
            absent = [name for name in columns if name not in frame.columns]
            if absent:
                raise ValueError(f"columns missing from {label}: {absent}")

    specs = dict(bin_specs or {})
    results: dict[str, DriftResult] = {}

    for column in columns:
        spec = specs.get(str(column)) or fit_bins(reference[column], n_bins=n_bins)
        results[str(column)] = psi_from_counts(
            spec.counts(reference[column]),
            spec.counts(current[column]),
            alpha=alpha,
            labels=spec.labels,
        )

    summary = pd.DataFrame(
        {
            "feature": list(results),
            "psi": [result.value for result in results.values()],
            "interpretation": [interpret_psi(result.value) for result in results.values()],
        }
    ).sort_values("psi", ascending=False, ignore_index=True)

    return MonitoringReport(results=results, summary=summary)
