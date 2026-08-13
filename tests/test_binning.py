from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftkit.binning import MISSING_LABEL, fit_bins


def test_outer_edges_are_infinite():
    spec = fit_bins(np.arange(100.0), n_bins=5)
    assert spec.edges is not None
    assert np.isneginf(spec.edges[0])
    assert np.isposinf(spec.edges[-1])


def test_values_outside_reference_range_still_bin():
    """Out-of-sample extremes must land in the end bins, not vanish."""
    spec = fit_bins(np.arange(100.0), n_bins=5)
    codes = spec.assign([-1e9, 1e9])
    assert codes[0] == 0
    assert codes[1] == spec.n_bins - 2  # last real bin, before the missing bin
    assert spec.missing_code not in codes


def test_nan_routed_to_missing_bin():
    spec = fit_bins(np.arange(100.0), n_bins=4)
    codes = spec.assign([np.nan, 50.0, None])
    assert codes[0] == spec.missing_code
    assert codes[2] == spec.missing_code
    assert codes[1] != spec.missing_code


def test_counts_conserve_total():
    spec = fit_bins(np.arange(100.0), n_bins=7)
    sample = [1.0, 2.0, np.nan, 99.0, 1e6]
    assert spec.counts(sample).sum() == len(sample)


def test_labels_end_with_missing():
    spec = fit_bins(np.arange(50.0), n_bins=5)
    assert spec.labels[-1] == MISSING_LABEL


def test_tied_feature_collapses_duplicate_edges():
    """A mostly-constant feature should yield fewer bins, not empty duplicates."""
    values = np.array([0.0] * 950 + list(range(1, 51)), dtype=float)
    spec = fit_bins(values, n_bins=10)
    assert spec.edges is not None
    assert len(np.unique(spec.edges)) == len(spec.edges)


def test_categorical_unseen_category_is_missing():
    spec = fit_bins(pd.Series(["a", "b", "c"]))
    codes = spec.assign(["a", "zzz"])
    assert codes[0] == 0
    assert codes[1] == spec.missing_code


def test_uniform_strategy_produces_equal_widths():
    spec = fit_bins(np.arange(101.0), n_bins=4, strategy="uniform")
    assert spec.edges is not None
    interior = spec.edges[1:-1]
    widths = np.diff(interior)
    assert np.allclose(widths, widths[0])


def test_rejects_too_few_bins():
    with pytest.raises(ValueError, match="at least 2"):
        fit_bins(np.arange(10.0), n_bins=1)


def test_rejects_all_missing_reference():
    with pytest.raises(ValueError, match="no non-missing values"):
        fit_bins(np.array([np.nan, np.nan]))


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        fit_bins(np.zeros((4, 4)))
