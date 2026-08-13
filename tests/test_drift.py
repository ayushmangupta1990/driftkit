from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from driftkit import fit_bins, psi, psi_from_counts
from driftkit.drift import csi, interpret_psi


def test_identical_samples_give_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5_000)
    assert psi(sample, sample).value == pytest.approx(0.0, abs=1e-12)


def test_shifted_distribution_flagged_significant():
    rng = np.random.default_rng(0)
    reference = rng.normal(700, 50, 20_000)
    current = rng.normal(640, 60, 20_000)
    result = psi(reference, current)
    assert result.value > 0.25
    assert result.interpretation == "significant shift"


def test_small_shift_stays_stable():
    rng = np.random.default_rng(1)
    reference = rng.normal(700, 50, 50_000)
    current = rng.normal(701, 50, 50_000)
    assert psi(reference, current).value < 0.10


def test_rebinning_the_comparison_sample_hides_drift():
    """Guards the library's core design decision.

    Re-fitting quantile bins on each sample makes both histograms uniform, so
    PSI reads ~0 on a population that plainly moved. Freezing the reference
    bins is what makes the metric detect it.
    """
    rng = np.random.default_rng(0)
    reference = rng.normal(700, 50, 20_000)
    current = rng.normal(600, 50, 20_000)

    correct = psi(reference, current).value

    naive = psi_from_counts(
        fit_bins(reference, n_bins=10).counts(reference),
        fit_bins(current, n_bins=10).counts(current),
    ).value

    assert correct > 0.25
    assert naive < 0.01
    assert correct > naive * 50


def test_contributions_sum_to_total():
    rng = np.random.default_rng(2)
    result = psi(rng.normal(0, 1, 5_000), rng.normal(0.5, 1, 5_000))
    assert result.table["contribution"].sum() == pytest.approx(result.value)


def test_top_bins_ordered_by_contribution():
    rng = np.random.default_rng(3)
    result = psi(rng.normal(0, 1, 5_000), rng.normal(1.5, 1, 5_000))
    contributions = result.top_bins["contribution"].to_numpy()
    assert np.all(np.diff(contributions) <= 0)


def test_empty_bin_raises_without_smoothing():
    with pytest.raises(ValueError, match="empty bin"):
        psi_from_counts(np.array([10.0, 0.0]), np.array([5.0, 5.0]), alpha=0.0)


def test_empty_bin_survives_with_smoothing():
    result = psi_from_counts(np.array([10.0, 0.0]), np.array([5.0, 5.0]), alpha=0.5)
    assert np.isfinite(result.value)
    assert result.value > 0


def test_mismatched_count_vectors_rejected():
    with pytest.raises(ValueError, match="must align"):
        psi_from_counts(np.array([1.0, 2.0]), np.array([1.0]))


def test_empty_sample_rejected():
    with pytest.raises(ValueError, match="at least one observation"):
        psi_from_counts(np.array([0.0, 0.0]), np.array([1.0, 1.0]), alpha=0.0)


def test_frozen_bins_reused_across_periods():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 10_000)
    spec = fit_bins(reference, n_bins=8)
    values = [psi(reference, rng.normal(shift, 1, 10_000), bins=spec).value for shift in (0, 1, 2)]
    assert values[0] < values[1] < values[2]


def test_categorical_psi():
    reference = ["a"] * 800 + ["b"] * 200
    current = ["a"] * 400 + ["b"] * 600
    assert psi(pd.Series(reference), pd.Series(current)).value > 0.25


def test_missing_data_spike_is_detected():
    """A feed that starts returning nulls must register as drift."""
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 10_000)
    current = rng.normal(0, 1, 10_000).astype(object)
    current[:3_000] = np.nan
    assert psi(reference, pd.Series(current)).value > 0.25


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "stable"), (0.09, "stable"), (0.10, "moderate shift"), (0.30, "significant shift")],
)
def test_interpretation_bands(value, expected):
    assert interpret_psi(value) == expected


@given(
    expected=st.lists(st.integers(min_value=0, max_value=5_000), min_size=2, max_size=12),
    actual=st.lists(st.integers(min_value=0, max_value=5_000), min_size=2, max_size=12),
)
@settings(max_examples=200, deadline=None)
def test_psi_is_non_negative_and_symmetric(expected, actual):
    """PSI is a symmetrised KL divergence, so both properties must hold."""
    size = min(len(expected), len(actual))
    left = np.array(expected[:size], dtype=float)
    right = np.array(actual[:size], dtype=float)
    if left.sum() == 0 or right.sum() == 0:
        return

    forward = psi_from_counts(left, right).value
    backward = psi_from_counts(right, left).value

    assert forward >= -1e-12
    assert forward == pytest.approx(backward, rel=1e-9, abs=1e-12)


def test_csi_score_impact_reported():
    rng = np.random.default_rng(6)
    reference = rng.normal(0, 1, 10_000)
    current = rng.normal(1, 1, 10_000)
    spec = fit_bins(reference, n_bins=5)
    points = np.arange(spec.n_bins, dtype=float) * 10

    result = csi(reference, current, points=points, bins=spec)

    assert "score_impact" in result.table.columns
    assert result.table["score_impact"].sum() > 0


def test_csi_rejects_wrong_points_length():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 1_000)
    with pytest.raises(ValueError, match="one entry per bin"):
        csi(reference, reference, points=np.array([1.0, 2.0]), n_bins=5)
