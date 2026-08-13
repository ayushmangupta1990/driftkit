from __future__ import annotations

import numpy as np
import pytest

from driftkit import make_credit_data, monitor, psi

EXPECTED_COLUMNS = {
    "fico_score",
    "debt_to_income",
    "revol_util",
    "delinq_2yrs",
    "text_sentiment",
    "age",
    "default_event",
}


def test_schema_and_shape():
    frame = make_credit_data(500, seed=0)
    assert len(frame) == 500
    assert set(frame.columns) == EXPECTED_COLUMNS


def test_seed_is_reproducible():
    left = make_credit_data(200, seed=7)
    right = make_credit_data(200, seed=7)
    assert left.equals(right)


def test_different_seeds_differ():
    left = make_credit_data(200, seed=1)
    right = make_credit_data(200, seed=2)
    assert not left.equals(right)


def test_values_stay_in_valid_ranges():
    frame = make_credit_data(5_000, seed=0)
    assert frame["fico_score"].between(300, 850).all()
    assert frame["debt_to_income"].between(0, 1).all()
    assert frame["revol_util"].between(0, 1.5).all()
    assert frame["text_sentiment"].between(-1, 1).all()
    assert frame["default_event"].isin([0, 1]).all()


def test_both_classes_present():
    frame = make_credit_data(5_000, seed=0)
    assert 0 < frame["default_event"].mean() < 1


def test_drift_worsens_the_portfolio():
    baseline = make_credit_data(20_000, seed=0)
    stressed = make_credit_data(20_000, drift=True, seed=1)

    assert stressed["fico_score"].mean() < baseline["fico_score"].mean()
    assert stressed["debt_to_income"].mean() > baseline["debt_to_income"].mean()
    assert stressed["default_event"].mean() > baseline["default_event"].mean()


def test_drift_is_detected_by_psi():
    """The generator's whole purpose: a known shift the metrics must catch."""
    baseline = make_credit_data(20_000, seed=0)
    stressed = make_credit_data(20_000, drift=True, seed=1)

    assert psi(baseline["fico_score"], stressed["fico_score"]).value > 0.10

    report = monitor(
        baseline.drop(columns=["default_event"]),
        stressed.drop(columns=["default_event"]),
    )
    assert "age" not in set(report.unstable["feature"])  # unchanged by design
    assert len(report.unstable) > 0


def test_stable_resample_shows_little_drift():
    """Two draws from the same process must not look like drift."""
    left = make_credit_data(20_000, seed=0)
    right = make_credit_data(20_000, seed=99)
    assert psi(left["fico_score"], right["fico_score"]).value < 0.10


def test_rejects_non_positive_size():
    with pytest.raises(ValueError, match="must be positive"):
        make_credit_data(0)


def test_seed_none_draws_fresh_entropy():
    left = make_credit_data(100, seed=None)
    right = make_credit_data(100, seed=None)
    assert not np.array_equal(left["fico_score"], right["fico_score"])
