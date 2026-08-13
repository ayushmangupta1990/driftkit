from __future__ import annotations

import numpy as np
import pytest

from driftkit import brier_decomposition, calibration_report, expected_calibration_error


@pytest.fixture
def perfectly_calibrated() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    prob = rng.random(50_000)
    return (rng.random(50_000) < prob).astype(int), prob


def test_calibrated_model_has_low_error(perfectly_calibrated):
    y, p = perfectly_calibrated
    report = calibration_report(y, p)
    assert report.ece < 0.01
    assert abs(report.bias) < 0.01


def test_overconfident_model_shows_bias(perfectly_calibrated):
    """Inflating every probability must surface as positive bias."""
    y, p = perfectly_calibrated
    report = calibration_report(y, np.clip(p + 0.15, 0, 1))
    assert report.bias > 0.10
    assert report.ece > 0.10


def test_murphy_decomposition_identity(perfectly_calibrated):
    y, p = perfectly_calibrated
    report = calibration_report(y, p, n_bins=20)
    reliability, resolution, uncertainty = brier_decomposition(y, p, n_bins=20)
    assert reliability - resolution + uncertainty == pytest.approx(report.brier, abs=1e-3)


def test_uncertainty_is_base_rate_variance():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    report = calibration_report(y, p)
    assert report.uncertainty == pytest.approx(0.25 * 0.75)


def test_probability_of_one_does_not_overflow_bins():
    y = np.array([1, 1, 0])
    p = np.array([1.0, 1.0, 0.0])
    report = calibration_report(y, p, n_bins=10)
    assert len(report.table) == 10
    assert report.table["count"].sum() == 3


def test_empty_bins_reported_as_nan():
    y = np.array([0, 1])
    p = np.array([0.05, 0.06])
    table = calibration_report(y, p, n_bins=10).table
    assert table["count"].iloc[5] == 0
    assert np.isnan(table["gap"].iloc[5])


def test_ece_helper_matches_report(perfectly_calibrated):
    y, p = perfectly_calibrated
    assert expected_calibration_error(y, p) == pytest.approx(calibration_report(y, p).ece)


def test_probability_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_report(np.array([0, 1]), np.array([0.5, 1.5]))


def test_non_binary_truth_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        calibration_report(np.array([0, 2]), np.array([0.5, 0.5]))


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="must align"):
        calibration_report(np.array([0, 1]), np.array([0.5]))


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        calibration_report(np.array([]), np.array([]))
