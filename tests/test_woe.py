from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftkit import WOEEncoder, information_value


@pytest.fixture
def credit_sample() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic retail-credit data where low FICO genuinely predicts default.

    Sized at 50k deliberately. IV is biased upward on small samples — each
    bin's empirical event rate wanders from the base rate by chance, and the
    bias scales roughly with n_bins/n. At 5k rows a pure-noise feature scores
    ~0.026, which would clear the conventional "<0.02 is useless" band and
    make the assertion below meaningless.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    fico = rng.normal(700, 50, n)
    log_odds = -0.02 * (fico - 680) - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return pd.DataFrame({"fico": fico, "noise": rng.normal(0, 1, n)}), y


def test_predictive_feature_beats_noise(credit_sample):
    X, y = credit_sample
    encoder = WOEEncoder(n_bins=10).fit(X, y)
    assert encoder.information_values_["fico"] > encoder.information_values_["noise"]
    assert encoder.information_values_["noise"] < 0.02


def test_woe_is_monotone_for_monotone_risk(credit_sample):
    """Higher FICO means fewer defaults, so WOE must rise across bins."""
    X, y = credit_sample
    encoder = WOEEncoder(n_bins=10).fit(X[["fico"]], y)
    woe = encoder.woe_maps_["fico"][:-1]  # drop the empty missing bin
    increases = np.diff(woe) > 0
    assert increases.mean() > 0.8


def test_transform_shape_and_columns(credit_sample):
    X, y = credit_sample
    encoded = WOEEncoder(n_bins=5).fit_transform(X, y)
    assert encoded.shape == X.shape
    assert list(encoded.columns) == list(X.columns)
    assert np.all(np.isfinite(encoded.to_numpy()))


def test_transform_preserves_index(credit_sample):
    X, y = credit_sample
    X = X.set_axis(pd.RangeIndex(100, 100 + len(X)))
    encoded = WOEEncoder(n_bins=5).fit(X, y).transform(X)
    assert encoded.index.equals(X.index)


def test_pure_bin_stays_finite():
    """Smoothing must keep an all-event bin from producing infinite WOE."""
    X = pd.DataFrame({"f": [0.0] * 100 + [1.0] * 100})
    y = np.array([0] * 100 + [1] * 100)
    encoder = WOEEncoder(n_bins=2).fit(X, y)
    assert np.all(np.isfinite(encoder.woe_maps_["f"]))


def test_unfitted_transform_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        WOEEncoder().transform(pd.DataFrame({"a": [1.0]}))


def test_missing_column_at_transform_raises(credit_sample):
    X, y = credit_sample
    encoder = WOEEncoder(n_bins=5).fit(X, y)
    with pytest.raises(ValueError, match="missing columns"):
        encoder.transform(X[["fico"]])


def test_single_class_target_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="both classes"):
        WOEEncoder(n_bins=2).fit(X, np.zeros(4, dtype=int))


def test_non_binary_target_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="binary"):
        WOEEncoder(n_bins=2).fit(X, np.array([0, 1, 7]))


def test_length_mismatch_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="rows but y has"):
        WOEEncoder(n_bins=2).fit(X, np.array([0, 1]))


def test_non_dataframe_input_raises():
    with pytest.raises(TypeError, match="DataFrame"):
        WOEEncoder().fit(np.zeros((3, 2)), np.array([0, 1, 0]))


def test_summary_sorted_by_iv(credit_sample):
    X, y = credit_sample
    summary = WOEEncoder(n_bins=5).fit(X, y).summary()
    ivs = summary.drop_duplicates("feature")["iv"].to_numpy()
    assert np.all(np.diff(ivs) <= 0)


def test_information_value_helper_matches_encoder(credit_sample):
    X, y = credit_sample
    encoder = WOEEncoder(n_bins=8).fit(X[["fico"]], y)
    assert information_value(X["fico"], y, n_bins=8) == pytest.approx(
        encoder.information_values_["fico"]
    )


def test_get_feature_names_out(credit_sample):
    X, y = credit_sample
    encoder = WOEEncoder(n_bins=5).fit(X, y)
    assert list(encoder.get_feature_names_out()) == list(X.columns)
