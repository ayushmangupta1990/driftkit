from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftkit import fit_bins, monitor


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 20_000),
            "drifted": rng.normal(0, 1, 20_000),
            "segment": rng.choice(["a", "b", "c"], 20_000),
        }
    )
    current = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 20_000),
            "drifted": rng.normal(2, 1, 20_000),
            "segment": rng.choice(["a", "b", "c"], 20_000),
        }
    )
    return reference, current


def test_most_drifted_feature_ranked_first(frames):
    reference, current = frames
    report = monitor(reference, current)
    assert report.summary.iloc[0]["feature"] == "drifted"
    assert report.unstable["feature"].tolist() == ["drifted"]


def test_summary_sorted_descending(frames):
    reference, current = frames
    psi_values = monitor(reference, current).summary["psi"].to_numpy()
    assert np.all(np.diff(psi_values) <= 0)


def test_results_expose_per_bin_detail(frames):
    reference, current = frames
    report = monitor(reference, current)
    assert set(report.results) == {"stable", "drifted", "segment"}
    assert "contribution" in report.results["drifted"].table.columns


def test_markdown_renders_all_rows(frames):
    reference, current = frames
    markdown = monitor(reference, current).to_markdown()
    assert markdown.count("\n") == 4  # header + separator + 3 features
    assert "drifted" in markdown


def test_markdown_limit(frames):
    reference, current = frames
    assert monitor(reference, current).to_markdown(limit=1).count("\n") == 2


def test_only_shared_columns_compared(frames):
    reference, current = frames
    report = monitor(reference, current.drop(columns=["segment"]))
    assert set(report.results) == {"stable", "drifted"}


def test_explicit_columns_respected(frames):
    reference, current = frames
    report = monitor(reference, current, columns=["stable"])
    assert set(report.results) == {"stable"}


def test_frozen_specs_shared_across_periods(frames):
    reference, current = frames
    specs = {name: fit_bins(reference[name], n_bins=10) for name in reference.columns}
    first = monitor(reference, current, bin_specs=specs)
    second = monitor(reference, current, bin_specs=specs)
    assert first.summary["psi"].tolist() == second.summary["psi"].tolist()


def test_disjoint_schemas_rejected():
    with pytest.raises(ValueError, match="share no columns"):
        monitor(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))


def test_unknown_column_rejected(frames):
    reference, current = frames
    with pytest.raises(ValueError, match="missing from reference"):
        monitor(reference, current, columns=["nope"])
