"""driftkit — correct, dependency-light stability metrics for tabular models.

Quick start:

    >>> import numpy as np
    >>> from driftkit import psi
    >>> rng = np.random.default_rng(0)
    >>> result = psi(rng.normal(700, 50, 10_000), rng.normal(650, 60, 10_000))
    >>> result.interpretation
    'significant shift'
"""

from __future__ import annotations

from .binning import MISSING_LABEL, BinSpec, fit_bins
from .calibration import (
    CalibrationReport,
    brier_decomposition,
    calibration_report,
    expected_calibration_error,
)
from .datasets import make_credit_data
from .drift import DriftResult, csi, interpret_psi, psi, psi_from_counts
from .report import MonitoringReport, monitor
from .woe import WOEEncoder, information_value

__version__ = "0.1.0"

__all__ = [
    "MISSING_LABEL",
    "BinSpec",
    "CalibrationReport",
    "DriftResult",
    "MonitoringReport",
    "WOEEncoder",
    "__version__",
    "brier_decomposition",
    "calibration_report",
    "csi",
    "expected_calibration_error",
    "fit_bins",
    "information_value",
    "interpret_psi",
    "make_credit_data",
    "monitor",
    "psi",
    "psi_from_counts",
]
