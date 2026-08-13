"""Synthetic retail-credit data, for examples and tests.

The generator is deliberately simple and fully specified in code: default
propensity is a logistic function of a handful of interpretable drivers. That
makes it useful for demonstrating the metrics, because the ground truth is
known — you can shift a driver by a known amount and check that PSI notices.

Nothing here is derived from real lending data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_credit_data"]


def make_credit_data(
    n_samples: int = 10_000,
    *,
    drift: bool = False,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Generate a synthetic retail-credit portfolio.

    Args:
        n_samples: Number of accounts.
        drift: When ``True``, shift the population as if a macroeconomic
            downturn had occurred — lower FICO, higher debt-to-income, higher
            revolving utilisation, more delinquencies and worse call-centre
            sentiment. The *relationship* between drivers and default is left
            unchanged, so this is pure covariate shift: the inputs move, the
            model's learned mapping does not become wrong.
        seed: Seed for reproducibility. ``None`` draws fresh entropy.

    Returns:
        A frame with columns ``fico_score``, ``debt_to_income``,
        ``revol_util``, ``delinq_2yrs``, ``text_sentiment``, ``age``,
        and the binary target ``default_event``.

    Example:
        >>> reference = make_credit_data(1_000, seed=0)
        >>> current = make_credit_data(1_000, drift=True, seed=1)
        >>> bool(reference["fico_score"].mean() > current["fico_score"].mean())
        True
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, size=n_samples)

    if drift:
        fico = rng.normal(660, 60, size=n_samples)
        dti = rng.beta(3, 4, size=n_samples) * 0.9
        revol_util = rng.beta(3, 2, size=n_samples)
        delinq = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.70, 0.18, 0.07, 0.05])
        sentiment = rng.normal(-0.1, 0.5, size=n_samples)
    else:
        fico = rng.normal(700, 50, size=n_samples)
        dti = rng.beta(2, 5, size=n_samples) * 0.8
        revol_util = rng.beta(2, 3, size=n_samples)
        delinq = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.10, 0.03, 0.02])
        sentiment = rng.normal(0.2, 0.4, size=n_samples)

    fico = np.clip(fico, 300, 850)
    dti = np.clip(dti, 0.0, 1.0)
    revol_util = np.clip(revol_util, 0.0, 1.5)
    sentiment = np.clip(sentiment, -1.0, 1.0)

    # Identical in both scenarios: only the inputs shift, not their meaning.
    log_odds = (
        -0.015 * (fico - 680) + 3.5 * dti + 2.0 * revol_util + 0.6 * delinq - 1.5 * sentiment - 3.8
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    default_event = (rng.random(n_samples) < probability).astype(np.int64)

    return pd.DataFrame(
        {
            "fico_score": fico,
            "debt_to_income": dti,
            "revol_util": revol_util,
            "delinq_2yrs": delinq,
            "text_sentiment": sentiment,
            "age": age,
            "default_event": default_event,
        }
    )
