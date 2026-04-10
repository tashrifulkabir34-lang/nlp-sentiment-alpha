"""Tests for signal_constructor module."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pytest
from src.data_generator import build_dataset
from src.sentiment_scorer import SentimentScorer
from src.signal_constructor import (
    aggregate_daily_sentiment,
    cross_sectional_zscore,
    apply_momentum_decay,
    compute_forward_returns,
    compute_ic_series,
    compute_ic_summary,
    compute_quantile_returns,
)


@pytest.fixture(scope="module")
def small_dataset():
    news, prices = build_dataset(n_tickers=10, start="2022-01-01", end="2022-06-30", seed=7)
    scorer = SentimentScorer(vader_weight=1.0, use_finbert=False)
    scored = scorer.score_dataframe(news)
    return scored, prices


def test_aggregate_daily_sentiment(small_dataset):
    scored, _ = small_dataset
    daily = aggregate_daily_sentiment(scored)
    assert "sentiment_score" in daily.columns
    assert daily.index.nlevels == 2


def test_zscore_range(small_dataset):
    scored, _ = small_dataset
    daily = aggregate_daily_sentiment(scored)
    normed = cross_sectional_zscore(daily)
    assert "factor_zscore" in normed.columns
    assert normed["factor_zscore"].abs().max() <= 3.01  # winsorised


def test_momentum_decay(small_dataset):
    scored, _ = small_dataset
    daily = aggregate_daily_sentiment(scored)
    normed = cross_sectional_zscore(daily)
    decayed = apply_momentum_decay(normed, halflife_days=3)
    assert "factor_decayed" in decayed.columns


def test_forward_returns_shape(small_dataset):
    _, prices = small_dataset
    fwd = compute_forward_returns(prices, periods=[1, 5])
    assert "1D" in fwd.columns
    assert "5D" in fwd.columns


def test_ic_series_not_all_nan(small_dataset):
    scored, prices = small_dataset
    daily = aggregate_daily_sentiment(scored)
    normed = cross_sectional_zscore(daily)
    factor = apply_momentum_decay(normed)
    fwd = compute_forward_returns(prices, periods=[1, 5])
    ic = compute_ic_series(factor, fwd, periods=[1, 5])
    # At least some IC values should be non-NaN
    assert ic["IC_1D"].dropna().shape[0] > 0


def test_ic_summary_columns(small_dataset):
    scored, prices = small_dataset
    daily = aggregate_daily_sentiment(scored)
    normed = cross_sectional_zscore(daily)
    factor = apply_momentum_decay(normed)
    fwd = compute_forward_returns(prices, periods=[1, 5])
    ic = compute_ic_series(factor, fwd, periods=[1, 5])
    summary = compute_ic_summary(ic)
    for col in ["mean_IC", "std_IC", "t_stat", "ICIR", "hit_rate"]:
        assert col in summary.columns


def test_ic_hit_rate_range(small_dataset):
    scored, prices = small_dataset
    daily = aggregate_daily_sentiment(scored)
    normed = cross_sectional_zscore(daily)
    factor = apply_momentum_decay(normed)
    fwd = compute_forward_returns(prices, periods=[1, 5])
    ic = compute_ic_series(factor, fwd, periods=[1, 5])
    summary = compute_ic_summary(ic)
    assert (summary["hit_rate"] >= 0).all()
    assert (summary["hit_rate"] <= 1).all()
