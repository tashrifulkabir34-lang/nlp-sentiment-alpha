"""Tests for data_generator module."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import pytest
from src.data_generator import build_dataset, _generate_tickers


def test_tickers_unique():
    tickers = _generate_tickers(30, seed=1)
    assert len(tickers) == len(set(tickers))

def test_tickers_uppercase():
    for t in _generate_tickers(10):
        assert t.isupper()

def test_build_dataset_shapes():
    news, prices = build_dataset(n_tickers=10, start="2022-01-01", end="2022-06-30", seed=0)
    assert prices.shape[1] == 10
    assert len(news) > 0

def test_news_columns():
    news, _ = build_dataset(n_tickers=5, start="2022-01-01", end="2022-03-31")
    for col in ["date", "ticker", "headline", "true_sentiment_bucket"]:
        assert col in news.columns

def test_prices_positive():
    _, prices = build_dataset(n_tickers=5, start="2022-01-01", end="2022-06-30")
    assert (prices > 0).all().all()

def test_sentiment_buckets_valid():
    news, _ = build_dataset(n_tickers=5, start="2022-01-01", end="2022-03-31")
    assert set(news["true_sentiment_bucket"].unique()).issubset({"positive", "negative", "neutral"})

def test_reproducibility():
    n1, p1 = build_dataset(seed=42, n_tickers=10)
    n2, p2 = build_dataset(seed=42, n_tickers=10)
    pd.testing.assert_frame_equal(p1, p2)

def test_date_range():
    _, prices = build_dataset(n_tickers=5, start="2022-01-01", end="2022-12-31")
    assert prices.index[0] >= pd.Timestamp("2022-01-01")
    assert prices.index[-1] <= pd.Timestamp("2022-12-31")
