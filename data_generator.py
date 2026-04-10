"""
data_generator.py
-----------------
Generates synthetic financial news headlines and equity price data for
backtesting the NLP Sentiment Alpha Signal pipeline.

Design goals:
  - Deterministic given a seed (reproducible research).
  - Realistic sentiment-return correlation with noise (IC ~ 0.03–0.07).
  - Cross-sectional universe of 50 synthetic tickers.
  - Three-year daily panel: 2021-01-04 → 2023-12-29.

Author : EquityXperts / Tasrif
"""

from __future__ import annotations

import random
import string
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Headline templates per sentiment bucket
# ---------------------------------------------------------------------------
_POSITIVE_HEADLINES: list[str] = [
    "{ticker} beats Q{q} earnings estimates by wide margin",
    "{ticker} raises full-year guidance after strong quarter",
    "{ticker} announces record revenue growth of {pct}%",
    "{ticker} expands into new markets, bullish outlook ahead",
    "{ticker} secures landmark {amt}B contract with government",
    "Analysts upgrade {ticker} to Buy on improving fundamentals",
    "{ticker} CEO announces major share buyback programme",
    "{ticker} reports better-than-expected margins and cash flow",
    "{ticker} dividend increased {pct}% — income investors rejoice",
    "{ticker} wins regulatory approval for flagship product",
    "{ticker} partnership with tech giant drives optimism",
    "{ticker} free cash flow hits all-time high",
]

_NEGATIVE_HEADLINES: list[str] = [
    "{ticker} misses Q{q} earnings, shares plunge in after-hours",
    "{ticker} cuts full-year guidance citing macro headwinds",
    "{ticker} faces SEC investigation over accounting irregularities",
    "{ticker} announces mass layoffs amid restructuring plan",
    "{ticker} loses key patent battle, faces royalty liabilities",
    "Analysts downgrade {ticker} to Sell on deteriorating margins",
    "{ticker} CFO resigns unexpectedly, raising governance concerns",
    "{ticker} debt downgraded to junk by rating agencies",
    "{ticker} product recall triggers liability fears",
    "{ticker} revenue misses estimates for third consecutive quarter",
    "{ticker} loses major client contract worth {amt}B",
    "{ticker} cash burn accelerates — solvency risks flagged",
]

_NEUTRAL_HEADLINES: list[str] = [
    "{ticker} reports Q{q} results in line with expectations",
    "{ticker} appoints new board member from industry",
    "{ticker} management hosts investor day, no new guidance",
    "{ticker} files 10-K with SEC — no material changes",
    "{ticker} completes previously announced acquisition",
    "{ticker} to present at upcoming industry conference",
    "{ticker} announces minor organisational restructuring",
    "{ticker} reaffirms existing full-year guidance range",
    "{ticker} releases sustainability report for fiscal year",
    "{ticker} changes CFO in planned leadership transition",
]


def _random_headline(
    ticker: str,
    sentiment_bucket: str,  # 'positive', 'negative', 'neutral'
    rng: np.random.Generator,
) -> str:
    """Return a synthetic headline for *ticker* with the given sentiment."""
    q = rng.integers(1, 5)
    pct = rng.integers(5, 40)
    amt = rng.integers(1, 20)

    if sentiment_bucket == "positive":
        templates = _POSITIVE_HEADLINES
    elif sentiment_bucket == "negative":
        templates = _NEGATIVE_HEADLINES
    else:
        templates = _NEUTRAL_HEADLINES

    idx = rng.integers(0, len(templates))
    return templates[idx].format(ticker=ticker, q=q, pct=pct, amt=amt)


# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

def _generate_tickers(n: int = 50, seed: int = 42) -> list[str]:
    """Generate *n* plausible-looking synthetic ticker symbols."""
    rng = random.Random(seed)
    tickers: set[str] = set()
    while len(tickers) < n:
        length = rng.choice([3, 4])
        tickers.add("".join(rng.choices(string.ascii_uppercase, k=length)))
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Price path simulation
# ---------------------------------------------------------------------------

def _simulate_prices(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    base_vol: float = 0.02,
) -> pd.DataFrame:
    """
    GBM price paths with cross-sectional correlation.

    Returns a DataFrame (dates × tickers) of adjusted close prices.
    """
    n_days = len(dates)
    n_tickers = len(tickers)

    # Common market factor
    market_returns = rng.normal(0.0003, base_vol, size=n_days)

    # Idiosyncratic returns
    idio_returns = rng.normal(0.0, base_vol, size=(n_days, n_tickers))

    # Beta in [0.6, 1.4]
    betas = rng.uniform(0.6, 1.4, size=n_tickers)

    total_returns = market_returns[:, None] * betas[None, :] + idio_returns * 0.8

    # Cumulative price starting at 100
    prices = 100.0 * np.exp(np.cumsum(total_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Headline dataset
# ---------------------------------------------------------------------------

def generate_news_dataset(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    headlines_per_day: int = 3,
    sentiment_ic_signal: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a panel of financial headlines with embedded sentiment signal.

    Each headline is labelled with a sentiment_bucket drawn such that
    positive buckets correlate with future returns (target IC ≈ sentiment_ic_signal).

    Returns
    -------
    pd.DataFrame with columns: [date, ticker, headline, true_sentiment_bucket]
    """
    rows: list[dict] = []

    # Pre-draw latent "true" factor scores per ticker-date
    # These will be mapped to sentiment bucket assignments with noise
    for date in dates:
        chosen_tickers = rng.choice(tickers, size=min(headlines_per_day * 5, len(tickers)), replace=False)
        for ticker in chosen_tickers:
            # Latent signal: partially predictive of returns
            latent = rng.normal(0.0, 1.0)
            noise = rng.normal(0.0, 1.0) * (1 - sentiment_ic_signal) / sentiment_ic_signal

            # Map to bucket with realistic misclassification
            score = latent + noise * 0.3
            if score > 0.5:
                bucket = "positive"
            elif score < -0.5:
                bucket = "negative"
            else:
                bucket = "neutral"

            headline = _random_headline(ticker, bucket, rng)
            rows.append({
                "date": date,
                "ticker": ticker,
                "headline": headline,
                "true_sentiment_bucket": bucket,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    n_tickers: int = 50,
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build synthetic news + prices dataset.

    Returns
    -------
    news_df : pd.DataFrame
        Columns: date, ticker, headline, true_sentiment_bucket
    prices_df : pd.DataFrame
        Index: dates, Columns: tickers (adjusted close prices)
    """
    rng = np.random.default_rng(seed)
    tickers = _generate_tickers(n_tickers, seed=seed)

    # Business day calendar
    dates = pd.bdate_range(start=start, end=end)

    prices_df = _simulate_prices(tickers, dates, rng)
    news_df = generate_news_dataset(tickers, dates, rng)

    return news_df, prices_df


if __name__ == "__main__":
    news, prices = build_dataset()
    print(f"News rows   : {len(news):,}")
    print(f"Tickers     : {prices.shape[1]}")
    print(f"Date range  : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(news.head(3))
