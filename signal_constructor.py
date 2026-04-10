"""
signal_constructor.py
---------------------
Cross-sectional sentiment alpha factor construction, IC analysis,
momentum decay, quantile returns.

Author : EquityXperts / Tasrif
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def aggregate_daily_sentiment(
    scored_news: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    score_col: str = "sentiment_score",
    agg: str = "mean",
) -> pd.DataFrame:
    """Collapse multiple headlines per (date, ticker) into one factor score."""
    df = scored_news[[date_col, ticker_col, score_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    aggregated = df.groupby([date_col, ticker_col])[score_col].agg(agg).reset_index()
    return aggregated.set_index([date_col, ticker_col])


def cross_sectional_zscore(
    factor: pd.DataFrame,
    score_col: str = "sentiment_score",
    winsorize_std: float = 3.0,
) -> pd.DataFrame:
    """Cross-sectional z-score with winsorisation per date."""
    df = factor.reset_index()
    date_col = df.columns[0]

    def _zscore(group):
        s = group[score_col]
        mu, sigma = s.mean(), s.std()
        if sigma < 1e-9:
            return pd.Series(np.zeros(len(s)), index=group.index)
        return ((s - mu) / sigma).clip(-winsorize_std, winsorize_std)

    df["factor_zscore"] = df.groupby(date_col, group_keys=False).apply(
        _zscore, include_groups=False
    )
    return df.set_index([date_col, df.columns[1]])


def apply_momentum_decay(
    factor: pd.DataFrame,
    halflife_days: int = 5,
    score_col: str = "factor_zscore",
) -> pd.DataFrame:
    """Apply EWM decay across time per ticker to model sentiment persistence."""
    df = factor.reset_index()
    date_col = df.columns[0]
    ticker_col = df.columns[1]
    wide = df.pivot(index=date_col, columns=ticker_col, values=score_col)
    decayed = wide.ewm(halflife=halflife_days).mean()
    long = decayed.stack().rename("factor_decayed").reset_index()
    long.columns = [date_col, ticker_col, "factor_decayed"]
    merged = df.merge(long, on=[date_col, ticker_col], how="left")
    return merged.set_index([date_col, ticker_col])


def compute_forward_returns(
    prices: pd.DataFrame,
    periods: list = [1, 5, 10, 21],
) -> pd.DataFrame:
    """Compute forward log returns for multiple horizons."""
    log_prices = np.log(prices)
    frames = {}
    for h in periods:
        fwd = log_prices.shift(-h) - log_prices
        frames[f"{h}D"] = fwd
    long_frames = []
    for label, df in frames.items():
        melted = df.stack().rename(label).reset_index()
        melted.columns = ["date", "ticker", label]
        long_frames.append(melted.set_index(["date", "ticker"]))
    return pd.concat(long_frames, axis=1)


def compute_ic_series(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_col: str = "factor_decayed",
    periods: list = [1, 5, 10, 21],
) -> pd.DataFrame:
    """Compute daily Pearson IC and Spearman Rank-IC."""
    merged = factor[[factor_col]].join(forward_returns, how="inner").reset_index()
    date_col = merged.columns[0]
    records = []
    for date in merged[date_col].unique():
        day = merged[merged[date_col] == date].dropna()
        if len(day) < 5:
            continue
        row = {"date": date}
        for h in periods:
            col = f"{h}D"
            if col not in day.columns:
                continue
            sub = day[[factor_col, col]].dropna()
            if len(sub) < 5:
                row[f"IC_{col}"] = np.nan
                row[f"RankIC_{col}"] = np.nan
                continue
            row[f"IC_{col}"] = sub[factor_col].corr(sub[col])
            row[f"RankIC_{col}"] = spearmanr(sub[factor_col], sub[col])[0]
        records.append(row)
    return pd.DataFrame(records).set_index("date").sort_index()


def compute_ic_summary(ic_series: pd.DataFrame) -> pd.DataFrame:
    """Summarise IC: mean, std, t-stat, ICIR, hit-rate."""
    rows = []
    for col in ic_series.columns:
        s = ic_series[col].dropna()
        n = len(s)
        mean_ic = s.mean()
        std_ic = s.std()
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 0 else np.nan
        icir = mean_ic / std_ic if std_ic > 0 else np.nan
        rows.append({
            "metric": col,
            "mean_IC": round(mean_ic, 5),
            "std_IC": round(std_ic, 5),
            "t_stat": round(t_stat, 3),
            "ICIR": round(icir, 3),
            "hit_rate": round((s > 0).mean(), 3),
            "n_obs": n,
        })
    return pd.DataFrame(rows).set_index("metric")


def compute_quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_col: str = "factor_decayed",
    period: int = 5,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return per factor quantile per date."""
    fwd_col = f"{period}D"
    merged = factor[[factor_col]].join(forward_returns[[fwd_col]], how="inner").reset_index()
    date_col = merged.columns[0]

    def _qret(group):
        sub = group[[factor_col, fwd_col]].dropna()
        if len(sub) < n_quantiles * 2:
            return pd.Series(dtype=float)
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[factor_col], n_quantiles,
                           labels=[f"Q{i+1}" for i in range(n_quantiles)],
                           duplicates="drop")
        return sub.groupby("q", observed=False)[fwd_col].mean()

    return merged.groupby(date_col).apply(_qret).unstack(level=1)


def build_factor(
    scored_news: pd.DataFrame,
    prices: pd.DataFrame,
    halflife_days: int = 5,
    periods: list = [1, 5, 10, 21],
) -> dict:
    """End-to-end factor construction pipeline."""
    logger.info("Aggregating daily sentiment …")
    daily = aggregate_daily_sentiment(scored_news)
    logger.info("Z-score normalisation …")
    normalised = cross_sectional_zscore(daily)
    logger.info("Momentum decay (halflife=%d) …", halflife_days)
    factor = apply_momentum_decay(normalised, halflife_days=halflife_days)
    logger.info("Forward returns …")
    fwd_returns = compute_forward_returns(prices, periods=periods)
    logger.info("IC series …")
    ic_series = compute_ic_series(factor, fwd_returns, periods=periods)
    ic_summary = compute_ic_summary(ic_series)
    quantile_ret = compute_quantile_returns(factor, fwd_returns, period=5)
    return {
        "factor": factor,
        "forward_returns": fwd_returns,
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "quantile_returns": quantile_ret,
    }
