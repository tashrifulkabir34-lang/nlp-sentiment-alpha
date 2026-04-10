"""
backtest.py
-----------
Walk-forward out-of-sample backtest of the NLP sentiment factor.

Design:
  • Split timeline into rolling train/test windows.
  • In each window, fit a ranking model (quantile rank) on train data.
  • Apply signal to test data; record IC statistics.
  • Compare in-sample vs out-of-sample IC to detect overfitting.
  • Simulate a long-short portfolio: long top quintile, short bottom quintile.

Author : EquityXperts / Tasrif
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Walk-forward backtest configuration."""
    train_months: int = 12     # months of data per training window
    test_months: int = 3       # months per out-of-sample test window
    step_months: int = 3       # step size (roll) in months
    n_quantiles: int = 5       # factor quantiles for L/S portfolio
    transaction_cost_bps: float = 10.0  # round-trip cost in basis points
    periods: list = field(default_factory=lambda: [1, 5, 10, 21])


@dataclass
class WalkForwardResult:
    """Container for a single walk-forward window result."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample_ic: dict      # period -> mean IC on train data
    out_of_sample_ic: dict  # period -> mean IC on test data
    long_short_returns: pd.Series   # daily L/S portfolio returns (test window)


def _mean_ic_for_window(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    factor_col: str = "factor_decayed",
    periods: list = [1, 5, 10, 21],
) -> dict:
    """Compute mean IC per period for [start, end] date range."""
    from .signal_constructor import compute_ic_series, compute_ic_summary

    idx = factor.index.get_level_values(0)
    mask = (idx >= start) & (idx <= end)
    f_window = factor[mask]

    idx2 = forward_returns.index.get_level_values(0)
    mask2 = (idx2 >= start) & (idx2 <= end)
    r_window = forward_returns[mask2]

    if len(f_window) < 10:
        return {f"{p}D": np.nan for p in periods}

    ic_series = compute_ic_series(f_window, r_window, factor_col=factor_col, periods=periods)
    ic_mean = ic_series.mean().to_dict()
    return ic_mean


def _simulate_long_short(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_quantiles: int = 5,
    factor_col: str = "factor_decayed",
    period: int = 1,
    tc_bps: float = 10.0,
) -> pd.Series:
    """
    Simulate daily long-short returns in [start, end].

    Long: top quintile | Short: bottom quintile.
    Returns are gross of the given transaction cost.
    """
    fwd_col = f"{period}D"
    idx = factor.index.get_level_values(0)
    mask = (idx >= start) & (idx <= end)
    f_window = factor[mask][[factor_col]]

    idx2 = forward_returns.index.get_level_values(0)
    mask2 = (idx2 >= start) & (idx2 <= end)
    r_window = forward_returns[mask2][[fwd_col]]

    merged = f_window.join(r_window, how="inner").reset_index()
    if merged.empty:
        return pd.Series(dtype=float)

    date_col = merged.columns[0]
    tc_per_trade = tc_bps / 10_000.0

    daily_returns: list[dict] = []
    prev_longs: set = set()
    prev_shorts: set = set()

    for date in sorted(merged[date_col].unique()):
        day = merged[merged[date_col] == date].dropna()
        if len(day) < n_quantiles * 2:
            continue

        day = day.copy()
        day["q"] = pd.qcut(day[factor_col], n_quantiles, labels=False, duplicates="drop")

        long_mask = day["q"] == (n_quantiles - 1)
        short_mask = day["q"] == 0

        longs = set(day.loc[long_mask, merged.columns[1]])
        shorts = set(day.loc[short_mask, merged.columns[1]])

        long_ret = day.loc[long_mask, fwd_col].mean()
        short_ret = day.loc[short_mask, fwd_col].mean()
        ls_gross = long_ret - short_ret

        # Turnover-based transaction cost
        turnover_l = len(longs.symmetric_difference(prev_longs)) / max(len(longs), 1)
        turnover_s = len(shorts.symmetric_difference(prev_shorts)) / max(len(shorts), 1)
        total_tc = (turnover_l + turnover_s) * tc_per_trade * 0.5

        ls_net = ls_gross - total_tc
        daily_returns.append({"date": date, "ls_return": ls_net})

        prev_longs, prev_shorts = longs, shorts

    if not daily_returns:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(daily_returns).set_index("date")["ls_return"]
    return ret_df


def run_walk_forward_backtest(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> tuple[list[WalkForwardResult], pd.DataFrame]:
    """
    Execute walk-forward backtest.

    Returns
    -------
    results : list of WalkForwardResult
    summary : pd.DataFrame summarising each window's IS vs OOS IC.
    """
    if cfg is None:
        cfg = BacktestConfig()

    all_dates = sorted(factor.index.get_level_values(0).unique())
    if not all_dates:
        raise ValueError("Factor has no dates.")

    global_start = all_dates[0]
    global_end = all_dates[-1]

    results: list[WalkForwardResult] = []
    train_start = global_start

    while True:
        train_end = train_start + pd.DateOffset(months=cfg.train_months)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=cfg.test_months)

        if test_start > global_end:
            break

        test_end = min(test_end, global_end)

        logger.info(
            "Window: train [%s → %s] | test [%s → %s]",
            train_start.date(), train_end.date(),
            test_start.date(), test_end.date(),
        )

        is_ic = _mean_ic_for_window(factor, forward_returns, train_start, train_end, periods=cfg.periods)
        oos_ic = _mean_ic_for_window(factor, forward_returns, test_start, test_end, periods=cfg.periods)

        ls_returns = _simulate_long_short(
            factor, forward_returns,
            test_start, test_end,
            n_quantiles=cfg.n_quantiles,
            tc_bps=cfg.transaction_cost_bps,
        )

        results.append(WalkForwardResult(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            in_sample_ic=is_ic,
            out_of_sample_ic=oos_ic,
            long_short_returns=ls_returns,
        ))

        train_start += pd.DateOffset(months=cfg.step_months)

    # Build summary table
    rows = []
    for r in results:
        row = {
            "test_start": r.test_start.date(),
            "test_end": r.test_end.date(),
        }
        for p in cfg.periods:
            key = f"IC_{p}D"
            row[f"IS_{key}"] = round(r.in_sample_ic.get(f"{p}D", np.nan), 5)
            row[f"OOS_{key}"] = round(r.out_of_sample_ic.get(f"{p}D", np.nan), 5)
        ls_ret = r.long_short_returns
        if len(ls_ret) > 0:
            ann_ret = ls_ret.mean() * 252
            ann_vol = ls_ret.std() * np.sqrt(252)
            row["ann_return"] = round(ann_ret, 4)
            row["ann_vol"] = round(ann_vol, 4)
            row["sharpe"] = round(ann_ret / ann_vol if ann_vol > 0 else np.nan, 3)
            row["max_drawdown"] = round(
                ((ls_ret.cumsum() - ls_ret.cumsum().cummax()).min()), 4
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    return results, summary


def aggregate_long_short_returns(results: list[WalkForwardResult]) -> pd.Series:
    """Concatenate OOS long-short returns across all windows."""
    all_ret = pd.concat([r.long_short_returns for r in results if len(r.long_short_returns) > 0])
    return all_ret.sort_index()
