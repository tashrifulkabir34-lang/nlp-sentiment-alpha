#!/usr/bin/env python3
"""
main.py
-------
Entry-point for the NLP Sentiment Alpha Signal pipeline.

Usage
-----
  python main.py                          # full run (VADER only, fast)
  python main.py --use-finbert            # include FinBERT (requires download ~440MB)
  python main.py --vader-weight 1.0       # pure VADER
  python main.py --halflife 10            # slower decay
  python main.py --output reports/my.html # custom output path

Author : EquityXperts / Tasrif
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NLP Sentiment Alpha Signal Pipeline")
    p.add_argument("--use-finbert", action="store_true",
                   help="Enable FinBERT scoring (downloads ~440 MB on first run)")
    p.add_argument("--vader-weight", type=float, default=1.0,
                   help="VADER weight [0,1]. FinBERT weight = 1 - vader_weight. (default: 1.0)")
    p.add_argument("--halflife", type=int, default=5,
                   help="EWM decay half-life in days (default: 5)")
    p.add_argument("--n-tickers", type=int, default=50,
                   help="Synthetic universe size (default: 50)")
    p.add_argument("--start", type=str, default="2021-01-01",
                   help="Simulation start date (default: 2021-01-01)")
    p.add_argument("--end", type=str, default="2023-12-31",
                   help="Simulation end date (default: 2023-12-31)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output", type=str, default="reports/tearsheet.html",
                   help="HTML tearsheet output path")
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--test-months", type=int, default=3)
    p.add_argument("--tc-bps", type=float, default=10.0,
                   help="Round-trip transaction cost in bps (default: 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("NLP Sentiment Alpha Signal Pipeline — EquityXperts")
    logger.info("=" * 60)
    logger.info("Config: FinBERT=%s | VADER weight=%.1f | Halflife=%d days",
                args.use_finbert, args.vader_weight, args.halflife)

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    from src.data_generator import build_dataset
    logger.info("Building synthetic dataset …")
    news_df, prices_df = build_dataset(
        n_tickers=args.n_tickers,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
    logger.info("  Headlines: %d | Tickers: %d | Days: %d",
                len(news_df), prices_df.shape[1], prices_df.shape[0])

    # ------------------------------------------------------------------
    # 2. Sentiment scoring
    # ------------------------------------------------------------------
    from src.sentiment_scorer import SentimentScorer
    logger.info("Scoring headlines …")
    scorer = SentimentScorer(
        vader_weight=args.vader_weight,
        use_finbert=args.use_finbert,
    )
    scored_news = scorer.score_dataframe(news_df)
    logger.info("  Scored %d headlines.", len(scored_news))

    # ------------------------------------------------------------------
    # 3. Factor construction
    # ------------------------------------------------------------------
    from src.signal_constructor import build_factor
    logger.info("Building alpha factor …")
    factor_results = build_factor(
        scored_news=scored_news,
        prices=prices_df,
        halflife_days=args.halflife,
        periods=[1, 5, 10, 21],
    )

    ic_summary = factor_results["ic_summary"]
    logger.info("\nIC Summary (full-sample):\n%s", ic_summary.to_string())

    # ------------------------------------------------------------------
    # 4. Walk-forward backtest
    # ------------------------------------------------------------------
    from src.backtest import BacktestConfig, run_walk_forward_backtest, aggregate_long_short_returns
    logger.info("Running walk-forward backtest …")
    cfg = BacktestConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=3,
        transaction_cost_bps=args.tc_bps,
    )
    wf_results, wf_summary = run_walk_forward_backtest(
        factor_results["factor"],
        factor_results["forward_returns"],
        cfg=cfg,
    )
    logger.info("  Walk-forward windows completed: %d", len(wf_results))

    ls_returns = aggregate_long_short_returns(wf_results)
    ann_ret = ls_returns.mean() * 252
    ann_vol = ls_returns.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    logger.info("  OOS L/S | Ann Return: %.2f%% | Sharpe: %.3f", ann_ret * 100, sharpe)

    # ------------------------------------------------------------------
    # 5. Tearsheet
    # ------------------------------------------------------------------
    from src.tearsheet import generate_tearsheet
    date_range = f"{prices_df.index[0].date()} → {prices_df.index[-1].date()}"
    output = generate_tearsheet(
        ls_returns=ls_returns,
        ic_series=factor_results["ic_series"],
        ic_summary=ic_summary,
        quantile_returns=factor_results["quantile_returns"],
        wf_summary=wf_summary,
        universe_size=prices_df.shape[1],
        date_range=date_range,
        output_path=args.output,
    )

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    Path("reports").mkdir(parents=True, exist_ok=True)
    ic_summary.to_csv("reports/ic_summary.csv")
    wf_summary.to_csv("reports/walk_forward_summary.csv", index=False)
    ls_returns.to_csv("reports/ls_returns.csv", header=["ls_return"])

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds.", elapsed)
    logger.info("Tearsheet   → %s", output)
    logger.info("IC Summary  → reports/ic_summary.csv")
    logger.info("WF Summary  → reports/walk_forward_summary.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
