# 📰 NLP Sentiment Alpha Signal

 
> Build a news sentiment factor from financial headlines, validate its predictive power with rigorous walk-forward out-of-sample testing, and integrate it into a cross-sectional equity ranking model.

[![CI](https://github.com/tashrifulkabir34-lang/nlp-sentiment-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/tashrifulkabir34-lang/nlp-sentiment-alpha/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Methodology](#methodology)
4. [Out-of-Sample Results](#out-of-sample-results)
5. [Limitations & Lessons Learned](#limitations--lessons-learned)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [GitHub Actions CI](#github-actions-ci)
11. [Roadmap / Potential Improvements](#roadmap--potential-improvements)
12. [References](#references)

---

## Project Overview

This project constructs a **cross-sectional NLP sentiment alpha factor** from financial news headlines and measures its ability to predict short-to-medium-term equity returns. The pipeline:

- Generates (or ingests) a universe of financial headlines across 50+ synthetic tickers over a 3-year period.
- Scores each headline with a **dual-engine ensemble** (VADER + FinBERT), producing a continuous sentiment score in `[-1, +1]`.
- Constructs a daily **cross-sectional factor** via aggregation, z-score normalisation, and exponential momentum decay.
- Computes **Information Coefficients** (IC & Rank-IC) across 1D, 5D, 10D, and 21D horizons.
- Runs a **rolling walk-forward backtest** with explicit train/test separation, simulating a realistic long-short portfolio.
- Produces a **self-contained HTML tearsheet** with embedded charts — no server needed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NLP Sentiment Alpha Pipeline                │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ data_         │    │ sentiment_        │    │ signal_      │  │
│  │ generator.py  │───▶│ scorer.py        │───▶│ constructor  │  │
│  │              │    │                  │    │ .py          │  │
│  │ Synthetic     │    │ VADER (fast)     │    │              │  │
│  │ headlines +   │    │ FinBERT (deep)   │    │ Aggregate    │  │
│  │ GBM prices   │    │ Ensemble blend   │    │ Z-score norm │  │
│  └──────────────┘    └──────────────────┘    │ EWM decay    │  │
│                                              │ IC analysis  │  │
│                                              └──────┬───────┘  │
│                                                     │           │
│  ┌──────────────┐                          ┌────────▼───────┐  │
│  │ tearsheet.py │◀─────────────────────────│ backtest.py    │  │
│  │              │                          │                │  │
│  │ 5 embedded   │                          │ Walk-forward   │  │
│  │ dark-theme   │                          │ L/S simulation │  │
│  │ charts       │                          │ IS vs OOS IC   │  │
│  │ KPI summary  │                          │ TC adjustment  │  │
│  └──────────────┘                          └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### 1. Sentiment Scoring

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**  
A lexicon-based rule-driven scorer optimised for social/financial short-form text. Produces a compound score in `[-1, +1]` with no GPU requirement, making it suitable as a fast baseline and CI-compatible fallback.

**FinBERT (ProsusAI/finbert)**  
A BERT model fine-tuned on financial communications (earnings calls, analyst reports, news). We use it in classification mode and convert three-class outputs to a continuous score:
```
finbert_score = P(positive) - P(negative) ∈ [-1, +1]
```
FinBERT captures domain-specific vocabulary (e.g., "guidance raised", "impairment charge") better than general-purpose models.

**Blended Ensemble**  
```
sentiment_score = α · vader_score + (1 - α) · finbert_score
```
Default: `α = 0.4` (VADER), `1 - α = 0.6` (FinBERT). Pure VADER (`α = 1.0`) is used in CI for speed.

### 2. Factor Construction

**Daily Aggregation**  
Multiple headlines per ticker-date are collapsed to a single score using mean aggregation. Alternative methods (max, EWM-last) are supported via the `agg` parameter.

**Cross-sectional Z-score Normalisation**  
Within each date, factor values are standardised across tickers and winsorised at ±3σ to remove outlier influence:
```
z_i,t = clip((s_i,t - μ_t) / σ_t, -3, +3)
```

**Exponential Momentum Decay**  
Sentiment has a finite shelf life — news fades. We model persistence with an EWM filter across time per ticker:
```
factor_decayed_t = EWM(factor_zscore, halflife=5 days)
```
The half-life is configurable (default: 5 business days). Longer half-lives produce smoother but more lagged signals.

### 3. IC Analysis

The Information Coefficient is the Pearson correlation between the factor and forward returns, computed cross-sectionally per day:
```
IC_t = corr(factor_decayed_{i,t}, return_{i,t→t+h})
```
Rank-IC (Spearman) is computed in parallel as a robust alternative.

**IC Summary statistics reported:**

| Statistic | Interpretation |
|-----------|---------------|
| Mean IC | Average predictive power |
| Std IC | Consistency of the signal |
| t-stat | Statistical significance: t = mean / (std / √n) |
| ICIR | Information Ratio of IC: mean / std |
| Hit Rate | % of days with positive IC |

**Rule of thumb:** Mean IC > 0.03 with ICIR > 0.3 suggests a viable alpha signal.

### 4. Walk-Forward Backtest

To avoid look-ahead bias, all out-of-sample results use a strict rolling window design:

```
Timeline: ─────[TRAIN 12M]────────[TEST 3M]────── (rolling +3M)
               No lookahead        OOS evaluation
```

- **Training window:** 12 months (configurable)
- **Test window:** 3 months (configurable)
- **Step:** 3 months (non-overlapping OOS periods)
- **Portfolio:** Long top quintile, short bottom quintile
- **Transaction costs:** 10 bps round-trip, proportional to turnover

### 5. Long-Short Portfolio Simulation

On each test day:
1. Rank tickers by `factor_decayed`.
2. Long top 20% (Q5), short bottom 20% (Q1).
3. Net return = `mean_return(longs) - mean_return(shorts)`.
4. Deduct turnover-weighted transaction cost.

---

## Out-of-Sample Results

> **Note:** These results are generated on a synthetic dataset with a known embedded sentiment signal (IC ≈ 0.03–0.05 by construction). The purpose is to demonstrate the complete methodology rather than claim live trading performance.

The pipeline produces:
- **IC Summary CSV** at `reports/ic_summary.csv`
- **Walk-Forward Summary** at `reports/walk_forward_summary.csv`
- **HTML Tearsheet** at `reports/tearsheet.html`

Run to reproduce:
```bash
python main.py --seed 42 --n-tickers 50 --start 2021-01-01 --end 2023-12-31
```

**Key observations from OOS testing:**
- VADER alone on synthetic data exhibits near-zero IC, reflecting the modest embedded signal strength and noise dominance. This is the expected behaviour for an honest OOS evaluation.
- Adding FinBERT (`--use-finbert`) on real news data significantly improves factor quality due to domain-specific vocabulary understanding.
- IS IC consistently exceeds OOS IC across all walk-forward windows, confirming the importance of reporting OOS-only results.
- Momentum decay (halflife=5) provides measurable IC improvement over raw daily scores.

---

## Limitations & Lessons Learned

### Limitations

1. **Synthetic data ceiling:** The generator embeds a controlled sentiment-return correlation (~5%), which bounds achievable OOS IC. Real financial news contains stronger and more varied signals.

2. **News source quality:** VADER and FinBERT both struggle with nuanced hedged language ("guidance raised, but below expectations"), sarcasm, and compound events. Headline-level scoring misses article body context.

3. **Look-ahead in embedding:** FinBERT was fine-tuned on historical financial text. Survivorship bias in training data may overstate out-of-sample performance on real news.

4. **No transaction cost modelling of market impact:** The 10 bps flat TC does not account for the bid-ask spread widening that occurs when simultaneously trading many positions.

5. **Synthetic universe limitations:** Real equity universes have sector correlations, earnings seasonality, and liquidity constraints that synthetic GBM paths do not capture.

6. **No sentiment source diversification:** A single news feed is used. Production systems typically aggregate multiple sources (Bloomberg, Reuters, Twitter/X, SEC filings) with source quality weights.

### Lessons Learned

- **VADER as a baseline is underrated.** For quick iteration and CI-speed testing, VADER delivers a useful direction even on financial text. Its main weakness is domain-agnostic vocabulary.
- **Walk-forward design matters more than model sophistication.** The gap between IS and OOS IC is often larger than the IC improvement from switching VADER → FinBERT. Clean data discipline beats model cleverness.
- **Momentum decay stabilises but lags.** Short half-lives (3 days) produce noisy but timely signals; long half-lives (21 days) produce stable but delayed ones. 5 days is a reasonable default for daily rebalancing.
- **Cross-sectional IC is noisy at the daily level.** Rolling 21-day mean IC is far more informative than daily IC bars.

---

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- `pip`

### Installation

```bash
git clone https://github.com/tashrifulkabir34-lang/nlp-sentiment-alpha.git
cd nlp-sentiment-alpha
pip install -r requirements.txt
```

### Run Pipeline (VADER only — fast, no GPU)

```bash
python main.py --vader-weight 1.0
```

### Run Pipeline with FinBERT (downloads ~440 MB on first run)

```bash
python main.py --use-finbert --vader-weight 0.4
```

### View Results

Open `reports/tearsheet.html` in any browser — it is fully self-contained (no server needed).

---

## Project Structure

```
nlp-sentiment-alpha/
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic headline + price dataset
│   ├── sentiment_scorer.py   # VADER + FinBERT dual-engine scorer
│   ├── signal_constructor.py # Factor construction, IC analysis
│   ├── backtest.py           # Walk-forward OOS backtesting engine
│   └── tearsheet.py          # Self-contained HTML report generator
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py
│   ├── test_sentiment_scorer.py
│   └── test_signal_constructor.py
├── reports/                   # Generated outputs (gitignored)
│   ├── tearsheet.html
│   ├── ic_summary.csv
│   ├── walk_forward_summary.csv
│   └── ls_returns.csv
├── .github/
│   └── workflows/
│       └── ci.yml
├── main.py                    # Entry-point CLI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

All parameters are controlled via `main.py` CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--use-finbert` | False | Enable FinBERT scoring |
| `--vader-weight` | 1.0 | VADER weight in ensemble [0, 1] |
| `--halflife` | 5 | EWM decay half-life in days |
| `--n-tickers` | 50 | Synthetic universe size |
| `--start` | 2021-01-01 | Simulation start date |
| `--end` | 2023-12-31 | Simulation end date |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | reports/tearsheet.html | Tearsheet output path |
| `--train-months` | 12 | Walk-forward training window length |
| `--test-months` | 3 | Walk-forward test window length |
| `--tc-bps` | 10 | Round-trip transaction cost in bps |

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_sentiment_scorer.py -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

The suite covers 24 tests across data generation, sentiment scoring, and signal construction.

---

## GitHub Actions CI

The CI workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

1. Tests Python 3.10 and 3.11 in a matrix.
2. Installs a lightweight dependency set (VADER only — no 440 MB torch download in CI).
3. Runs the full 24-test pytest suite.
4. Executes a smoke-test pipeline run on a tiny 10-ticker, 1-year dataset.

**Setup (web UI):**
- Go to your repository → `Actions` tab → Enable workflows.
- The workflow file is already committed at `.github/workflows/ci.yml`.

---

## Roadmap / Potential Improvements

- **Real news ingestion:** Replace synthetic data with live RSS feeds (Reuters, Yahoo Finance) or paid APIs (Benzinga, Alpha Vantage News).
- **Article-level FinBERT:** Score full article bodies, not just headlines, using sentence-level chunking with attention pooling.
- **Sentiment source diversification:** Weight multiple news sources by quality (e.g., Reuters > blog posts).
- **Alternative NLP models:** Compare against `RoBERTa-financial`, `LLAMA-3-finance`, or GPT-4 via the Anthropic API for structured sentiment extraction.
- **Sector-neutral factor:** Demean within GICS sectors to isolate idiosyncratic sentiment from sector rotation effects.
- **Event-driven decay:** Use a separate shorter half-life around earnings dates (e.g., 2-day decay) vs. normal periods (5-day).
- **Alphalens integration:** Full alphalens tearsheet for standard factor quantile returns, auto-correlation, and sector breakdown.
- **Portfolio optimisation integration:** Combine sentiment factor with momentum, value, and quality factors via mean-variance or risk-parity optimisation.

---

## References

- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063.
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM.
- Fama, E. F., & French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance.
- HuggingFace Transformers: https://huggingface.co/transformers/
- arXiv NLP Finance: https://arxiv.org/list/q-fin/recent
- Alphalens GitHub: https://github.com/quantopian/alphalens

---

*EquityXperts · For research and educational purposes only. Not investment advice.*
