"""
tearsheet.py
------------
Generates a self-contained HTML tearsheet for the NLP Sentiment Alpha Signal.

Outputs a single .html file with embedded base64 charts — no server needed.

Author : EquityXperts / Tasrif
"""

from __future__ import annotations
import base64
import io
import logging
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DARK_BG = "#0d1117"
DARK_PANEL = "#161b22"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
GOLD = "#d29922"
WHITE = "#e6edf3"
GREY = "#8b949e"

STYLE = {
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_PANEL,
    "axes.edgecolor": GREY,
    "axes.labelcolor": WHITE,
    "axes.titlecolor": WHITE,
    "text.color": WHITE,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "font.family": "DejaVu Sans",
}


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_cumulative_ls(ls_returns: pd.Series) -> str:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        cumret = ls_returns.cumsum() * 100
        ax.fill_between(cumret.index, cumret.values, alpha=0.15, color=GREEN)
        ax.plot(cumret.index, cumret.values, color=GREEN, linewidth=1.8, label="L/S Cumulative Return")
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
        ax.set_title("Long–Short Portfolio Cumulative Return (OOS)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Return (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.legend(fontsize=9)
        ax.grid(True, axis="y")
        fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_ic_rolling(ic_series: pd.DataFrame) -> str:
    with plt.rc_context(STYLE):
        cols = [c for c in ic_series.columns if c.startswith("IC_") and "Rank" not in c]
        fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.5 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        colors = [ACCENT, GREEN, GOLD, RED]
        for ax, col, color in zip(axes, cols, colors):
            s = ic_series[col].dropna()
            roll = s.rolling(21).mean()
            ax.bar(s.index, s.values, color=color, alpha=0.3, width=1)
            ax.plot(roll.index, roll.values, color=color, linewidth=1.5,
                    label=f"21d MA — {col}")
            ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
            ax.set_ylabel("IC")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y")
        axes[0].set_title("Rolling Information Coefficient by Horizon", fontsize=13, fontweight="bold")
        fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_quantile_returns(quantile_returns: pd.DataFrame) -> str:
    with plt.rc_context(STYLE):
        mean_qret = quantile_returns.mean() * 100
        n = len(mean_qret)
        palette = [RED, "#e36209", GREY, GREEN, ACCENT][:n]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(mean_qret.index, mean_qret.values, color=palette, width=0.6, edgecolor=DARK_BG)
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, mean_qret.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.3f}%", ha="center", va="bottom", fontsize=8.5, color=WHITE)
        ax.set_title("Mean 5D Forward Return by Factor Quintile", fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean Return (%)")
        ax.grid(True, axis="y")
        fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_is_vs_oos(summary: pd.DataFrame, period: str = "5D") -> str:
    with plt.rc_context(STYLE):
        is_col = f"IS_IC_{period}"
        oos_col = f"OOS_IC_{period}"
        if is_col not in summary.columns or oos_col not in summary.columns:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, "No IS/OOS data", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            return _fig_to_b64(fig)
        xs = np.arange(len(summary))
        is_vals = summary[is_col].values
        oos_vals = summary[oos_col].values
        fig, ax = plt.subplots(figsize=(10, 3.5))
        w = 0.35
        ax.bar(xs - w / 2, is_vals, width=w, label="In-Sample IC", color=ACCENT, alpha=0.8)
        ax.bar(xs + w / 2, oos_vals, width=w, label="OOS IC", color=GREEN, alpha=0.8)
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(summary["test_start"].astype(str) if "test_start" in summary.columns else xs,
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(f"IS vs OOS IC ({period} horizon) — Walk-Forward Windows", fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean IC")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y")
        fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_drawdown(ls_returns: pd.Series) -> str:
    with plt.rc_context(STYLE):
        cum = ls_returns.cumsum()
        drawdown = (cum - cum.cummax()) * 100
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color=RED)
        ax.plot(drawdown.index, drawdown.values, color=RED, linewidth=1.2)
        ax.set_title("Drawdown (OOS L/S Portfolio)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, axis="y")
        fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_portfolio_stats(ls_returns: pd.Series) -> dict:
    if ls_returns.empty:
        return {}
    ann_ret = ls_returns.mean() * 252
    ann_vol = ls_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = ls_returns.cumsum()
    dd = cum - cum.cummax()
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit = (ls_returns > 0).mean()
    return {
        "Annualised Return": f"{ann_ret*100:.2f}%",
        "Annualised Volatility": f"{ann_vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Calmar Ratio": f"{calmar:.3f}",
        "Hit Rate": f"{hit*100:.1f}%",
        "Total OOS Days": str(len(ls_returns)),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NLP Sentiment Alpha Signal — Tearsheet</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --gold: #d29922; --white: #e6edf3; --grey: #8b949e;
    --font: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--white); font-family: var(--font); font-size: 14px; }}
  header {{ background: var(--panel); border-bottom: 1px solid var(--border);
            padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ font-size: 22px; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; }}
  header .meta {{ color: var(--grey); font-size: 12px; text-align: right; line-height: 1.7; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px; }}
  .section-title {{ font-size: 13px; font-weight: 600; color: var(--grey);
                    text-transform: uppercase; letter-spacing: 1px;
                    margin: 36px 0 16px; border-left: 3px solid var(--accent); padding-left: 10px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 8px; }}
  .kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
          padding: 14px 16px; }}
  .kpi .label {{ font-size: 11px; color: var(--grey); margin-bottom: 4px; text-transform: uppercase; }}
  .kpi .value {{ font-size: 20px; font-weight: 700; color: var(--white); }}
  .kpi .value.pos {{ color: var(--green); }}
  .kpi .value.neg {{ color: var(--red); }}
  .chart-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
                padding: 16px; margin-bottom: 16px; }}
  .chart-box img {{ width: 100%; border-radius: 4px; }}
  .ic-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  .ic-table th {{ background: #1c2128; color: var(--grey); padding: 8px 12px;
                  text-align: left; font-weight: 600; border-bottom: 1px solid var(--border); }}
  .ic-table td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  .ic-table tr:hover td {{ background: #1c2128; }}
  .positive {{ color: var(--green); font-weight: 600; }}
  .negative {{ color: var(--red); font-weight: 600; }}
  footer {{ text-align: center; color: var(--grey); font-size: 11px;
            padding: 24px; border-top: 1px solid var(--border); margin-top: 40px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 600; }}
  .badge-blue {{ background: #1f3a5f; color: var(--accent); }}
  .badge-green {{ background: #1a3a2a; color: var(--green); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>📰 NLP Sentiment Alpha Signal</h1>
    <div style="color:var(--grey);font-size:12px;margin-top:4px;">
      FinBERT + VADER Dual-Engine · Walk-Forward OOS Validation · EquityXperts
    </div>
  </div>
  <div class="meta">
    Generated: {generated}<br/>
    Universe: {universe}<br/>
    Period: {period}
  </div>
</header>
<div class="container">

  <div class="section-title">Portfolio Performance (OOS)</div>
  <div class="kpi-grid">{kpi_html}</div>

  <div class="section-title">Cumulative Long–Short Return</div>
  <div class="chart-box"><img src="data:image/png;base64,{chart_cumret}" alt="cumulative returns"/></div>

  <div class="section-title">Drawdown</div>
  <div class="chart-box"><img src="data:image/png;base64,{chart_drawdown}" alt="drawdown"/></div>

  <div class="section-title">Rolling Information Coefficient</div>
  <div class="chart-box"><img src="data:image/png;base64,{chart_ic_rolling}" alt="IC rolling"/></div>

  <div class="section-title">Factor Quintile Returns</div>
  <div class="two-col">
    <div class="chart-box"><img src="data:image/png;base64,{chart_quintile}" alt="quintile returns"/></div>
    <div class="chart-box"><img src="data:image/png;base64,{chart_is_oos}" alt="IS vs OOS"/></div>
  </div>

  <div class="section-title">IC Summary Table</div>
  <div class="chart-box" style="overflow-x:auto;">
    <table class="ic-table">{ic_table_html}</table>
  </div>

  <div class="section-title">Walk-Forward Window Summary</div>
  <div class="chart-box" style="overflow-x:auto;">
    <table class="ic-table">{wf_table_html}</table>
  </div>

</div>
<footer>
  NLP Sentiment Alpha Signal · EquityXperts · {generated}<br/>
  <span style="color:#30363d;font-size:10px;">For research purposes only. Not investment advice.</span>
</footer>
</body>
</html>"""


def _kpi_html(stats: dict) -> str:
    html = ""
    for label, value in stats.items():
        cls = ""
        if "%" in value or "Ratio" in label:
            try:
                v = float(value.replace("%", ""))
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
            except ValueError:
                pass
        html += f'<div class="kpi"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>\n'
    return html


def _df_to_table_html(df: pd.DataFrame) -> str:
    df = df.reset_index()
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for val in row.values:
            try:
                f = float(val)
                cls = " class='positive'" if f > 0.01 else (" class='negative'" if f < -0.01 else "")
                cells += f"<td{cls}>{round(f, 5)}</td>"
            except (ValueError, TypeError):
                cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"
    return f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody>"


def generate_tearsheet(
    ls_returns: pd.Series,
    ic_series: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    wf_summary: pd.DataFrame,
    universe_size: int,
    date_range: str,
    output_path: str = "reports/tearsheet.html",
) -> str:
    """
    Build and save the HTML tearsheet.

    Returns the path of the saved file.
    """
    logger.info("Rendering tearsheet charts …")

    chart_cumret = _plot_cumulative_ls(ls_returns)
    chart_drawdown = _plot_drawdown(ls_returns)
    chart_ic_rolling = _plot_ic_rolling(ic_series)
    chart_quintile = _plot_quantile_returns(quantile_returns)
    chart_is_oos = _plot_is_vs_oos(wf_summary)

    stats = _compute_portfolio_stats(ls_returns)
    kpi = _kpi_html(stats)

    html = HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe=f"{universe_size} synthetic tickers",
        period=date_range,
        kpi_html=kpi,
        chart_cumret=chart_cumret,
        chart_drawdown=chart_drawdown,
        chart_ic_rolling=chart_ic_rolling,
        chart_quintile=chart_quintile,
        chart_is_oos=chart_is_oos,
        ic_table_html=_df_to_table_html(ic_summary),
        wf_table_html=_df_to_table_html(wf_summary) if not wf_summary.empty else "<tr><td>No data</td></tr>",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Tearsheet saved → %s", output_path)
    return output_path
