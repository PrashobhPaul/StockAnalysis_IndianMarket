"""
pipeline/oneil_scorer.py — CANSLIM-inspired fundamental scoring.

William O'Neil's CANSLIM framework distilled to what we can actually compute
from yfinance for Indian tickers:

    C — Current quarterly EPS growth (YoY)      → from quarterly_income_stmt
    A — Annual EPS growth (3Y)                  → from income_stmt
    N — New highs / product / management        → proxied by 52w high proximity
    S — Supply & demand (shares, volume)        → avg 20D volume × price
    L — Leader not laggard (relative strength)  → 12M return vs NIFTY50
    I — Institutional sponsorship               → heldPercentInstitutions
    M — Market direction                        → handled in regime_detector

Each sub-component is scored 0–15 (rough weight reflects O'Neil's emphasis on
earnings). Total OneilScore is 0–100. Missing data → 50 (neutral) so a single
missing field doesn't tank an otherwise-healthy stock.

Never raises. `score_ticker` returns a dict with score + sub-components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from pipeline.utils import get_logger, safe_float, fetch_ohlcv

log = get_logger("oneil")

NIFTY50_SYMBOL = "^NSEI"


@dataclass
class OneilBreakdown:
    eps_growth_qoq: float     # 0-15
    eps_growth_annual: float  # 0-15
    proximity_52w: float      # 0-10
    liquidity: float          # 0-10
    relative_strength: float  # 0-20  ← highest weight; "L" is core
    institutional: float      # 0-15
    roe: float                # 0-15

    def total(self) -> float:
        return (
            self.eps_growth_qoq
            + self.eps_growth_annual
            + self.proximity_52w
            + self.liquidity
            + self.relative_strength
            + self.institutional
            + self.roe
        )


def _get_quarterly_eps_growth(tk) -> float:
    """
    Latest quarter EPS YoY growth in %. Requires ≥5 quarters of data
    (current quarter + same quarter 1 year ago). Returns 0.0 if unavailable.

    yfinance `quarterly_earnings` is deprecated. We use `quarterly_income_stmt`
    and read 'Diluted EPS' or 'Basic EPS' rows.
    """
    try:
        qis = tk.quarterly_income_stmt
        if qis is None or qis.empty:
            return 0.0
        for row_name in ("Diluted EPS", "Basic EPS"):
            if row_name in qis.index:
                row = qis.loc[row_name].dropna()
                if len(row) >= 5:
                    current = safe_float(row.iloc[0], 0.0)
                    year_ago = safe_float(row.iloc[4], 0.0)
                    if year_ago > 0:
                        return (current - year_ago) / year_ago * 100.0
        return 0.0
    except Exception as e:
        log.debug("quarterly EPS lookup failed: %s", e)
        return 0.0


def _get_annual_eps_cagr(tk) -> float:
    """
    3-year EPS CAGR in %. Annual income statement has ~4 years of data.
    Returns 0.0 if unavailable.
    """
    try:
        ais = tk.income_stmt
        if ais is None or ais.empty:
            return 0.0
        for row_name in ("Diluted EPS", "Basic EPS"):
            if row_name in ais.index:
                row = ais.loc[row_name].dropna()
                if len(row) >= 4:
                    latest = safe_float(row.iloc[0], 0.0)
                    three_ago = safe_float(row.iloc[3], 0.0)
                    if three_ago > 0 and latest > 0:
                        return ((latest / three_ago) ** (1 / 3) - 1.0) * 100.0
        return 0.0
    except Exception as e:
        log.debug("annual EPS lookup failed: %s", e)
        return 0.0


def _score_eps_growth_qoq(growth_pct: float) -> float:
    """O'Neil benchmark: look for ≥25% QoQ. Scale 0-15."""
    if growth_pct >= 50:
        return 15.0
    if growth_pct >= 25:
        return 12.0
    if growth_pct >= 10:
        return 8.0
    if growth_pct >= 0:
        return 4.0
    return 0.0


def _score_eps_growth_annual(cagr_pct: float) -> float:
    """Annual 3Y CAGR. ≥20% is O'Neil's threshold."""
    if cagr_pct >= 30:
        return 15.0
    if cagr_pct >= 20:
        return 12.0
    if cagr_pct >= 10:
        return 8.0
    if cagr_pct >= 0:
        return 4.0
    return 0.0


def _score_proximity_52w(close: float, high_52w: float) -> float:
    """
    O'Neil buys near 52w highs (breakouts). Score 0-10 based on how close
    current price is to the 52w high. 100% = at high, <70% = laggard.
    """
    if high_52w <= 0:
        return 5.0
    ratio = close / high_52w
    if ratio >= 0.98:
        return 10.0
    if ratio >= 0.90:
        return 8.0
    if ratio >= 0.80:
        return 5.0
    if ratio >= 0.70:
        return 2.0
    return 0.0


def _score_liquidity(avg_dollar_vol_cr: float) -> float:
    """
    Average daily turnover in ₹crore. We want stocks we can actually trade.
    ≥50 Cr/day is institutional-grade; <2 Cr is illiquid.
    """
    if avg_dollar_vol_cr >= 50:
        return 10.0
    if avg_dollar_vol_cr >= 20:
        return 8.0
    if avg_dollar_vol_cr >= 5:
        return 5.0
    if avg_dollar_vol_cr >= 2:
        return 2.0
    return 0.0


def _score_relative_strength(stock_return_pct: float, market_return_pct: float) -> float:
    """
    12M stock return minus 12M NIFTY return. O'Neil wants leaders beating
    the index by 20%+. Score 0-20.
    """
    outperformance = stock_return_pct - market_return_pct
    if outperformance >= 30:
        return 20.0
    if outperformance >= 20:
        return 17.0
    if outperformance >= 10:
        return 13.0
    if outperformance >= 0:
        return 9.0
    if outperformance >= -10:
        return 4.0
    return 0.0


def _score_institutional(pct: float) -> float:
    """heldPercentInstitutions from yfinance. 15-40% is healthy; >60% crowded."""
    if pct is None:
        return 7.5
    pct_num = pct * 100 if pct <= 1 else pct
    if 20 <= pct_num <= 50:
        return 15.0
    if 10 <= pct_num <= 60:
        return 11.0
    if 5 <= pct_num <= 70:
        return 7.0
    return 3.0


def _score_roe(roe: float) -> float:
    """returnOnEquity. O'Neil wants ≥17%."""
    if roe is None:
        return 7.5
    roe_num = roe * 100 if abs(roe) <= 1 else roe
    if roe_num >= 25:
        return 15.0
    if roe_num >= 17:
        return 12.0
    if roe_num >= 12:
        return 8.0
    if roe_num >= 5:
        return 4.0
    return 0.0


def _compute_market_return_12m() -> float:
    """NIFTY50 12-month return in %. Cached via fetch_ohlcv parquet. 0.0 on failure."""
    try:
        frames = fetch_ohlcv([NIFTY50_SYMBOL], period="1y", interval="1d")
        df = frames.get(NIFTY50_SYMBOL)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        close = df["Close"].astype(float)
        return (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
    except Exception as e:
        log.debug("market return calc failed: %s", e)
        return 0.0


def score_ticker(
    yahoo_ticker: str,
    ohlcv_df: Optional[pd.DataFrame],
    yf_ticker_obj,
    market_return_12m_pct: float,
) -> dict[str, Any]:
    """
    Compute CANSLIM-inspired score for one ticker.

    Args:
        yahoo_ticker:   e.g. "RELIANCE.NS"
        ohlcv_df:       DataFrame indexed by date with OHLCV columns (may be None)
        yf_ticker_obj:  pre-constructed yfinance.Ticker (so caller controls session)
        market_return_12m_pct: precomputed NIFTY50 12M return to avoid N refetches

    Returns:
        {
            "score": 0-100,
            "breakdown": {...},
            "metrics": { raw inputs for debugging }
        }
    Never raises. Missing/broken data produces a neutral 50.
    """
    try:
        info = yf_ticker_obj.info or {}
    except Exception as e:
        log.debug("%s info fetch failed: %s", yahoo_ticker, e)
        info = {}

    # Price metrics
    if ohlcv_df is not None and not ohlcv_df.empty:
        close = safe_float(ohlcv_df["Close"].iloc[-1], 0.0)
        high_52w = safe_float(ohlcv_df["High"].tail(252).max(), 0.0)
        avg_vol = safe_float(ohlcv_df["Volume"].tail(20).mean(), 0.0)
        avg_dollar_vol_cr = (avg_vol * close) / 1e7  # rupees → crore
        if len(ohlcv_df) >= 252:
            ret_12m = (close / safe_float(ohlcv_df["Close"].iloc[-252], close) - 1.0) * 100.0
        else:
            ret_12m = 0.0
    else:
        close = safe_float(info.get("currentPrice"), 0.0)
        high_52w = safe_float(info.get("fiftyTwoWeekHigh"), 0.0)
        avg_dollar_vol_cr = 0.0
        ret_12m = 0.0

    # Fundamentals
    eps_qoq = _get_quarterly_eps_growth(yf_ticker_obj)
    eps_cagr = _get_annual_eps_cagr(yf_ticker_obj)
    roe = info.get("returnOnEquity")
    inst_pct = info.get("heldPercentInstitutions")

    breakdown = OneilBreakdown(
        eps_growth_qoq=_score_eps_growth_qoq(eps_qoq),
        eps_growth_annual=_score_eps_growth_annual(eps_cagr),
        proximity_52w=_score_proximity_52w(close, high_52w),
        liquidity=_score_liquidity(avg_dollar_vol_cr),
        relative_strength=_score_relative_strength(ret_12m, market_return_12m_pct),
        institutional=_score_institutional(inst_pct),
        roe=_score_roe(roe),
    )

    total = breakdown.total()
    # If we had almost no data, fall back to neutral 50 instead of 0
    if total < 5 and eps_qoq == 0 and eps_cagr == 0 and close == 0:
        total = 50.0

    return {
        "score": round(total, 2),
        "breakdown": {
            "eps_growth_qoq": breakdown.eps_growth_qoq,
            "eps_growth_annual": breakdown.eps_growth_annual,
            "proximity_52w": breakdown.proximity_52w,
            "liquidity": breakdown.liquidity,
            "relative_strength": breakdown.relative_strength,
            "institutional": breakdown.institutional,
            "roe": breakdown.roe,
        },
        "metrics": {
            "eps_growth_qoq_pct": round(eps_qoq, 2),
            "eps_cagr_3y_pct": round(eps_cagr, 2),
            "close": round(close, 2),
            "high_52w": round(high_52w, 2),
            "avg_daily_turnover_cr": round(avg_dollar_vol_cr, 2),
            "return_12m_pct": round(ret_12m, 2),
            "return_vs_nifty_pct": round(ret_12m - market_return_12m_pct, 2),
            "roe": roe,
            "institutional_pct": inst_pct,
        },
    }


def precompute_market_context() -> dict[str, float]:
    """Call once at pipeline start. Returns shared context all tickers need."""
    return {"market_return_12m_pct": _compute_market_return_12m()}
