"""
pipeline/risk_penalties.py — penalties subtracted from composite score.

We score positives in technical/oneil/news. We subtract here. The separation
matters because penalties represent *conditions we avoid* (not weak signals),
and they should show up in the frontend breakdown as a negative chip.

Penalties cap at -25 in aggregate so a truly toxic stock gets pushed out of
the top picks, but a single red flag doesn't disqualify an otherwise-strong
leader.

Flags:
    overbought_rsi       RSI(14) > 78                   → -8
    below_sma200         close < SMA200                  → -6
    elevated_atr         ATR%(14) > 6% (daily)           → -5
    earnings_blackout    earnings date within ±3 days    → -6
    wide_dd              drawdown from 52w high > 35%    → -4
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.utils import get_logger, safe_float, IST

log = get_logger("risk")


@dataclass
class PenaltyResult:
    total: float          # negative number (or 0)
    flags: dict[str, float]
    notes: list[str]


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as percent of close. Classic Wilder. Returns 0 on failure."""
    if len(df) < period + 1:
        return 0.0
    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return safe_float(atr / close.iloc[-1] * 100.0, 0.0)
    except Exception:
        return 0.0


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Copy of technical_scorer RSI. Kept local to avoid cross-module import cycles."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi.iloc[-1], 50.0)


def _days_to_earnings(yf_ticker_obj) -> Optional[int]:
    """
    Return absolute days to the next earnings date (past or future).
    None if yfinance doesn't return one. We penalize blackout ± 3 days.
    """
    try:
        cal = yf_ticker_obj.calendar
        if cal is None:
            return None
        # Can be a DataFrame (newer) or dict (older)
        earnings_date = None
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if isinstance(earnings_date, list) and earnings_date:
                earnings_date = earnings_date[0]
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"]
            earnings_date = val.iloc[0] if hasattr(val, "iloc") else val

        if earnings_date is None:
            return None

        if isinstance(earnings_date, str):
            earnings_date = datetime.fromisoformat(earnings_date.replace("Z", "+00:00"))
        if isinstance(earnings_date, pd.Timestamp):
            earnings_date = earnings_date.to_pydatetime()

        now = datetime.now(IST).replace(tzinfo=None)
        if hasattr(earnings_date, "replace"):
            try:
                earnings_date = earnings_date.replace(tzinfo=None)
            except Exception:
                pass
        delta = abs((earnings_date - now).days)
        return delta
    except Exception as e:
        log.debug("earnings date lookup failed: %s", e)
        return None


def apply_penalties(
    ohlcv_df: Optional[pd.DataFrame],
    yf_ticker_obj=None,
) -> PenaltyResult:
    """
    Compute total penalty (≤0) plus per-flag breakdown and human-readable notes.
    Never raises.
    """
    flags: dict[str, float] = {}
    notes: list[str] = []

    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 30:
        return PenaltyResult(total=0.0, flags={}, notes=["insufficient_history"])

    try:
        close = ohlcv_df["Close"].astype(float)
        high = ohlcv_df["High"].astype(float)

        # 1. Overbought RSI
        rsi_val = _rsi(close)
        if rsi_val > 78:
            flags["overbought_rsi"] = -8.0
            notes.append(f"RSI extended at {rsi_val:.0f}")

        # 2. Below SMA200
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            if close.iloc[-1] < sma200:
                flags["below_sma200"] = -6.0
                notes.append("Trading below 200-day MA")

        # 3. Elevated ATR
        atr_p = _atr_pct(ohlcv_df, 14)
        if atr_p > 6.0:
            flags["elevated_atr"] = -5.0
            notes.append(f"High volatility (ATR {atr_p:.1f}%)")

        # 4. Drawdown from 52w high
        if len(high) >= 60:
            hi_52w = high.tail(252).max()
            if hi_52w > 0:
                dd = (close.iloc[-1] / hi_52w - 1.0) * 100.0
                if dd < -35:
                    flags["wide_dd"] = -4.0
                    notes.append(f"Down {dd:.0f}% from 52w high")

        # 5. Earnings blackout
        if yf_ticker_obj is not None:
            dte = _days_to_earnings(yf_ticker_obj)
            if dte is not None and dte <= 3:
                flags["earnings_blackout"] = -6.0
                notes.append(f"Earnings in ~{dte}d")

    except Exception as e:
        log.warning("penalty computation partial failure: %s", e)

    total = sum(flags.values())
    # Cap total penalty at -25
    if total < -25:
        total = -25
    return PenaltyResult(total=round(total, 2), flags=flags, notes=notes)
