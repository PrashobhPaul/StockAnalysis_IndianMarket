"""
pipeline/technical_scorer.py — rule-based technical scoring (0-100).

This is the refactored descendant of v2's analyze.py. Same philosophy —
transparent, explainable, no ML — but cleaner component isolation so
final_scorer.py can show per-category chips on the frontend.

Five independent sub-scores, each 0-20, summed into 0-100:

    trend          — price vs SMA50/SMA200, SMA50 vs SMA200 ordering
    momentum       — RSI(14), MACD histogram, rate of change
    volume         — current vs 20D avg, OBV slope
    breakout       — proximity to 52w high, consolidation breaks
    price_action   — last N candle structure, higher-highs/higher-lows

All raw math is done from OHLCV (no talib dependency — pandas rolling is
fast enough for 500 tickers once a day). Never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.utils import get_logger, safe_float

log = get_logger("tech")


@dataclass
class TechBreakdown:
    trend: float          # 0-20
    momentum: float       # 0-20
    volume: float         # 0-20
    breakout: float       # 0-20
    price_action: float   # 0-20

    def total(self) -> float:
        return self.trend + self.momentum + self.volume + self.breakout + self.price_action


# ───────────────────────── Indicators ─────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> float:
    """Classic Wilder RSI. Returns last value. Neutral 50 on short series."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return safe_float(val, 50.0)


def _macd_histogram(close: pd.Series) -> float:
    """MACD histogram last value. 0 if insufficient data."""
    if len(close) < 35:
        return 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return safe_float(hist.iloc[-1], 0.0)


def _roc(close: pd.Series, period: int = 10) -> float:
    """Rate of change % over `period` bars."""
    if len(close) < period + 1:
        return 0.0
    return safe_float((close.iloc[-1] / close.iloc[-period - 1] - 1.0) * 100.0, 0.0)


def _obv_slope(close: pd.Series, volume: pd.Series, lookback: int = 20) -> float:
    """OBV slope over `lookback` bars, normalized. Positive = accumulation."""
    if len(close) < lookback + 1:
        return 0.0
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    recent = obv.tail(lookback)
    if recent.std() == 0:
        return 0.0
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    return safe_float(slope / (recent.abs().mean() + 1e-9), 0.0)


# ───────────────────────── Component scores ─────────────────────────
def _score_trend(close: pd.Series) -> float:
    """
    Trend score 0-20. Award points for:
      - close > SMA50        (+6)
      - close > SMA200       (+6)
      - SMA50 > SMA200       (+5)  — golden-cross state
      - SMA50 rising         (+3)  — fresh trend
    """
    if len(close) < 200:
        return 10.0
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    last_close = close.iloc[-1]
    score = 0.0
    if last_close > sma50.iloc[-1]:
        score += 6
    if last_close > sma200.iloc[-1]:
        score += 6
    if sma50.iloc[-1] > sma200.iloc[-1]:
        score += 5
    if sma50.iloc[-1] > sma50.iloc[-5]:
        score += 3
    return score


def _score_momentum(close: pd.Series) -> float:
    """
    Momentum 0-20. RSI sweet spot is 55-70 (strong but not overbought).
    MACD histogram positive = bullish; ROC(10) > 5% = accelerating.
    """
    score = 0.0
    rsi = _rsi(close)
    if 55 <= rsi <= 70:
        score += 8
    elif 50 <= rsi < 55 or 70 < rsi <= 75:
        score += 5
    elif rsi > 75 or rsi < 40:
        score += 1  # extremes get a token point
    else:
        score += 3

    hist = _macd_histogram(close)
    if hist > 0:
        score += 6
    elif hist > -0.5:
        score += 3

    roc = _roc(close, 10)
    if roc >= 10:
        score += 6
    elif roc >= 5:
        score += 4
    elif roc >= 0:
        score += 2

    return min(score, 20.0)


def _score_volume(volume: pd.Series, close: pd.Series) -> float:
    """
    Volume 0-20. Today vs 20D avg, plus OBV slope.
    Big volume on up days is the bullish signal we want.
    """
    if len(volume) < 21:
        return 10.0
    avg20 = volume.tail(21).iloc[:-1].mean()
    today = volume.iloc[-1]
    ratio = today / avg20 if avg20 > 0 else 1.0

    score = 0.0
    if ratio >= 2.0:
        score += 10
    elif ratio >= 1.5:
        score += 7
    elif ratio >= 1.0:
        score += 4
    else:
        score += 2

    # Was today an up day? Volume only matters in context.
    if len(close) >= 2 and close.iloc[-1] > close.iloc[-2]:
        score += 2

    obv_s = _obv_slope(close, volume, 20)
    if obv_s > 0.02:
        score += 8
    elif obv_s > 0:
        score += 5
    elif obv_s > -0.02:
        score += 2

    return min(score, 20.0)


def _score_breakout(close: pd.Series, high: pd.Series) -> float:
    """
    Breakout 0-20. Proximity to 52w high + whether we're breaking prior
    consolidation. Very close to 52w high = strongest signal.
    """
    if len(close) < 60:
        return 10.0
    last_close = close.iloc[-1]
    hi_52w = high.tail(252).max() if len(high) >= 252 else high.max()
    proximity = last_close / hi_52w if hi_52w > 0 else 0

    score = 0.0
    if proximity >= 1.0:
        score += 12  # new 52w high
    elif proximity >= 0.97:
        score += 10
    elif proximity >= 0.90:
        score += 6
    elif proximity >= 0.80:
        score += 3

    # 20D consolidation break: is today > 20D high excluding today?
    if len(high) >= 21:
        prior_20d_high = high.tail(21).iloc[:-1].max()
        if last_close > prior_20d_high:
            score += 8
        elif last_close > prior_20d_high * 0.98:
            score += 4

    return min(score, 20.0)


def _score_price_action(df: pd.DataFrame) -> float:
    """
    Price action 0-20. Last 10 bars: higher-highs/lows count, positive close streak,
    candle body strength (close near high = buying pressure).
    """
    if len(df) < 10:
        return 10.0
    tail = df.tail(10)
    score = 0.0

    # Higher highs
    hh = sum(tail["High"].iloc[i] > tail["High"].iloc[i - 1] for i in range(1, len(tail)))
    score += min(hh, 5)

    # Higher lows
    hl = sum(tail["Low"].iloc[i] > tail["Low"].iloc[i - 1] for i in range(1, len(tail)))
    score += min(hl, 5)

    # Positive-close streak (most recent)
    streak = 0
    for i in range(len(tail) - 1, 0, -1):
        if tail["Close"].iloc[i] > tail["Close"].iloc[i - 1]:
            streak += 1
        else:
            break
    score += min(streak, 5)

    # Candle strength today: close in top 30% of the day's range?
    today = tail.iloc[-1]
    rng = safe_float(today["High"] - today["Low"], 0.0)
    if rng > 0:
        pos = (safe_float(today["Close"], 0.0) - safe_float(today["Low"], 0.0)) / rng
        if pos >= 0.7:
            score += 5
        elif pos >= 0.5:
            score += 3

    return min(score, 20.0)


# ───────────────────────── Public API ─────────────────────────
def score_ticker(ohlcv_df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """
    Score one ticker's technicals. Returns 0-100 plus breakdown.
    Missing data → neutral 50 (not 0) so we don't unfairly punish fresh IPOs.
    """
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 30:
        return {
            "score": 50.0,
            "breakdown": {
                "trend": 10.0, "momentum": 10.0, "volume": 10.0,
                "breakout": 10.0, "price_action": 10.0,
            },
            "note": "insufficient_history",
        }

    try:
        close = ohlcv_df["Close"].astype(float)
        high = ohlcv_df["High"].astype(float)
        volume = ohlcv_df["Volume"].astype(float)

        br = TechBreakdown(
            trend=_score_trend(close),
            momentum=_score_momentum(close),
            volume=_score_volume(volume, close),
            breakout=_score_breakout(close, high),
            price_action=_score_price_action(ohlcv_df),
        )

        return {
            "score": round(br.total(), 2),
            "breakdown": {
                "trend": round(br.trend, 2),
                "momentum": round(br.momentum, 2),
                "volume": round(br.volume, 2),
                "breakout": round(br.breakout, 2),
                "price_action": round(br.price_action, 2),
            },
            "indicators": {
                "rsi_14": round(_rsi(close), 2),
                "macd_hist": round(_macd_histogram(close), 4),
                "roc_10": round(_roc(close, 10), 2),
            },
        }
    except Exception as e:
        log.warning("technical scoring failed: %s", e)
        return {
            "score": 50.0,
            "breakdown": {
                "trend": 10.0, "momentum": 10.0, "volume": 10.0,
                "breakout": 10.0, "price_action": 10.0,
            },
            "note": f"error: {e}",
        }
