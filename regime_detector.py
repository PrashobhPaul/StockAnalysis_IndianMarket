"""
pipeline/regime_detector.py — market regime classification.

A single label (BULL / BEAR / SIDEWAYS) that downstream scorers use to adjust
weights. We keep this deliberately simple: two moving-average gates on NIFTY50
plus a 3-month return check. Over-engineering a regime model gives false
confidence — the goal is to tilt scoring, not to predict.

Rules:
    BULL     → close > SMA50  AND close > SMA200  AND ret_3m > 0
    BEAR     → close < SMA200 AND ret_3m < 0
    SIDEWAYS → everything else

If the NIFTY50 fetch fails we return SIDEWAYS with a warning. Never raise.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from pipeline.utils import get_logger, fetch_ohlcv, safe_float

log = get_logger("regime")

NIFTY50_SYMBOL = "^NSEI"  # Yahoo ticker for NIFTY 50


@dataclass
class Regime:
    label: str            # "BULL" | "BEAR" | "SIDEWAYS"
    nifty_close: float
    sma50: float
    sma200: float
    return_3m_pct: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sideways(reason: str) -> Regime:
    """Safe default when we can't compute. Downstream treats SIDEWAYS as neutral."""
    log.warning("regime → SIDEWAYS (fallback): %s", reason)
    return Regime(
        label="SIDEWAYS",
        nifty_close=0.0,
        sma50=0.0,
        sma200=0.0,
        return_3m_pct=0.0,
        rationale=f"Fallback: {reason}",
    )


def detect_regime() -> Regime:
    """
    Classify current market regime using NIFTY50 daily OHLCV.

    Fetches ~260 trading days (≈1 year) which is enough for SMA200 and 3M return.
    Never raises — all failure paths return a SIDEWAYS regime with rationale.
    """
    try:
        frames = fetch_ohlcv([NIFTY50_SYMBOL], period="1y", interval="1d")
    except Exception as e:
        return _sideways(f"OHLCV fetch threw: {e}")

    df = frames.get(NIFTY50_SYMBOL)
    if df is None or df.empty or len(df) < 200:
        return _sideways(f"Insufficient NIFTY50 history (rows={0 if df is None else len(df)})")

    try:
        close = df["Close"].astype(float)
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        last_close = close.iloc[-1]

        # 3-month return: ~63 trading days
        if len(close) >= 63:
            ret_3m = (last_close / close.iloc[-63] - 1.0) * 100.0
        else:
            ret_3m = 0.0

        last_close = safe_float(last_close, 0.0)
        sma50 = safe_float(sma50, 0.0)
        sma200 = safe_float(sma200, 0.0)
        ret_3m = safe_float(ret_3m, 0.0)
    except Exception as e:
        return _sideways(f"Indicator math failed: {e}")

    # Classification
    if last_close > sma50 and last_close > sma200 and ret_3m > 0:
        label = "BULL"
        rationale = (
            f"NIFTY {last_close:.0f} above 50DMA ({sma50:.0f}) and 200DMA "
            f"({sma200:.0f}); 3M return {ret_3m:+.1f}%"
        )
    elif last_close < sma200 and ret_3m < 0:
        label = "BEAR"
        rationale = (
            f"NIFTY {last_close:.0f} below 200DMA ({sma200:.0f}); "
            f"3M return {ret_3m:+.1f}%"
        )
    else:
        label = "SIDEWAYS"
        rationale = (
            f"NIFTY {last_close:.0f}, 50DMA {sma50:.0f}, 200DMA {sma200:.0f}, "
            f"3M {ret_3m:+.1f}% — mixed signals"
        )

    log.info("regime → %s :: %s", label, rationale)
    return Regime(
        label=label,
        nifty_close=round(last_close, 2),
        sma50=round(sma50, 2),
        sma200=round(sma200, 2),
        return_3m_pct=round(ret_3m, 2),
        rationale=rationale,
    )


if __name__ == "__main__":
    import json
    r = detect_regime()
    print(json.dumps(r.to_dict(), indent=2))
