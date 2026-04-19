"""
pipeline/backtester.py — 3-year walk-forward backtest per ticker.

The gate: only tickers whose historical hit-rate ≥ 60% can become picks.
This is the single most important change from v2 — it forces honesty about
which scoring signals actually predicted moves in THIS stock's history.

Mechanics:
    • For each month-end in the 3-year window, compute the same composite
      score using data available AT THAT POINT (no look-ahead).
    • Take the top-10% scored stocks as hypothetical picks.
    • Check forward rolling-max over horizons (10d, 30d, 60d):
        target_hit = forward_max / entry_price - 1 >= horizon_target
        10d  → 5%
        30d  → 10%
        60d  → 15%
    • Aggregate per ticker: win_rate_pct = hits / opportunities.
    • Composite win-rate is a weighted avg across horizons (40/35/25).

This is vectorized pandas, not an event-driven backtester — we're not
modeling slippage or commissions because gating at 60% gives us ~15% margin
for those frictions and we only need relative ranking.

Parallelism: ThreadPoolExecutor(max_workers=4). Each worker reads its own
ticker's parquet slice so there's no shared mutable state.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.utils import get_logger, safe_float, fetch_ohlcv

log = get_logger("backtest")

HORIZONS = [
    ("10d", 10, 5.0),    # 5% target in 10 trading days
    ("30d", 30, 10.0),
    ("60d", 60, 15.0),
]
HORIZON_WEIGHTS = {"10d": 0.40, "30d": 0.35, "60d": 0.25}

# Rebalance points: last trading day of each month for 36 months
REBALANCE_FREQ_DAYS = 21  # ~monthly, in trading days


@dataclass
class TickerBacktestResult:
    ticker: str
    win_rate_pct: float                 # weighted composite
    per_horizon_win_rate: dict[str, float]
    opportunities: int                  # total rebalances evaluated
    avg_forward_return_pct: float       # across all rebalance points
    method: str = "walk_forward_3y"


def _simple_score(df_slice: pd.DataFrame) -> float:
    """
    Lightweight score function used during backtest — NOT the full composite.
    We can't run FinBERT historically (no archived news) and can't get
    point-in-time fundamentals from yfinance. So we use a robust technical
    proxy: proximity to 52w high + trend + momentum.

    This is deliberately simpler than technical_scorer.score_ticker — the
    point is to test whether *any* systematic ranking worked for this ticker
    historically. If a stock has never responded to technical breakouts,
    today's technical score is meaningless regardless of FinBERT.
    """
    if len(df_slice) < 60:
        return 50.0
    try:
        close = df_slice["Close"].astype(float)
        high = df_slice["High"].astype(float)

        last = close.iloc[-1]
        sma50 = close.tail(50).mean()
        sma200 = close.tail(200).mean() if len(close) >= 200 else sma50

        # Trend (0-40)
        trend = 0.0
        if last > sma50:
            trend += 15
        if last > sma200:
            trend += 15
        if sma50 > sma200:
            trend += 10

        # Momentum (0-30): RSI sweet spot + ROC
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss > 0 else 2.0
        rsi = 100 - (100 / (1 + rs))
        if 55 <= rsi <= 70:
            trend += 20
        elif 50 <= rsi < 55 or 70 < rsi <= 75:
            trend += 12
        else:
            trend += 5

        roc10 = (last / close.iloc[-11] - 1.0) * 100.0 if len(close) >= 11 else 0
        if roc10 >= 5:
            trend += 10
        elif roc10 >= 0:
            trend += 5

        # Proximity to 52w high (0-30)
        hi = high.tail(252).max()
        prox = last / hi if hi > 0 else 0
        if prox >= 0.98:
            trend += 30
        elif prox >= 0.90:
            trend += 20
        elif prox >= 0.80:
            trend += 10

        return max(0, min(100, trend))
    except Exception:
        return 50.0


def _backtest_one(ticker: str, df: pd.DataFrame) -> TickerBacktestResult:
    """
    Walk-forward backtest for a single ticker's 3-year OHLCV.

    For every ~month (21-bar stride), if the simple_score at that bar was
    "high" (≥70), treat it as a hypothetical buy and check 10d/30d/60d
    forward returns against their respective targets.

    We don't compare this ticker to a universe — we ask "when THIS stock
    scored high, did THIS stock subsequently rise?" A 60%+ hit rate means
    the signal has worked for this specific name. Different stocks respond
    to different signals, and the backtest filters for stocks that do.
    """
    if df is None or df.empty or len(df) < 300:
        return TickerBacktestResult(
            ticker=ticker,
            win_rate_pct=0.0,
            per_horizon_win_rate={h: 0.0 for h, _, _ in HORIZONS},
            opportunities=0,
            avg_forward_return_pct=0.0,
            method="insufficient_history",
        )

    df = df.sort_index()
    close = df["Close"].astype(float).values
    n = len(df)

    hits = {h: 0 for h, _, _ in HORIZONS}
    opps = {h: 0 for h, _, _ in HORIZONS}
    fwd_returns: list[float] = []

    # Start scanning after we have 200 bars of warmup for scoring
    i = 200
    while i < n - max(h for _, h, _ in HORIZONS):
        slice_df = df.iloc[max(0, i - 252):i + 1]
        score = _simple_score(slice_df)
        if score >= 70:
            entry = close[i]
            if entry > 0:
                for name, horizon, tgt_pct in HORIZONS:
                    end = min(i + horizon, n - 1)
                    if end <= i:
                        continue
                    window_high = close[i + 1:end + 1].max()
                    fwd_max_pct = (window_high / entry - 1.0) * 100.0
                    opps[name] += 1
                    if fwd_max_pct >= tgt_pct:
                        hits[name] += 1
                    if name == "30d":
                        fwd_returns.append(fwd_max_pct)
        i += REBALANCE_FREQ_DAYS

    per_h: dict[str, float] = {}
    for name, _, _ in HORIZONS:
        per_h[name] = (hits[name] / opps[name] * 100.0) if opps[name] > 0 else 0.0

    if sum(opps.values()) == 0:
        composite = 0.0
    else:
        composite = sum(per_h[name] * HORIZON_WEIGHTS[name] for name in per_h)

    total_opps = max(opps.values()) if opps else 0
    avg_fwd = float(np.mean(fwd_returns)) if fwd_returns else 0.0

    return TickerBacktestResult(
        ticker=ticker,
        win_rate_pct=round(composite, 2),
        per_horizon_win_rate={k: round(v, 2) for k, v in per_h.items()},
        opportunities=total_opps,
        avg_forward_return_pct=round(avg_fwd, 2),
    )


def backtest_universe(
    tickers: list[str],
    ohlcv_map: Optional[dict[str, pd.DataFrame]] = None,
    max_workers: int = 4,
) -> dict[str, TickerBacktestResult]:
    """
    Run walk-forward backtest for every ticker.

    If `ohlcv_map` is None we fetch 3y history ourselves (parquet-cached).
    Using the cache matters — this runs on every daily job and refetching
    500 × 3y bars from Yahoo would burn the rate limit.
    """
    if ohlcv_map is None:
        log.info("Backtest: fetching 3y OHLCV for %d tickers", len(tickers))
        ohlcv_map = fetch_ohlcv(tickers, period="3y", interval="1d")

    results: dict[str, TickerBacktestResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_backtest_one, tk, ohlcv_map.get(tk)): tk
            for tk in tickers
        }
        for idx, fut in enumerate(as_completed(futures), 1):
            tk = futures[fut]
            try:
                results[tk] = fut.result()
            except Exception as e:
                log.warning("backtest failed for %s: %s", tk, e)
                results[tk] = TickerBacktestResult(
                    ticker=tk,
                    win_rate_pct=0.0,
                    per_horizon_win_rate={h: 0.0 for h, _, _ in HORIZONS},
                    opportunities=0,
                    avg_forward_return_pct=0.0,
                    method="error",
                )
            if idx % 50 == 0:
                log.info("backtest progress: %d/%d", idx, len(tickers))

    passing = sum(1 for r in results.values() if r.win_rate_pct >= 60.0)
    log.info("backtest done: %d/%d tickers ≥ 60%% win-rate",
             passing, len(results))
    return results
