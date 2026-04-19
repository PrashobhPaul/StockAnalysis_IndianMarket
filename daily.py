"""
api/daily.py — the daily ProfitPilot v3 pipeline orchestrator.

Entrypoint: `python -m api.daily` (called from .github/workflows/daily.yml)

Stages (each independently degrades; a single stage failing does not abort):

    1. Build universe               (universe_builder)
    2. Detect market regime         (regime_detector)
    3. Fetch 3y OHLCV once          (utils.fetch_ohlcv, parquet-cached)
    4. Run walk-forward backtest    (backtester)
    5. Score each ticker:
         a. technical               (technical_scorer)
         b. CANSLIM                 (oneil_scorer)
         c. penalties               (risk_penalties)
         d. news / FinBERT          (finbert_news)
    6. Combine → final composite    (final_scorer)
    7. Build narratives             (narrative_v3)
    8. Generate daily brief         (daily_brief)
    9. Write predictions.json + news_cache.json atomically

Preserve-last-good: if universe build returns 0 tickers OR total failure
across >=50% of tickers, we load the previous predictions.json, mark
`_stale: true`, bump the generated_at, and exit 0. This keeps the static
GitHub Pages dashboard showing yesterday's data rather than a blank page.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.utils import (
    get_logger,
    atomic_write_json,
    fetch_ohlcv,
    chrome_session,
    REPO_ROOT,
    IST,
)
from pipeline.universe_builder import build_universe
from pipeline.regime_detector import detect_regime
from pipeline.technical_scorer import score_ticker as score_tech
from pipeline.oneil_scorer import score_ticker as score_oneil, precompute_market_context
from pipeline.risk_penalties import apply_penalties
from pipeline.finbert_news import score_universe_news
from pipeline.backtester import backtest_universe
from pipeline.final_scorer import score_ticker_final, rank_and_select
from pipeline.narrative_v3 import attach_narratives
from pipeline.daily_brief import build_brief

log = get_logger("orchestrator")

PREDICTIONS_PATH = REPO_ROOT / "predictions.json"
NEWS_CACHE_PATH = REPO_ROOT / "news_cache.json"

# Tuning knobs — exposed here so a bad day's run can be reproduced
MAX_UNIVERSE_SIZE = 500       # cap — we don't need all 1250 in the index CSVs
SCORING_WORKERS = 4           # per-ticker tech+oneil+penalty is IO-light; 4 is safe
WRITE_FAILURE_RATE_CAP = 0.5  # if ≥50% tickers fail scoring, preserve last-good


def _load_last_good() -> dict[str, Any] | None:
    """Load previous predictions.json for preserve-last-good."""
    if not PREDICTIONS_PATH.exists():
        return None
    try:
        with open(PREDICTIONS_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load last-good predictions: %s", e)
        return None


def _score_one_ticker(
    tk: dict[str, Any],
    ohlcv_map: dict,
    yf_session,
    market_ctx: dict[str, float],
) -> dict[str, Any] | None:
    """
    Run tech + oneil + penalty for a single ticker. Returns component dict
    or None on total failure. News + backtest are handled elsewhere (bulk).
    """
    try:
        import yfinance as yf
    except Exception as e:
        log.error("yfinance unavailable: %s", e)
        return None

    display = tk.get("display_symbol") or tk.get("yahoo_ticker") or tk["symbol"]
    yahoo = tk.get("yahoo_ticker") or f"{tk['symbol']}.NS"
    name = tk.get("name") or display

    df = ohlcv_map.get(yahoo)
    try:
        yf_obj = yf.Ticker(yahoo, session=yf_session)
    except Exception:
        yf_obj = yf.Ticker(yahoo)

    try:
        tech = score_tech(df)
    except Exception as e:
        log.warning("tech failed %s: %s", display, e)
        tech = {"score": 50.0, "breakdown": {}, "note": "error"}

    try:
        oneil = score_oneil(yahoo, df, yf_obj, market_ctx["market_return_12m_pct"])
    except Exception as e:
        log.warning("oneil failed %s: %s", display, e)
        oneil = {"score": 50.0, "breakdown": {}, "metrics": {}, "note": "error"}

    try:
        pen = apply_penalties(df, yf_obj)
    except Exception as e:
        log.warning("penalty failed %s: %s", display, e)
        pen = None

    return {
        "display": display,
        "yahoo": yahoo,
        "name": name,
        "sector": tk.get("sector"),
        "industry": tk.get("industry"),
        "tech": tech,
        "oneil": oneil,
        "penalty": pen,
    }


def run_pipeline() -> int:
    """Top-level orchestrator. Returns process exit code."""
    start = time.time()
    log.info("=== ProfitPilot v3 daily pipeline: START ===")

    # ── Stage 1: universe ────────────────────────────────────────────
    try:
        universe = build_universe()
    except Exception as e:
        log.error("universe build threw: %s\n%s", e, traceback.format_exc())
        universe = []

    if not universe:
        log.error("Empty universe — preserving last-good")
        return _preserve_last_good("empty_universe")

    if len(universe) > MAX_UNIVERSE_SIZE:
        log.info("Capping universe %d → %d", len(universe), MAX_UNIVERSE_SIZE)
        universe = universe[:MAX_UNIVERSE_SIZE]

    yahoo_tickers = [u.get("yahoo_ticker") or f"{u['symbol']}.NS" for u in universe]
    log.info("Universe size: %d", len(yahoo_tickers))

    # ── Stage 2: regime ──────────────────────────────────────────────
    regime = detect_regime()
    regime_d = regime.to_dict()

    # ── Stage 3: 3y OHLCV (shared across backtest + scoring) ─────────
    log.info("Fetching 3y OHLCV (parquet-cached)")
    try:
        ohlcv_3y = fetch_ohlcv(yahoo_tickers, period="3y", interval="1d")
    except Exception as e:
        log.error("OHLCV fetch threw: %s", e)
        ohlcv_3y = {}

    got_count = sum(1 for v in ohlcv_3y.values() if v is not None and not v.empty)
    log.info("OHLCV obtained for %d/%d tickers", got_count, len(yahoo_tickers))
    if got_count < len(yahoo_tickers) * (1 - WRITE_FAILURE_RATE_CAP):
        log.error("Too many OHLCV failures — preserving last-good")
        return _preserve_last_good("ohlcv_fetch_failure")

    # ── Stage 4: backtest ────────────────────────────────────────────
    log.info("Running 3y walk-forward backtest")
    backtest_map = backtest_universe(yahoo_tickers, ohlcv_map=ohlcv_3y, max_workers=4)

    # ── Stage 5: per-ticker tech + canslim + penalty ─────────────────
    market_ctx = precompute_market_context()
    yf_session = chrome_session()

    log.info("Scoring technicals/fundamentals/penalties (%d workers)", SCORING_WORKERS)
    ticker_components: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as pool:
        futures = {
            pool.submit(_score_one_ticker, tk, ohlcv_3y, yf_session, market_ctx): tk
            for tk in universe
        }
        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
                if res:
                    ticker_components[res["yahoo"]] = res
            except Exception as e:
                log.warning("scoring worker exception: %s", e)
            if idx % 50 == 0:
                log.info("scoring progress: %d/%d", idx, len(universe))

    if len(ticker_components) < len(universe) * (1 - WRITE_FAILURE_RATE_CAP):
        log.error("Too many scoring failures — preserving last-good")
        return _preserve_last_good("scoring_failure")

    # ── News / FinBERT ───────────────────────────────────────────────
    log.info("Running FinBERT news scoring")
    news_inputs = [
        (comp["display"], comp["name"])
        for comp in ticker_components.values()
    ]
    try:
        news_map = score_universe_news(news_inputs, use_finbert=True)
    except Exception as e:
        log.warning("News stage failed wholesale: %s — using neutral scores", e)
        from pipeline.finbert_news import TickerNewsScore
        news_map = {
            disp: TickerNewsScore(disp, 50.0, 0.0, 0, [], "error")
            for disp, _ in news_inputs
        }

    # ── Stage 6: final composite ─────────────────────────────────────
    log.info("Assembling final composite scores")
    scored: list[dict[str, Any]] = []
    for yahoo, comp in ticker_components.items():
        try:
            final_rec = score_ticker_final(
                ticker_display=comp["display"],
                name=comp["name"],
                tech_result=comp["tech"],
                oneil_result=comp["oneil"],
                news_result=news_map.get(comp["display"]),
                penalty_result=comp["penalty"],
                backtest_result=backtest_map.get(yahoo),
                regime_label=regime_d["label"],
            )
            final_rec["sector"] = comp.get("sector")
            final_rec["industry"] = comp.get("industry")
            scored.append(final_rec)
        except Exception as e:
            log.warning("final score failed for %s: %s", comp.get("display"), e)

    # ── Rank & select ────────────────────────────────────────────────
    ranked = rank_and_select(scored, top_n=15, min_conviction="MEDIUM")
    picks = ranked["picks"]
    near = ranked["near_misses"]
    gated = ranked["gated"]
    log.info("Ranking: %d picks, %d near-misses, %d gated",
             len(picks), len(near), len(gated))

    # ── Stage 7: narratives ──────────────────────────────────────────
    attach_narratives(picks, regime_d["label"])
    attach_narratives(near, regime_d["label"])

    # ── Stage 8: brief ───────────────────────────────────────────────
    brief = build_brief(scored, picks, near, regime_d)

    # ── Stage 9: write outputs ───────────────────────────────────────
    now_ist = datetime.now(IST).isoformat()

    predictions = {
        "generated_at_ist": now_ist,
        "pipeline_version": "v3.0",
        "regime": regime_d,
        "brief": brief,
        "picks": picks,
        "near_misses": near,
        "gated": gated,
        "counts": {
            "universe_scored": len(scored),
            "picks": len(picks),
            "near_misses": len(near),
            "gated": len(gated),
        },
        "duration_seconds": round(time.time() - start, 1),
        "_stale": False,
    }

    # News cache — separate file so frontend can lazy-load
    news_cache = {
        "generated_at_ist": now_ist,
        "tickers": {
            disp: {
                "score_0_100": getattr(ns, "score_0_100", 50.0),
                "raw_score": getattr(ns, "raw_score", 0.0),
                "headline_count": getattr(ns, "headline_count", 0),
                "top_headlines": getattr(ns, "top_headlines", []),
                "method": getattr(ns, "method", "unknown"),
            }
            for disp, ns in news_map.items()
        },
    }

    atomic_write_json(PREDICTIONS_PATH, predictions)
    atomic_write_json(NEWS_CACHE_PATH, news_cache)

    log.info("=== Pipeline DONE in %.1fs — wrote %s and %s ===",
             time.time() - start, PREDICTIONS_PATH.name, NEWS_CACHE_PATH.name)
    return 0


def _preserve_last_good(reason: str) -> int:
    """
    Write last-good predictions with _stale=true and refreshed timestamp.
    Return 0 so GitHub Actions job doesn't red-X on known-degraded days.
    """
    last = _load_last_good()
    if last is None:
        log.error("No last-good predictions file exists. Emitting minimal stub.")
        stub = {
            "generated_at_ist": datetime.now(IST).isoformat(),
            "pipeline_version": "v3.0",
            "regime": {"label": "SIDEWAYS", "rationale": "unavailable"},
            "brief": {
                "headline": "Data unavailable today — check back after the next run.",
                "action_plan": [f"Pipeline degraded: {reason}. Waiting for next scheduled run."],
            },
            "picks": [],
            "near_misses": [],
            "gated": [],
            "counts": {"universe_scored": 0, "picks": 0, "near_misses": 0, "gated": 0},
            "_stale": True,
            "_stale_reason": reason,
        }
        atomic_write_json(PREDICTIONS_PATH, stub)
        return 0

    last["_stale"] = True
    last["_stale_reason"] = reason
    last["generated_at_ist"] = datetime.now(IST).isoformat()
    atomic_write_json(PREDICTIONS_PATH, last)
    log.info("Preserved last-good (reason=%s). Frontend will show _stale badge.", reason)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_pipeline())
    except Exception as e:
        log.error("Unhandled exception in pipeline: %s\n%s", e, traceback.format_exc())
        # Even on catastrophic failure, try to preserve last-good
        sys.exit(_preserve_last_good(f"unhandled_exception: {type(e).__name__}"))
