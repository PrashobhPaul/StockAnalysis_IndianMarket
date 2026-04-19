"""
pipeline/final_scorer.py — combine Tech + CANSLIM + News, gate on backtest.

This is the final ranking layer. It produces the `score_breakdown` dict that
the frontend's 3 new chips render.

Composite formula (regime-adjusted):

    BULL regime:
        composite = 0.40·tech + 0.30·canslim + 0.20·news + 0.10·bt_winrate + penalties
    SIDEWAYS:
        composite = 0.35·tech + 0.30·canslim + 0.25·news + 0.10·bt_winrate + penalties
    BEAR:
        composite = 0.25·tech + 0.35·canslim + 0.25·news + 0.15·bt_winrate + penalties

Gate: stocks with backtest_winrate < 60% are dropped from final picks list
(they remain in the full scored universe for debugging, flagged `gated=true`).

Emits per-ticker:
    {
        symbol, name, composite_score, gated,
        winrate_pct, score_breakdown: {
            technical, canslim, news, penalties, winrate_pct
        },
        components: { full nested dicts for power users },
        conviction_tier: "HIGH" | "MEDIUM" | "LOW"
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pipeline.utils import get_logger

log = get_logger("final")

WINRATE_GATE_PCT = 60.0

WEIGHTS = {
    "BULL":     {"tech": 0.40, "canslim": 0.30, "news": 0.20, "winrate": 0.10},
    "SIDEWAYS": {"tech": 0.35, "canslim": 0.30, "news": 0.25, "winrate": 0.10},
    "BEAR":     {"tech": 0.25, "canslim": 0.35, "news": 0.25, "winrate": 0.15},
}


def _conviction_tier(composite: float, winrate: float) -> str:
    """HIGH ≥75 & wr≥70 ; MEDIUM ≥60 & wr≥60 ; LOW otherwise."""
    if composite >= 75 and winrate >= 70:
        return "HIGH"
    if composite >= 60 and winrate >= 60:
        return "MEDIUM"
    return "LOW"


def score_ticker_final(
    ticker_display: str,
    name: str,
    tech_result: dict[str, Any],
    oneil_result: dict[str, Any],
    news_result: Any,             # TickerNewsScore
    penalty_result: Any,          # PenaltyResult
    backtest_result: Any,         # TickerBacktestResult
    regime_label: str,
) -> dict[str, Any]:
    """
    Build the per-ticker final record. Weights vary by regime.

    All score-component values are normalized to 0-100 so weights are
    interpretable. Penalties are added as a (negative) post-term, not
    weighted — a penalty should show up in full.
    """
    w = WEIGHTS.get(regime_label, WEIGHTS["SIDEWAYS"])

    tech_s = float(tech_result.get("score", 50.0))
    canslim_s = float(oneil_result.get("score", 50.0))
    news_s = float(getattr(news_result, "score_0_100", 50.0))
    winrate = float(getattr(backtest_result, "win_rate_pct", 0.0))
    penalty = float(getattr(penalty_result, "total", 0.0))

    composite = (
        w["tech"] * tech_s
        + w["canslim"] * canslim_s
        + w["news"] * news_s
        + w["winrate"] * winrate
        + penalty
    )
    composite = max(0.0, min(100.0, composite))

    gated = winrate < WINRATE_GATE_PCT
    tier = _conviction_tier(composite, winrate)

    return {
        "symbol": ticker_display,
        "name": name,
        "composite_score": round(composite, 2),
        "gated": gated,
        "gate_reason": (
            f"backtest_winrate {winrate:.0f}% < {WINRATE_GATE_PCT:.0f}%"
            if gated else None
        ),
        "conviction_tier": tier,
        "regime_used": regime_label,
        "weights_used": w,
        # Frontend chips read this flat dict
        "score_breakdown": {
            "technical": round(tech_s, 2),
            "canslim": round(canslim_s, 2),
            "news": round(news_s, 2),
            "penalties": round(penalty, 2),
            "winrate_pct": round(winrate, 2),
        },
        # Full components for debug / narrative
        "components": {
            "technical": tech_result,
            "canslim": oneil_result,
            "news": {
                "score_0_100": getattr(news_result, "score_0_100", 50.0),
                "raw_score": getattr(news_result, "raw_score", 0.0),
                "headline_count": getattr(news_result, "headline_count", 0),
                "top_headlines": getattr(news_result, "top_headlines", []),
                "method": getattr(news_result, "method", "unknown"),
            },
            "penalties": {
                "total": getattr(penalty_result, "total", 0.0),
                "flags": getattr(penalty_result, "flags", {}),
                "notes": getattr(penalty_result, "notes", []),
            },
            "backtest": {
                "win_rate_pct": winrate,
                "per_horizon_win_rate": getattr(backtest_result, "per_horizon_win_rate", {}),
                "opportunities": getattr(backtest_result, "opportunities", 0),
                "avg_forward_return_pct": getattr(backtest_result, "avg_forward_return_pct", 0.0),
            },
        },
    }


def rank_and_select(
    scored: list[dict[str, Any]],
    top_n: int = 15,
    min_conviction: str = "MEDIUM",
) -> dict[str, list[dict[str, Any]]]:
    """
    Sort full scored universe into picks vs gated.

    Returns:
        {
            "picks":       non-gated, ≥ min_conviction, sorted by composite desc
            "near_misses": non-gated but conviction<min (the watchlist)
            "gated":       failed the backtest gate (reference only)
        }
    """
    tier_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_order = tier_order.get(min_conviction, 2)

    picks: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    gated: list[dict[str, Any]] = []

    for rec in scored:
        if rec.get("gated"):
            gated.append(rec)
        elif tier_order.get(rec.get("conviction_tier", "LOW"), 1) >= min_order:
            picks.append(rec)
        else:
            near.append(rec)

    picks.sort(key=lambda r: r["composite_score"], reverse=True)
    near.sort(key=lambda r: r["composite_score"], reverse=True)
    gated.sort(key=lambda r: r["composite_score"], reverse=True)

    return {
        "picks": picks[:top_n],
        "near_misses": near[:top_n],
        "gated": gated[:50],  # keep top-50 gated for debugging visibility
    }
