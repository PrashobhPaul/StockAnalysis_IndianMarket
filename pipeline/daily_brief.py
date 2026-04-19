"""
pipeline/daily_brief.py — the EOD "AI Brief" block on the homepage.

Deterministic (no LLM). Produces a structured dict the frontend renders as:

    {
        headline:             short one-liner summary of the day
        key_insight:          single most important takeaway
        conviction_board:     top 3 HIGH-tier picks with one-line rationale
        risk_watchlist:       picks with penalty flags to watch
        breakout_watch:       near-misses that are close to triggering
        sector_heatmap:       sector → (avg_score, count) map
        action_plan:          3 numbered next-step items
        news_highlights:      top 5 HIGH-impact headlines across universe
    }

Important: v2 had an optional Gemini path for the brief. Prashobh explicitly
removed it — this file MUST stay rule-based. No LLM calls.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from pipeline.utils import get_logger, IST

log = get_logger("brief")


# Keep this small — frontend has limited real estate for the brief.
MAX_CONVICTION = 3
MAX_RISK = 4
MAX_BREAKOUT = 4
MAX_NEWS = 5


def _sector_of(rec: dict[str, Any]) -> str:
    """Resolve sector from any layer of the record. Fallback 'Other'."""
    for key_path in (
        ("sector",),
        ("meta", "sector"),
        ("components", "canslim", "metrics", "sector"),
    ):
        cur: Any = rec
        for k in key_path:
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                cur = None
                break
        if isinstance(cur, str) and cur:
            return cur
    return "Other"


def _headline(
    picks: list[dict[str, Any]],
    regime_label: str,
    regime_rationale: str,
) -> str:
    """One-line summary. Leads with the regime + count of HIGH convictions."""
    high = sum(1 for p in picks if p.get("conviction_tier") == "HIGH")
    med = sum(1 for p in picks if p.get("conviction_tier") == "MEDIUM")
    if regime_label == "BULL":
        tone = "Broad strength continues"
    elif regime_label == "BEAR":
        tone = "Defensive tone warranted"
    else:
        tone = "Range-bound tape"

    if high >= 3:
        return f"{tone} — {high} high-conviction setups pass today's gates."
    if high + med >= 5:
        return f"{tone} — {high} high and {med} medium-conviction ideas on the board."
    return f"{tone} — thin setup list today ({high+med} qualifying names)."


def _key_insight(
    picks: list[dict[str, Any]],
    sector_map: dict[str, dict[str, float]],
    regime_label: str,
) -> str:
    """Pull the most informative single observation for the day."""
    if not picks:
        return (
            "No names cleared today's 60% backtest win-rate gate plus "
            "conviction threshold. Staying flat is a valid call."
        )
    if not sector_map:
        top = picks[0]
        return (
            f"{top.get('name', top.get('symbol'))} leads the board at "
            f"{top.get('composite_score', 0):.0f}/100 with "
            f"{top.get('components', {}).get('backtest', {}).get('win_rate_pct', 0):.0f}% historical win-rate."
        )
    # Find the hottest sector (highest avg score with ≥ 2 names)
    ranked = sorted(
        [(s, d["avg"], d["count"]) for s, d in sector_map.items() if d["count"] >= 2],
        key=lambda x: x[1], reverse=True,
    )
    if ranked:
        sec, avg, cnt = ranked[0]
        return (
            f"{sec} sector dominates today's leaderboard "
            f"({cnt} names averaging {avg:.0f}/100). "
            "Sector rotation beats single-name bets in this tape."
        )
    top = picks[0]
    return (
        f"{top.get('name', top.get('symbol'))} is the standout at "
        f"{top.get('composite_score', 0):.0f}/100."
    )


def _conviction_board(picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top N picks (already sorted) with compact one-liner."""
    board = []
    for p in picks[:MAX_CONVICTION]:
        bd = p.get("score_breakdown", {})
        board.append({
            "symbol": p.get("symbol"),
            "name": p.get("name"),
            "composite": p.get("composite_score"),
            "tier": p.get("conviction_tier"),
            "winrate_pct": bd.get("winrate_pct"),
            "one_liner": (
                f"Tech {bd.get('technical', 0):.0f}, "
                f"CANSLIM {bd.get('canslim', 0):.0f}, "
                f"News {bd.get('news', 0):.0f}, "
                f"Backtest {bd.get('winrate_pct', 0):.0f}%."
            ),
        })
    return board


def _risk_watchlist(picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Picks with one or more penalty flags tripped."""
    out = []
    for p in picks:
        flags = p.get("components", {}).get("penalties", {}).get("flags") or {}
        notes = p.get("components", {}).get("penalties", {}).get("notes") or []
        if flags:
            out.append({
                "symbol": p.get("symbol"),
                "name": p.get("name"),
                "flag_count": len(flags),
                "notes": notes[:3],
                "penalty": sum(flags.values()),
            })
        if len(out) >= MAX_RISK:
            break
    return out


def _breakout_watch(near_misses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Near-misses that are within striking distance of qualifying."""
    out = []
    for p in near_misses[:MAX_BREAKOUT]:
        bd = p.get("score_breakdown", {})
        out.append({
            "symbol": p.get("symbol"),
            "name": p.get("name"),
            "composite": p.get("composite_score"),
            "missing": (
                "composite ≥60 needed" if p.get("composite_score", 0) < 60
                else "win-rate gate close"
            ),
            "winrate_pct": bd.get("winrate_pct"),
        })
    return out


def _sector_heatmap(all_scored: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Sector → {avg, count} across the full scored universe (non-gated)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in all_scored:
        if r.get("gated"):
            continue
        buckets[_sector_of(r)].append(float(r.get("composite_score", 0)))
    return {
        sec: {
            "avg": round(sum(scores) / len(scores), 2),
            "count": len(scores),
        }
        for sec, scores in buckets.items() if scores
    }


def _news_highlights(all_scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top HIGH-impact headlines across the scored universe."""
    pool: list[dict[str, Any]] = []
    for r in all_scored:
        news = r.get("components", {}).get("news", {})
        for h in (news.get("top_headlines") or []):
            if h.get("tier") == "HIGH":
                pool.append({
                    "symbol": r.get("symbol"),
                    "name": r.get("name"),
                    "title": h.get("title"),
                    "source": h.get("source"),
                    "sentiment": "positive" if h.get("score", 0) > 0 else "negative",
                })
    # Dedup by title
    seen = set()
    unique = []
    for p in pool:
        key = (p.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:MAX_NEWS]


def _action_plan(
    picks: list[dict[str, Any]],
    risk: list[dict[str, Any]],
    regime_label: str,
) -> list[str]:
    """3 numbered actionable items."""
    plan = []
    if picks:
        top = picks[0]
        plan.append(
            f"Review {top.get('symbol')} ({top.get('composite_score', 0):.0f}/100) "
            f"— top conviction idea of the day."
        )
    else:
        plan.append("No qualifying picks today — defer fresh adds; revisit tomorrow's EOD run.")

    if risk:
        names = ", ".join(r.get("symbol", "") for r in risk[:3])
        plan.append(f"Check risk flags on {names} before adding or holding.")
    else:
        plan.append("Penalty flags are clean across today's leaderboard — no immediate risk to address.")

    if regime_label == "BULL":
        plan.append("Bias: lean in on breakouts, trail stops on existing winners.")
    elif regime_label == "BEAR":
        plan.append("Bias: reduce gross exposure, wait for confirmation before new adds.")
    else:
        plan.append("Bias: setup-quality > market thesis; take what the tape gives you.")

    return plan


def build_brief(
    all_scored: list[dict[str, Any]],
    picks: list[dict[str, Any]],
    near_misses: list[dict[str, Any]],
    regime: dict[str, Any],
) -> dict[str, Any]:
    """
    Entry point. Call once per daily run with already-sorted picks/near_misses
    and regime dict (as returned by regime_detector.Regime.to_dict()).
    """
    regime_label = regime.get("label", "SIDEWAYS")
    regime_rationale = regime.get("rationale", "")

    sector_map = _sector_heatmap(all_scored)

    brief = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "regime": {
            "label": regime_label,
            "rationale": regime_rationale,
            "nifty_close": regime.get("nifty_close"),
        },
        "headline": _headline(picks, regime_label, regime_rationale),
        "key_insight": _key_insight(picks, sector_map, regime_label),
        "conviction_board": _conviction_board(picks),
        "risk_watchlist": _risk_watchlist(picks),
        "breakout_watch": _breakout_watch(near_misses),
        "sector_heatmap": sector_map,
        "action_plan": _action_plan(picks, _risk_watchlist(picks), regime_label),
        "news_highlights": _news_highlights(all_scored),
        "counts": {
            "universe_scored": len(all_scored),
            "picks": len(picks),
            "near_misses": len(near_misses),
            "gated_out": sum(1 for r in all_scored if r.get("gated")),
        },
    }
    return brief
