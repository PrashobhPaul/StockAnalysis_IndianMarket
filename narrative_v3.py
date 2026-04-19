"""
pipeline/narrative_v3.py — per-stock 4-6 sentence rationale.

Replaces v2's narrative_engine.py. Still deterministic (no LLM), but now
integrates 4 scoring lenses instead of 1. The narrative is the user-facing
explanation of WHY a stock made the list.

Structure: 4-6 sentences, each drawn from a different signal source:
    1. Technical summary      (from tech.breakdown)
    2. Fundamental summary    (from canslim.metrics)
    3. News context           (from news.top_headlines)
    4. Risk / caveat          (from penalties.notes)
    5. Historical reliability (from backtest.win_rate_pct)
    [6. Regime-specific tone  (from regime.label)]

Deterministic templating with numeric placeholders so identical inputs
produce identical narratives (easier for users to trust, easier to test).
"""
from __future__ import annotations

import random
from typing import Any

from pipeline.utils import get_logger

log = get_logger("narrative")


# ───────────────────────── Template pools ─────────────────────────
TECH_TEMPLATES_STRONG = [
    "{name} is in a clear uptrend — trading above both its 50- and 200-day moving averages with RSI in the {rsi_bucket} zone and volume running {vol_ratio}× its 20-day average.",
    "Price action is decisively bullish: {name} just {breakout_phrase} on volume {vol_ratio}× its 20-day average, with momentum indicators confirming the move.",
    "Technically, {name} is showing textbook accumulation — trending above key moving averages, expanding volume, and RSI at {rsi_value:.0f} in the sweet spot.",
]
TECH_TEMPLATES_MIXED = [
    "Technicals are constructive but not extended — {name} is holding above its 50-day average with RSI at {rsi_value:.0f}, leaving room for further upside.",
    "The chart shows a developing setup rather than a breakout — {name} is building a base above support with moderate volume.",
]

CANSLIM_TEMPLATES = [
    "Fundamentally, the name checks several CANSLIM boxes: quarterly EPS grew {eps_qoq:+.0f}% YoY, 3Y EPS CAGR is {eps_cagr:+.0f}%, and relative strength vs NIFTY is {rel_str:+.0f}%.",
    "The business is compounding — EPS growth of {eps_qoq:+.0f}% last quarter and {eps_cagr:+.0f}% CAGR over three years, with ROE of {roe_pct:.0f}%.",
    "Earnings quality backs the chart: {eps_qoq:+.0f}% YoY quarterly EPS growth, {rel_str:+.0f}% outperformance vs NIFTY, and healthy institutional ownership.",
]
CANSLIM_WEAK_TEMPLATES = [
    "Fundamentals are mixed — recent EPS growth of {eps_qoq:+.0f}% is less robust than the technical setup suggests, so position sizing matters.",
    "Earnings momentum is modest ({eps_qoq:+.0f}% QoQ) relative to the price action; this is more of a technical play than a growth-story conviction.",
]

NEWS_TEMPLATES_POS = [
    "News flow is supportive — {headline_count} recent headlines skew positive, including coverage of {top_theme}.",
    "Sentiment is tilting bullish across {headline_count} recent items, with the standout being {top_theme}.",
]
NEWS_TEMPLATES_NEU = [
    "News coverage ({headline_count} items) is largely neutral — no clear positive or negative catalyst in the last week.",
    "The newsflow is quiet — {headline_count} routine items, nothing moving the narrative either way.",
]
NEWS_TEMPLATES_NEG = [
    "News flow is a caution flag — {headline_count} recent headlines lean negative, notably around {top_theme}.",
    "Sentiment is mixed-to-weak with {headline_count} items in the last week; the dominant theme is {top_theme}.",
]

RISK_TEMPLATES = [
    "Risks to watch: {risk_notes}.",
    "Caveats include {risk_notes} — size accordingly.",
    "On the risk side: {risk_notes}.",
]
RISK_CLEAN = [
    "No significant risk flags tripped at current levels.",
    "Risk checks are clean — no overbought, blackout, or volatility flags.",
]

BACKTEST_STRONG = [
    "Historically, this pattern has worked on {name} — the setup achieved its targets {winrate:.0f}% of the time across the last 3 years.",
    "Backtesting gives us conviction: signals like today's scored through to target in {winrate:.0f}% of historical instances.",
]
BACKTEST_MEDIUM = [
    "The 3-year backtest shows a {winrate:.0f}% hit rate on similar setups — above our gate but not a sure thing.",
]
BACKTEST_WEAK = [
    "The 3-year backtest win-rate of {winrate:.0f}% is below our conviction threshold, which is why this ranks as {tier} and not higher.",
]

REGIME_ADDENDUM = {
    "BULL": "Broad market remains supportive, reinforcing the case.",
    "BEAR": "Given the weak market backdrop, waiting for confirmation or smaller sizing is prudent.",
    "SIDEWAYS": "In a range-bound tape, individual setup quality matters more than the macro.",
}


def _rsi_bucket(rsi: float) -> str:
    if 55 <= rsi <= 70:
        return "strong but not overbought"
    if rsi > 70:
        return "extended"
    if rsi < 40:
        return "oversold"
    return "neutral"


def _breakout_phrase(prox_52w: float) -> str:
    if prox_52w >= 9.5:
        return "broke to new 52-week highs"
    if prox_52w >= 8:
        return "pushed within 3% of 52-week highs"
    if prox_52w >= 5:
        return "reclaimed multi-month resistance"
    return "is working off its base"


def _top_theme(news_component: dict) -> str:
    """Extract the dominant theme from top headlines. Falls back to generic."""
    top = (news_component or {}).get("top_headlines", [])
    if not top:
        return "routine corporate updates"
    # Use the highest-impact headline's title as theme phrase
    headline = top[0].get("title", "")
    tier = top[0].get("tier", "GEN")
    if not headline:
        return "routine corporate updates"
    # Lightly normalize
    if len(headline) > 85:
        headline = headline[:82].rsplit(" ", 1)[0] + "…"
    return f"\u201C{headline}\u201D" + (" (high-impact)" if tier == "HIGH" else "")


def build_narrative(record: dict[str, Any], regime_label: str, seed: int = 0) -> str:
    """
    Construct the 4-6 sentence narrative for one scored ticker.

    `record` is the output of final_scorer.score_ticker_final.
    `seed` lets callers pin randomness per-ticker (e.g. hash the symbol).
    """
    rnd = random.Random(seed)
    name = record.get("name") or record.get("symbol", "This stock")
    tier = record.get("conviction_tier", "MEDIUM")
    comps = record.get("components", {})

    tech = comps.get("technical", {})
    tech_br = tech.get("breakdown", {})
    tech_ind = tech.get("indicators", {})
    rsi = float(tech_ind.get("rsi_14", 50))
    vol_ratio = 1.0  # v3 backend doesn't expose exact ratio; default 1.0
    # Reasonable proxy: volume score >=14 implies ~2x
    vol_score = float(tech_br.get("volume", 10))
    if vol_score >= 14:
        vol_ratio = 2.0
    elif vol_score >= 10:
        vol_ratio = 1.4

    canslim = comps.get("canslim", {})
    cs_metrics = canslim.get("metrics", {})
    eps_qoq = float(cs_metrics.get("eps_growth_qoq_pct", 0))
    eps_cagr = float(cs_metrics.get("eps_cagr_3y_pct", 0))
    rel_str = float(cs_metrics.get("return_vs_nifty_pct", 0))
    roe_raw = cs_metrics.get("roe") or 0
    roe_pct = roe_raw * 100 if abs(roe_raw) <= 1 else roe_raw

    news = comps.get("news", {})
    news_score = float(news.get("score_0_100", 50))
    n_headlines = int(news.get("headline_count", 0))

    penalty = comps.get("penalties", {})
    risk_notes = penalty.get("notes", [])

    backtest = comps.get("backtest", {})
    winrate = float(backtest.get("win_rate_pct", 0))

    sentences: list[str] = []

    # 1. Technical
    tech_score = float(tech.get("score", 50))
    tpl_pool = TECH_TEMPLATES_STRONG if tech_score >= 75 else TECH_TEMPLATES_MIXED
    sentences.append(rnd.choice(tpl_pool).format(
        name=name,
        rsi_bucket=_rsi_bucket(rsi),
        rsi_value=rsi,
        vol_ratio=vol_ratio,
        breakout_phrase=_breakout_phrase(float(tech_br.get("breakout", 0))),
    ))

    # 2. Fundamental
    cs_score = float(canslim.get("score", 50))
    cs_pool = CANSLIM_TEMPLATES if cs_score >= 55 else CANSLIM_WEAK_TEMPLATES
    sentences.append(rnd.choice(cs_pool).format(
        eps_qoq=eps_qoq, eps_cagr=eps_cagr, rel_str=rel_str, roe_pct=roe_pct,
    ))

    # 3. News
    if n_headlines == 0:
        pool = NEWS_TEMPLATES_NEU
    elif news_score >= 60:
        pool = NEWS_TEMPLATES_POS
    elif news_score <= 40:
        pool = NEWS_TEMPLATES_NEG
    else:
        pool = NEWS_TEMPLATES_NEU
    sentences.append(rnd.choice(pool).format(
        headline_count=n_headlines,
        top_theme=_top_theme(news),
    ))

    # 4. Risk
    if risk_notes:
        sentences.append(rnd.choice(RISK_TEMPLATES).format(
            risk_notes="; ".join(risk_notes[:3]).lower(),
        ))
    else:
        sentences.append(rnd.choice(RISK_CLEAN))

    # 5. Backtest
    if winrate >= 75:
        bt_pool = BACKTEST_STRONG
    elif winrate >= 60:
        bt_pool = BACKTEST_MEDIUM
    else:
        bt_pool = BACKTEST_WEAK
    sentences.append(rnd.choice(bt_pool).format(
        name=name, winrate=winrate, tier=tier,
    ))

    # 6. Regime addendum (optional — only for non-HIGH so the narrative stays tight)
    if tier != "HIGH" and regime_label in REGIME_ADDENDUM:
        sentences.append(REGIME_ADDENDUM[regime_label])

    return " ".join(sentences)


def attach_narratives(picks: list[dict[str, Any]], regime_label: str) -> list[dict[str, Any]]:
    """Mutate picks in place adding `narrative`. Returns same list for chaining."""
    for rec in picks:
        try:
            seed = hash(rec.get("symbol", "")) & 0xFFFFFFFF
            rec["narrative"] = build_narrative(rec, regime_label, seed=seed)
        except Exception as e:
            log.warning("narrative build failed for %s: %s", rec.get("symbol"), e)
            rec["narrative"] = (
                f"{rec.get('name', rec.get('symbol', 'Stock'))} scored "
                f"{rec.get('composite_score', 0):.0f}/100 with "
                f"{rec.get('components', {}).get('backtest', {}).get('win_rate_pct', 0):.0f}% "
                "historical win-rate."
            )
    return picks
