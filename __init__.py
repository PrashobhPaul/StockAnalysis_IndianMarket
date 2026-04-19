"""
ProfitPilot v3 — Pipeline package.

Flat 7-stage rule-driven pipeline replacing the legacy scattered scripts:

    universe_builder   →   OHLCV fetch (utils)     →   technical_scorer
         ↓                                                    ↓
    regime_detector ─────────────────────────────────→   oneil_scorer
         ↓                                                    ↓
    finbert_news   ───────────────────────────────────→   risk_penalties
         ↓                                                    ↓
                       backtester  ────→   final_scorer  ────→  narrative_v3
                                                                     ↓
                                                               daily_brief

Every stage degrades gracefully: on exception it returns a safe default
(empty DataFrame, zero score, neutral label) and logs a warning. The
pipeline never crashes the workflow — a partial result is always better
than a 24h-stale app.

Import-time side effects are forbidden. All heavy work (HTTP, torch load,
parquet IO) happens inside explicit functions that the orchestrator
(api/daily.py) calls in order.
"""

__version__ = "3.0.0"
