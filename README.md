# ProfitPilot v3

**Rule-based, backtest-gated NSE stock intelligence. No LLMs in the scoring loop.**

Daily pipeline that scores the NIFTY 500 / BSE 500 / SMALLCAP 250 universe across four independent lenses — technical, fundamental (CANSLIM-inspired), news sentiment (FinBERT), and a 3-year walk-forward backtest — then publishes picks to a static GitHub Pages PWA.

- **Live dashboard:** https://prashobhpaul.github.io/ProfitPilot/
- **Pipeline runs:** 4× daily via GitHub Actions (pre-market, mid-day, EOD, post-close)
- **Output:** static `predictions.json` + `news_cache.json` consumed by an offline-capable PWA

---

## Why v3 exists

v2 scored stocks and published picks. It had two structural problems:

1. **No validation.** A high technical score didn't mean the signal had ever worked on that stock. Every name was treated as equally responsive to the same rules.
2. **Optional LLM drift.** The "AI Brief" had a Gemini path that was removed after it produced inconsistent outputs. v3 enforces that the scoring + brief layer is 100% deterministic.

v3 addresses both:

- **60% backtest gate.** Every picked stock must have achieved its forward-return targets ≥60% of the time across the last 3 years of its own history. Stocks whose signals never worked get filtered out regardless of how good today's chart looks.
- **FinBERT for news only.** The one ML component is sentiment classification of news headlines — a well-scoped, reproducible classification task. It does not influence rankings beyond a weighted sub-score, and if transformers/torch fail to load, the pipeline degrades to rule-based keyword sentiment without crashing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ GitHub Actions (daily.yml)                                       │
│   Schedules: 03:30 UTC (pre-mkt), 07:00, 10:30 (EOD), 13:00      │
│   Caches: .cache/huggingface (FinBERT 440MB), .cache/ohlcv       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  python -m api.daily
┌─────────────────────────────────────────────────────────────────┐
│ api/daily.py  (orchestrator, preserve-last-good on failure)      │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─[1]─► universe_builder.py      → ~500 symbols
         │       (niftyindices CSV ▸ nselib ▸ static seed fallback)
         │
         ├─[2]─► regime_detector.py       → BULL / BEAR / SIDEWAYS
         │       (NIFTY50 vs 50/200 DMA + 3M return)
         │
         ├─[3]─► utils.fetch_ohlcv        → 3y daily OHLCV (parquet-cached)
         │       (curl_cffi impersonate="chrome" for TLS)
         │
         ├─[4]─► backtester.py            → 3y walk-forward, per-horizon win-rate
         │       (ThreadPoolExecutor×4, monthly rebalance, 10d/30d/60d hits)
         │
         ├─[5]─► per-ticker scoring (4 parallel):
         │       ├─ technical_scorer.py   → 0-100 (trend/momentum/vol/breakout/action)
         │       ├─ oneil_scorer.py       → 0-100 (EPS growth, RS, ROE, inst %, 52w)
         │       ├─ risk_penalties.py     → 0 to -25 (RSI>78, <SMA200, ATR, earnings)
         │       └─ finbert_news.py       → 0-100 (ProsusAI/finbert CPU + impact tier)
         │
         ├─[6]─► final_scorer.py          → composite + 60% win-rate gate
         │       (regime-weighted: BULL 40/30/20/10, BEAR 25/35/25/15, SIDEWAYS 35/30/25/10)
         │
         ├─[7]─► narrative_v3.py          → 4-6 sentence rationale per pick
         │
         ├─[8]─► daily_brief.py           → EOD brief (deterministic, no LLM)
         │
         └─[9]─► atomic write              → predictions.json + news_cache.json
                                            (temp file + os.replace; readers never see half-files)
```

### Module inventory

| File | Role | Degradation |
|------|------|-------------|
| `pipeline/utils.py` | logging, HTTP session (Chrome TLS), OHLCV parquet cache, atomic JSON writes | — |
| `pipeline/universe_builder.py` | NSE/BSE universe construction | CSV ▸ JSON ▸ nselib ▸ static seed |
| `pipeline/regime_detector.py` | Market regime classification | returns SIDEWAYS on any failure |
| `pipeline/technical_scorer.py` | 5-component rule-based technicals | returns 50 neutral on insufficient history |
| `pipeline/oneil_scorer.py` | CANSLIM-inspired fundamentals | neutral per sub-component on missing data |
| `pipeline/risk_penalties.py` | Red-flag deductions | zero penalty if data missing |
| `pipeline/finbert_news.py` | FinBERT + impact-tier + recency decay | rule-based keyword fallback if transformers unavailable |
| `pipeline/backtester.py` | 3y walk-forward, multi-horizon | 0% win-rate (auto-gated out) on failure |
| `pipeline/final_scorer.py` | Regime-weighted composite + gating | — |
| `pipeline/narrative_v3.py` | Deterministic templated narrative | fallback one-liner on exception |
| `pipeline/daily_brief.py` | EOD brief structured dict | returns minimal brief on empty picks |
| `api/daily.py` | Orchestrator | preserve-last-good with `_stale=true` |

### Frontend

Single-file PWA (`index.html`) consuming `predictions.json`. Obsidian dark aesthetic with violet accents. Five tabs: Picks, Near-misses, Risk watch, Sectors, News.

**v3-specific UI additions** (per spec): three chips per pick card —

- **Win-rate** — 3-year backtest hit-rate (green)
- **CANSLIM** — fundamental sub-score (violet)
- **News** — FinBERT sentiment score (green/yellow/red based on polarity)

Plus a Tech chip and a conditional Risk chip if any penalty flags tripped.

Service worker strategy is **network-first for JSON, cache-first for shell** — the v2 bug where stale `predictions.json` caused a splash-screen hang is structurally impossible in v3.

---

## Repository layout

```
.
├── .github/workflows/daily.yml          # 4× daily cron + caches + retry
├── api/
│   ├── __init__.py
│   └── daily.py                         # `python -m api.daily` entry point
├── pipeline/
│   ├── __init__.py
│   ├── utils.py
│   ├── universe_builder.py
│   ├── regime_detector.py
│   ├── technical_scorer.py
│   ├── oneil_scorer.py
│   ├── risk_penalties.py
│   ├── finbert_news.py
│   ├── backtester.py
│   ├── final_scorer.py
│   ├── narrative_v3.py
│   └── daily_brief.py
├── config/
│   └── universe_sources.yml             # index CSVs + nselib fallback + static seed
├── index.html                           # single-file PWA
├── manifest.json
├── sw.js
├── requirements.txt
├── predictions.json                     # generated (gitignored or committed per preference)
├── news_cache.json                      # generated
└── .cache/                              # gitignored; persisted by actions/cache@v4
    ├── huggingface/                     # ~440MB FinBERT weights
    └── ohlcv/                           # parquet, keyed on today's date
```

---

## Local development

```bash
# Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline locally
python -m api.daily

# Expected output: predictions.json and news_cache.json in repo root
# First run downloads FinBERT (~440MB) into .cache/huggingface

# Run a single stage
python -m pipeline.regime_detector
python -m pipeline.finbert_news           # smoke-tests RELIANCE news

# Serve the frontend
python -m http.server 8000
# Open http://localhost:8000
```

### Environment tuning

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_HOME` | `.cache/huggingface` | FinBERT model cache location |
| `MAX_UNIVERSE_SIZE` | 500 | Cap in `api/daily.py` — lower for testing |
| `SCORING_WORKERS` | 4 | Per-ticker scoring parallelism |

---

## Deployment (8 steps)

Assuming your GitHub Pages repo is already set up at `prashobhpaul/ProfitPilot`:

1. **Backup the old branch**
   ```bash
   git checkout main
   git branch v2-archive
   git push origin v2-archive
   ```

2. **Remove legacy files** (v2 scripts that v3 replaces)
   ```bash
   git rm api/quotes.py api/quotes_daily.py \
          analyze.py advisor_narrator.py narrative_engine.py \
          forecast_engine.py news_analyzer.py news_fetcher.py \
          data_fetch.py gpt_analyze.py \
          patch_frontend_v2.py patch_orphaned_interval.py \
          index-emergency.html offline.html
   ```

3. **Add v3 files** — copy the 21 files from `profitpilot-v3/` into the repo root, preserving directory structure (`pipeline/`, `api/`, `config/`, `.github/workflows/`).

4. **Commit and push**
   ```bash
   git add -A
   git commit -m "v3: FinBERT + CANSLIM + backtest-gated pipeline"
   git push origin main
   ```

5. **Trigger the workflow manually** to seed `predictions.json`
   - Go to **Actions → Daily ProfitPilot → Run workflow**
   - First run takes ~15 min (FinBERT download + 3y OHLCV fetch for 500 tickers)
   - Subsequent runs are ~6-8 min thanks to `actions/cache@v4`

6. **Verify GitHub Pages**
   - Settings → Pages → Source: `main` branch, root (`/`)
   - Visit https://prashobhpaul.github.io/ProfitPilot/
   - Check that the regime chip loads, brief populates, picks render with the three new chips

7. **Monitor the first scheduled run**
   - Check the step summary in Actions for: universe size, OHLCV fetch rate, backtest passing count, scoring progress, total duration
   - If `_stale: true` appears in `predictions.json`, read `_stale_reason` to diagnose

8. **Optional hardening** for chronic Yahoo rate-limiting
   - If OHLCV fetch fails frequently, lower `MAX_UNIVERSE_SIZE` to 300 in `api/daily.py`
   - Add a secondary cron slot offset by 30 min
   - Consider enabling `curl_cffi` proxy rotation in `pipeline/utils.chrome_session`

---

## Scoring weights reference

Composite = `w_tech·tech + w_canslim·canslim + w_news·news + w_winrate·winrate + penalties`

| Regime | Tech | CANSLIM | News | Backtest |
|--------|-----:|--------:|-----:|---------:|
| BULL | 0.40 | 0.30 | 0.20 | 0.10 |
| SIDEWAYS | 0.35 | 0.30 | 0.25 | 0.10 |
| BEAR | 0.25 | 0.35 | 0.25 | 0.15 |

Penalties are added post-weighting (not weighted themselves) and capped at −25. Gate: win-rate < 60% → `gated: true`, excluded from picks.

Conviction tiers:
- **HIGH** — composite ≥75 AND win-rate ≥70
- **MEDIUM** — composite ≥60 AND win-rate ≥60
- **LOW** — anything else

---

## Backtest methodology

For each ticker independently:

1. Walk through the last 3 years at ~monthly stride (21 trading days).
2. At each rebalance point, compute a simplified technical score using only data available up to that bar (no look-ahead).
3. If score ≥70, treat it as a hypothetical entry.
4. Check forward rolling-max against targets: **10d: 5%**, **30d: 10%**, **60d: 15%**.
5. Aggregate hits across all rebalance points → per-horizon win-rate.
6. Composite = weighted average (40/35/25 across 10d/30d/60d).

**Why simplified scoring during backtest?** We can't replay FinBERT historically (no archived news) and can't retrieve point-in-time fundamentals from yfinance. So the backtest tests whether systematic technical ranking has ever worked on this specific stock. A stock whose chart never respects breakouts gets gated out regardless of today's FinBERT or CANSLIM score — which is the point.

No slippage or commission modeling. The 60% gate provides ~15% buffer for real-world friction; we only need relative ranking here.

---

## Degradation behavior

Every stage is designed to fail safely:

- **Empty universe** → preserve last-good, mark `_stale=true`
- **>50% OHLCV fetch failures** → preserve last-good
- **>50% scoring failures** → preserve last-good
- **FinBERT import fails** → rule-based keyword sentiment, flagged `method: "rule_fallback"`
- **Google News RSS fails for a ticker** → neutral 50 news score, `method: "no_news"`
- **yfinance `.info` 404s** → neutral fundamental sub-scores (not zero)
- **Earnings calendar unavailable** → skip blackout penalty (no false negatives)
- **Catastrophic unhandled exception** → preserve last-good with exception name as `_stale_reason`

The frontend renders a `⚠ Stale data` badge whenever `_stale=true`.

---

## Disclaimer

**This is an educational tool, not investment advice.**

- All scores are deterministic rule-based outputs. There is no recommendation engine, no personalized advice, no portfolio management.
- Past backtest performance does not guarantee future returns. The 60% historical win-rate gate is a filter, not a prediction.
- The FinBERT sentiment layer classifies headline text. It does not know if the headline is factually correct, whether the event is priced in, or how the broader market will react.
- Market data may be delayed, incomplete, or incorrect. Always verify critical data from primary sources (exchange filings, company announcements) before making any decision.
- The author is not a registered investment advisor. Use at your own risk. Do your own research.

---

## License & credits

- **FinBERT:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (Apache 2.0) — pre-trained financial sentiment classifier.
- **yfinance, pandas, numpy, curl_cffi, transformers, torch** — each under their respective licenses.
- **Market data:** Yahoo Finance (via yfinance), niftyindices.com CSVs, nselib — subject to upstream terms.
- **News:** Google News RSS — headlines only; linked back to original sources.

Pipeline code: MIT.
