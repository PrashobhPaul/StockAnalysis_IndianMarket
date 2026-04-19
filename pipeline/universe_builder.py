"""
pipeline/universe_builder.py — build today's tradeable universe.

Priority chain (first success wins):
    1. niftyindices.com CSVs (NIFTY 500 + NIFTY SMALLCAP 250)
    2. BSE 500 JSON API
    3. nselib.capital_market  (upstox-maintained fallback library)
    4. Static seed list from config (last resort, always works)

Post-processing:
    - Deduplicate by symbol
    - Assign default Yahoo suffix (.NS); dead tickers get filtered out
      naturally by the downstream OHLCV fetch stage
    - NO per-ticker fundamentals/market-cap calls — oneil_scorer does
      this work anyway with its own yfinance.Ticker, parallelized across
      4 workers; doing it twice just burned the Yahoo rate limit and
      pushed total runtime past the 40-min budget.

Output: DataFrame with columns
    symbol, company, industry, source_index, yahoo_ticker, sector

FIXES (2026-04 debug pass):
  - UNI-F1: Removed serial per-ticker yfinance `.info` + `_resolve_yahoo_ticker`
    calls from Phase 4. Was taking 15-30 min for 1250 rows, frequently
    tripping the workflow timeout or exhausting Yahoo rate limits before
    the OHLCV fetch could even start. Dead tickers are now filtered
    downstream where the batch-download mechanism already handles them.
  - UNI-F2: Still populates `market_cap_crore`, `debt_to_equity`,
    `profit_ttm_crore` as None so the DataFrame schema stays stable
    for any downstream consumers that expect those columns.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.utils import (
    chrome_session,
    get_logger,
    http_get,
)

log = get_logger("universe_builder")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "universe_sources.yml"


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.error("could not load %s: %s", CONFIG_PATH, e)
        return {}


def _fetch_csv_source(url: str, session) -> list[dict]:
    """Parse a niftyindices CSV. Returns list of {symbol, company, industry}."""
    body = http_get(url, session=session, retries=3, timeout=15.0)
    if not body:
        return []
    try:
        df = pd.read_csv(io.BytesIO(body))
        # niftyindices columns: "Company Name", "Industry", "Symbol", "Series", "ISIN Code"
        col_map: dict[str, str] = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ("symbol", "ticker"):
                col_map[c] = "symbol"
            elif lc in ("company name", "company"):
                col_map[c] = "company"
            elif lc == "industry":
                col_map[c] = "industry"
        df = df.rename(columns=col_map)
        if "symbol" not in df.columns:
            return []
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["company"] = df.get("company", "").astype(str).str.strip()
        df["industry"] = df.get("industry", "").astype(str).str.strip()
        return df[["symbol", "company", "industry"]].to_dict(orient="records")
    except Exception as e:
        log.warning("could not parse CSV from %s: %s", url, e)
        return []


def _fetch_bse_json(url: str, session) -> list[dict]:
    """BSE India's JSON API — different shape. Mapped into our CSV record shape."""
    body = http_get(url, session=session, retries=3, timeout=15.0)
    if not body:
        return []
    try:
        import json
        data = json.loads(body.decode("utf-8", errors="ignore"))
        rows = data.get("Table") or data.get("data") or data
        if not isinstance(rows, list):
            return []
        out = []
        for r in rows:
            sym = r.get("scrip_cd") or r.get("SYMBOL") or r.get("Symbol")
            name = r.get("ss_name") or r.get("company") or r.get("Company")
            if sym and name:
                out.append({
                    "symbol": str(sym).strip().upper(),
                    "company": str(name).strip(),
                    "industry": r.get("industry") or "",
                })
        return out
    except Exception as e:
        log.warning("BSE JSON parse failed: %s", e)
        return []


def _nselib_fallback() -> list[dict]:
    """Pull the full NSE equity list via nselib, used when CSVs fail."""
    try:
        from nselib import capital_market
        df = capital_market.nse_equity_list()
        if df is None or df.empty:
            return []
        col_map: dict[str, str] = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc == "symbol":
                col_map[c] = "symbol"
            elif "name" in lc:
                col_map[c] = "company"
        df = df.rename(columns=col_map)
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["company"] = df.get("company", "").astype(str).str.strip()
        df["industry"] = ""
        log.info("nselib fallback returned %d rows", len(df))
        return df[["symbol", "company", "industry"]].to_dict(orient="records")
    except Exception as e:
        log.warning("nselib fallback failed: %s", e)
        return []


def build_universe() -> pd.DataFrame:
    """
    Assemble the full universe DataFrame. Never raises — returns an empty
    DataFrame if every source fails.

    UNI-F1: Phase 4 is now a fast pure-Python pass (no network I/O). Dead
    tickers get filtered naturally by the downstream OHLCV batch fetch
    (which is already parallelized + parquet-cached). Missing fundamentals
    are recomputed by oneil_scorer.
    """
    cfg = _load_config()
    session = chrome_session()
    rows: list[dict] = []

    # Phase 1: CSV sources
    for src in cfg.get("csv_sources", []) or []:
        index_name = src.get("index", "?")
        url = src.get("url", "")
        fmt = src.get("format", "csv")
        if not url:
            continue
        log.info("fetching %s from %s", index_name, url)
        if fmt == "bse_json":
            got = _fetch_bse_json(url, session)
        else:
            got = _fetch_csv_source(url, session)
        for r in got:
            r["source_index"] = index_name
        rows.extend(got)
        log.info("  → got %d rows from %s", len(got), index_name)

    # Phase 2: nselib fallback if phase 1 was thin
    if len(rows) < 100 and cfg.get("nselib_fallback", {}).get("enabled"):
        log.info("CSV sources thin (%d rows), trying nselib fallback", len(rows))
        for r in _nselib_fallback():
            r["source_index"] = "nselib"
            rows.append(r)

    # Phase 3: static seed
    if len(rows) < 50:
        seed = cfg.get("static_seed", []) or []
        log.warning("falling back to static seed (%d symbols)", len(seed))
        for sym in seed:
            rows.append({
                "symbol": sym.strip().upper(),
                "company": "",
                "industry": "",
                "source_index": "static_seed",
            })

    if not rows:
        log.error("universe builder: no rows from any source")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    log.info("universe pre-filter: %d unique symbols", len(df))

    # Phase 4 (UNI-F1): fast local expansion. No per-ticker network calls.
    # Dead tickers are weeded out naturally by the OHLCV batch fetch later.
    cfg_filters = cfg.get("quality_filters", {}) or {}
    suffixes = cfg_filters.get("yahoo_suffixes", [".NS", ".BO"])
    default_suffix = suffixes[0] if suffixes else ".NS"

    keep: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).strip().upper()
        if not sym or not sym.replace("&", "").replace("-", "").replace(".", "").isalnum():
            # Skip obviously malformed symbols (keep '&', '-', '.' which appear
            # in legitimate NSE tickers like 'M&M', 'BAJAJ-AUTO', 'MRF.NS')
            continue
        keep.append({
            "symbol": sym,
            "company": (r.get("company") or "").strip(),
            "industry": (r.get("industry") or "").strip(),
            "source_index": r.get("source_index", ""),
            "yahoo_ticker": f"{sym}{default_suffix}",
            # Schema-stable columns — populated by oneil_scorer downstream
            "market_cap_crore": None,
            "debt_to_equity": None,
            "profit_ttm_crore": None,
            "sector": (r.get("industry") or "").strip(),
        })

    out = pd.DataFrame(keep)
    log.info("universe post-filter: %d tickers", len(out))
    return out


if __name__ == "__main__":
    df = build_universe()
    print(df.head(20).to_string())
    print(f"\nTotal: {len(df)} tickers")
