"""
pipeline/universe_builder.py — build today's tradeable universe.

Priority chain (first success wins):
    1. niftyindices.com CSVs (NIFTY 500 + NIFTY SMALLCAP 250)
    2. BSE 500 JSON API
    3. nselib.capital_market  (upstox-maintained fallback library)
    4. Static seed list from config (last resort, always works)

Post-filter:
    - Market cap between [min_crore, max_crore]
    - Debt/Equity ≤ max
    - Trailing-twelve-month net profit > 0 (skipped if flag off)
    - Yahoo symbol resolvable under one of the suffixes in config

Output: DataFrame with columns
    symbol, company, source_index, yahoo_ticker, market_cap_crore,
    debt_to_equity, profit_ttm_crore
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
    safe_float,
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


def _resolve_yahoo_ticker(symbol: str, suffixes: list[str]) -> str | None:
    """
    Try each suffix and return the first one yfinance resolves. We only check
    that .info returns a non-empty dict — a full history fetch would be too
    slow for 750 symbols. Real validation happens downstream.
    """
    import yfinance as yf
    for sfx in suffixes:
        candidate = f"{symbol}{sfx}"
        try:
            tk = yf.Ticker(candidate)
            info = tk.fast_info
            # fast_info raises on unknown tickers and is cheap.
            if info is not None and getattr(info, "last_price", None):
                return candidate
        except Exception:
            continue
    return None


def _fundamentals(ticker: str) -> dict[str, Any]:
    """
    Pull market-cap, debt/equity, TTM net profit via yfinance.
    Best-effort: every field can come back None; callers must handle that.
    """
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        mcap = info.get("marketCap")  # INR
        de = info.get("debtToEquity")  # percentage per yfinance
        # yfinance returns debtToEquity as a PERCENT (e.g. 42.5 means 0.425).
        # Normalize so callers can compare against the 3.0 cap meaningfully.
        de_normalized = safe_float(de) / 100.0 if de is not None else None

        # Earnings — yfinance removed `quarterly_earnings` in 0.2.50.
        # Use quarterly_income_stmt for TTM net income; fall back to info.
        ttm_net = None
        try:
            qi = tk.quarterly_income_stmt
            if qi is not None and not qi.empty and "Net Income" in qi.index:
                # Sum last 4 quarters
                ttm_net = float(qi.loc["Net Income"].iloc[:4].sum())
        except Exception:
            pass
        if ttm_net is None:
            ttm_net = safe_float(info.get("netIncomeToCommon"), default=0.0)

        return {
            "market_cap_crore": safe_float(mcap) / 1e7 if mcap else None,  # 1 cr = 1e7
            "debt_to_equity": de_normalized,
            "profit_ttm_crore": ttm_net / 1e7 if ttm_net else None,
            "sector": info.get("sector") or info.get("industry") or "",
        }
    except Exception as e:
        log.debug("fundamentals failed for %s: %s", ticker, e)
        return {
            "market_cap_crore": None,
            "debt_to_equity": None,
            "profit_ttm_crore": None,
            "sector": "",
        }


def build_universe() -> pd.DataFrame:
    """
    Assemble the full universe DataFrame. Never raises — returns an empty
    DataFrame if every source fails.
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

    # Phase 4: Yahoo resolution + fundamentals + filter
    filters = cfg.get("quality_filters", {}) or {}
    suffixes = filters.get("yahoo_suffixes", [".NS", ".BO"])
    min_mcap = float(filters.get("min_market_cap_crore", 0))
    max_mcap = float(filters.get("max_market_cap_crore", 1e18))
    max_de = float(filters.get("max_debt_to_equity", 1e9))
    need_profit = bool(filters.get("require_positive_net_profit_ttm", False))

    keep: list[dict] = []
    for _, r in df.iterrows():
        sym = r["symbol"]
        yticker = _resolve_yahoo_ticker(sym, suffixes)
        if not yticker:
            continue
        f = _fundamentals(yticker)
        mcap = f.get("market_cap_crore")
        de = f.get("debt_to_equity")
        pft = f.get("profit_ttm_crore")

        # Apply filters permissively — missing data is NOT an auto-reject,
        # because yfinance is notoriously sparse for small caps. Only reject
        # when we have a value AND it's outside the band.
        if mcap is not None and (mcap < min_mcap or mcap > max_mcap):
            continue
        if de is not None and de > max_de:
            continue
        if need_profit and pft is not None and pft <= 0:
            continue

        keep.append({
            "symbol": sym,
            "company": r.get("company", ""),
            "industry": r.get("industry", "") or f.get("sector", ""),
            "source_index": r.get("source_index", ""),
            "yahoo_ticker": yticker,
            "market_cap_crore": mcap,
            "debt_to_equity": de,
            "profit_ttm_crore": pft,
            "sector": f.get("sector", ""),
        })

    out = pd.DataFrame(keep)
    log.info("universe post-filter: %d tickers", len(out))
    return out


if __name__ == "__main__":
    df = build_universe()
    print(df.head(20).to_string())
    print(f"\nTotal: {len(df)} tickers")
