"""
pipeline/utils.py — shared plumbing: HTTP, caching, logging, OHLCV fetch.

Nothing in here is domain logic. Domain logic lives in the dedicated
scorer/regime/backtester modules. Utils exist so those modules don't each
re-invent a requests session or a parquet cache.

Key primitives:
    get_logger(name)         → stderr logger with ISO timestamps
    chrome_session()         → curl_cffi Session with Chrome TLS fingerprint
    http_get(...)            → retrying GET with bounded timeout
    ohlcv_parquet_path(date) → parquet cache keyed on today's UTC date
    fetch_ohlcv(tickers,...) → batched + per-ticker retry + parquet cache
    atomic_write_json(...)   → write-then-rename so readers never see half-files
    safe_float / safe_int    → coerce yfinance junk into clean numbers
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
OHLCV_CACHE = CACHE_DIR / "ohlcv"
OHLCV_CACHE.mkdir(parents=True, exist_ok=True)


# ───────────────────────── Logging ─────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a singleton stderr logger. Idempotent across calls."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(h)
    log.propagate = False
    return log


log = get_logger("utils")


# ───────────────────────── HTTP ─────────────────────────
def chrome_session():
    """
    curl_cffi Session with Chrome TLS fingerprint + plausible headers.
    Returns None if curl_cffi is not importable — callers must handle that.

    Why: NSE/BSE and Yahoo both block requests.Session() because its TLS
    fingerprint (ciphers, extensions, ALPN order) is trivially identifiable
    as Python. curl_cffi loads Chrome's BoringSSL fingerprint and replays it.
    """
    try:
        from curl_cffi import requests as curl_requests
        s = curl_requests.Session(impersonate="chrome")
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return s
    except Exception as e:
        log.warning("curl_cffi unavailable (%s); falling back to plain requests", e)
        return None


def http_get(
    url: str,
    *,
    session=None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> Optional[bytes]:
    """
    Bounded-retry GET. Returns response bytes or None on total failure.

    Tries the provided session first, then falls back to a fresh
    chrome_session, then to plain requests.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if session is not None:
                r = session.get(url, timeout=timeout)
            else:
                import requests
                r = requests.get(url, timeout=timeout, headers={
                    "User-Agent": "Mozilla/5.0 Chrome/125.0.0.0",
                })
            if r.status_code == 200 and r.content:
                return r.content
            last_err = f"status={r.status_code} len={len(r.content or b'')}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(backoff * attempt)
    log.warning("http_get failed url=%s err=%s", url, last_err)
    return None


# ───────────────────────── Numeric coercion ─────────────────────────
def safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce anything to float, swallowing NaN/None/strings."""
    try:
        if x is None:
            return default
        f = float(x)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        i = int(float(x))
        return i
    except (TypeError, ValueError):
        return default


# ───────────────────────── OHLCV fetch + parquet cache ─────────────────────────
def ohlcv_parquet_path(today: Optional[date] = None) -> Path:
    """Cache key: one parquet per trading day. Older files stay for back-test."""
    d = today or datetime.now(IST).date()
    return OHLCV_CACHE / f"ohlcv_{d.isoformat()}.parquet"


def _fetch_batch(tickers: list[str], period: str, interval: str, session) -> pd.DataFrame:
    """yfinance batch download. Returns a MultiIndex DataFrame or empty."""
    import yfinance as yf
    try:
        kw: dict[str, Any] = dict(
            tickers=tickers, period=period, interval=interval,
            progress=False, auto_adjust=True, group_by="ticker",
            threads=True,
        )
        if session is not None:
            kw["session"] = session
        df = yf.download(**kw)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        log.warning("yfinance batch failed: %s", e)
        return pd.DataFrame()


def _fetch_single(ticker: str, period: str, interval: str, session) -> Optional[pd.DataFrame]:
    """Per-ticker retry path for tickers missing from batch results."""
    import yfinance as yf
    try:
        if session is not None:
            tk = yf.Ticker(ticker, session=session)
        else:
            tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty and "Close" in df.columns:
            return df.dropna(subset=["Close"])
    except Exception as e:
        log.debug("single fetch %s failed: %s", ticker, e)
    return None


def fetch_ohlcv(
    tickers: Iterable[str],
    period: str = "3y",
    interval: str = "1d",
    *,
    use_cache: bool = True,
    today: Optional[date] = None,
) -> dict[str, pd.DataFrame]:
    """
    Resilient OHLCV fetch. Returns {ticker: DataFrame}.

    Strategy:
      1. If parquet cache for today exists, load it.
      2. Otherwise batch-download via curl_cffi session.
      3. Per-ticker retry for anything missing after batch (max 2 retries).
      4. Write parquet for next run.

    Never raises. Tickers that fail in every phase are simply absent from
    the result dict — callers must handle missing keys.
    """
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    if not tickers:
        return {}

    cache_path = ohlcv_parquet_path(today)
    if use_cache and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty and isinstance(df.columns, pd.MultiIndex):
                have = set(df.columns.get_level_values(0))
                missing = [t for t in tickers if t not in have]
                if not missing:
                    log.info("ohlcv cache hit: %s (%d tickers)", cache_path.name, len(have))
                    return {t: df[t].dropna(subset=["Close"]) for t in tickers if t in have}
                log.info("ohlcv cache partial: %d/%d tickers, refetching missing",
                         len(have & set(tickers)), len(tickers))
        except Exception as e:
            log.warning("could not read ohlcv cache %s: %s", cache_path, e)

    session = chrome_session()
    got: dict[str, pd.DataFrame] = {}

    # Phase A: batch
    batch = _fetch_batch(tickers, period, interval, session)
    if not batch.empty and isinstance(batch.columns, pd.MultiIndex):
        for t in tickers:
            if t in batch.columns.get_level_values(0):
                sub = batch[t].dropna(subset=["Close"])
                if not sub.empty:
                    got[t] = sub
    log.info("ohlcv batch: %d/%d tickers usable", len(got), len(tickers))

    # Phase B: per-ticker retry
    missing = [t for t in tickers if t not in got]
    if missing:
        log.info("retrying %d missing tickers individually…", len(missing))
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_fetch_single, t, period, interval, session): t for t in missing}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        got[t] = df
                except Exception:
                    pass
    log.info("ohlcv final: %d/%d tickers", len(got), len(tickers))

    # Phase C: persist parquet so the other crons today reuse it
    try:
        if got:
            combined = pd.concat(got, axis=1)  # MultiIndex (ticker, OHLCV)
            combined.to_parquet(cache_path, compression="snappy")
            log.info("ohlcv cache written: %s", cache_path.name)
    except Exception as e:
        log.warning("ohlcv cache write failed: %s", e)

    return got


# ───────────────────────── Atomic IO ─────────────────────────
def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Write JSON via temp-file + rename. Frontend never sees a half-written
    predictions.json during the GitHub Actions commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=str, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ───────────────────────── Time helpers ─────────────────────────
def now_iso_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def utc_today() -> date:
    return datetime.now(timezone.utc).date()
