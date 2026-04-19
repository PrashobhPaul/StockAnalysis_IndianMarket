"""
pipeline/finbert_news.py — FinBERT sentiment + impact-tier overlay.

Per-ticker news flow:
    1. Fetch headlines (Google News RSS) for each symbol   [PARALLEL, I/O bound]
    2. Dedup near-duplicates (SequenceMatcher ≥ 0.85)
    3. Apply recency decay (24h = 1.0, 72h = 0.5, 7d = 0.1, older = 0)
    4. Classify each headline with FinBERT → (pos, neg, neu) probs
       [ONE flat batched inference pass across ALL tickers — CPU bound]
    5. Apply impact-tier keyword multiplier
         HIGH  (merger, acquisition, bankruptcy, fraud, raid)   × 1.5
         MED   (results, guidance, order, contract)             × 1.0
         LOW   (analyst, target, report)                        × 0.6
    6. Aggregate per ticker → NewsScore in [-100, +100]
    7. Rescale to 0-100 for the final composite (50 = neutral)

Degradation (never raises):
    - If transformers/torch import fails → pure keyword sentiment fallback.
    - If RSS fetch fails for a ticker → score = 50 (neutral) with note.
    - If Google News rate-limits → circuit breaker aborts remaining fetches.

FIXES (2026-04 debug pass):
  - NEWS-F1 (CRITICAL): `utils.http_get()` returns bytes, not a Response object.
    Previous code did `resp.status_code` and `resp.text`, raising AttributeError
    silently inside a try/except. Every ticker got 0 headlines. Fixed to handle
    bytes properly.
  - NEWS-F2: Parallelized RSS fetches (8 workers). Was serial for 500 tickers.
    Drops stage runtime from ~12 min to ~3 min.
  - NEWS-F3: Flattened FinBERT inference across tickers. Was doing
    per-ticker batches of 10-15 headlines; now one pass over all ~5000.
    ~3x speedup on CPU.
  - NEWS-F4: Thread-safe model load with lock. Parallel workers could race
    `_load_model()` and duplicate the 440MB load.
  - NEWS-F5: Circuit breaker — after 30 consecutive RSS failures we stop
    calling Google News and return neutral scores for remaining tickers.
  - NEWS-F6: Read FinBERT label order from `model.config.id2label` instead
    of hardcoded comment, in case the checkpoint is ever updated upstream.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import quote_plus

from pipeline.utils import get_logger, http_get

log = get_logger("finbert_news")

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
MAX_LENGTH = 256

# Parallelism + circuit-breaker tuning
RSS_WORKERS = 8
RSS_TIMEOUT = 12.0
RSS_FAILURE_CIRCUIT_BREAKER = 30  # consecutive failures before we stop trying

# ───────────────────────── Impact keyword tiers ─────────────────────────
HIGH_IMPACT = {
    "merger", "acquisition", "acquires", "takeover", "bankruptcy",
    "fraud", "sebi", "raid", "investigation", "default", "delisting",
    "demerger", "spin-off", "open offer", "preferential allotment",
}
MED_IMPACT = {
    "results", "earnings", "guidance", "order", "contract",
    "revenue", "profit", "loss", "quarterly", "q1", "q2", "q3", "q4",
    "beats", "misses", "upgrade", "downgrade", "rating",
}
LOW_IMPACT = {
    "analyst", "target", "price target", "report", "says", "views",
    "commentary", "view", "forecast",
}

# ───────────────────────── Rule-based fallback ─────────────────────────
POSITIVE_WORDS = {
    "beat", "beats", "record", "surge", "rally", "soar", "soars",
    "upgrade", "upgraded", "profit", "gain", "gains", "jump", "strong",
    "growth", "bullish", "outperform", "buy", "raises",
    "approval", "wins", "contract", "order", "milestone",
}
NEGATIVE_WORDS = {
    "miss", "misses", "slump", "fall", "falls", "plunge", "plunges",
    "downgrade", "downgraded", "loss", "losses", "warning", "weak",
    "bearish", "underperform", "sell", "cut", "cuts", "lowers",
    "probe", "raid", "fraud", "delay", "recall", "scam",
}

# Suffixes we strip from company names for better RSS matching
_COMPANY_SUFFIXES = (
    " Limited", " Ltd.", " Ltd", " Private", " Pvt", " Pvt.",
    " Corporation", " Corp.", " Corp", " Inc.", " Inc", " Company",
)


@dataclass
class NewsItem:
    title: str
    link: str
    published: Optional[datetime]
    source: str = ""


@dataclass
class TickerNewsScore:
    ticker: str
    score_0_100: float              # final rescaled score for composite
    raw_score: float                # -100 .. +100
    headline_count: int
    top_headlines: list[dict[str, Any]] = field(default_factory=list)
    method: str = "finbert"         # or "rule_fallback" or "no_news" or "error"


# ───────────────────────── Model loading (thread-safe, eager option) ─────
_model = None
_tokenizer = None
_label_index: dict[str, int] = {}  # {"positive":0,"negative":1,"neutral":2}
_model_load_failed = False
_load_lock = threading.Lock()


def _load_model():
    """
    Load FinBERT exactly once. Thread-safe via double-checked locking.
    Sets `_model_load_failed=True` on failure so future callers skip instantly.
    Also populates `_label_index` by reading model.config.id2label.
    """
    global _model, _tokenizer, _label_index, _model_load_failed
    if _model is not None:
        return _model, _tokenizer
    if _model_load_failed:
        return None, None

    with _load_lock:
        # Re-check inside the lock
        if _model is not None:
            return _model, _tokenizer
        if _model_load_failed:
            return None, None
        try:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch  # noqa: F401

            log.info("Loading FinBERT model: %s", MODEL_NAME)
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            mdl.eval()

            # NEWS-F6: read label order from config instead of trusting a comment
            id2label = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
            _label_index = {v: k for k, v in id2label.items()}
            # Safety: ensure all three labels present
            for want in ("positive", "negative", "neutral"):
                if want not in _label_index:
                    log.warning("FinBERT config missing label %r; falling back to positional order", want)
                    _label_index = {"positive": 0, "negative": 1, "neutral": 2}
                    break

            log.info("FinBERT loaded in %.1fs (labels=%s)", time.time() - t0, _label_index)
            _model, _tokenizer = mdl, tok
            return _model, _tokenizer
        except Exception as e:
            log.warning("FinBERT load failed (%s) — will use rule-based fallback", e)
            _model_load_failed = True
            return None, None


# ───────────────────────── News fetching ─────────────────────────
def _clean_company_name(name: str) -> str:
    """Strip common corporate suffixes so RSS queries match better."""
    if not name:
        return name
    cleaned = name.strip()
    # Strip common suffixes iteratively (e.g. "Foo Ltd." → "Foo")
    changed = True
    while changed:
        changed = False
        for suf in _COMPANY_SUFFIXES:
            if cleaned.lower().endswith(suf.lower()):
                cleaned = cleaned[: -len(suf)].rstrip(" ,.")
                changed = True
    return cleaned or name


def _google_news_rss(query: str, max_items: int = 15) -> list[NewsItem]:
    """
    Fetch headlines from Google News RSS. Returns list of NewsItem.
    Empty list on any failure — never raises.

    NEWS-F1: `http_get` returns Optional[bytes], NOT a Response object.
    Previous code accessing `.status_code` / `.text` raised AttributeError
    silently, which is why every ticker got 0 headlines.
    """
    if not query or not query.strip():
        return []

    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{query} when:7d")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        body = http_get(url, timeout=RSS_TIMEOUT, retries=2)
        if not body:
            return []
        xml = body.decode("utf-8", errors="ignore")
    except Exception as e:
        log.debug("RSS fetch failed for %s: %s", query, e)
        return []

    items: list[NewsItem] = []
    # Lightweight parsing — avoids adding feedparser dep
    entries = re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL)
    for entry in entries[:max_items]:
        title_m = re.search(r"<title>(.*?)</title>", entry, flags=re.DOTALL)
        link_m = re.search(r"<link>(.*?)</link>", entry)
        pub_m = re.search(r"<pubDate>(.*?)</pubDate>", entry)
        src_m = re.search(r"<source[^>]*>(.*?)</source>", entry)
        if not title_m:
            continue
        title = re.sub(r"<!\[CDATA\[|\]\]>", "", title_m.group(1)).strip()
        title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()  # trim "- Source" suffix
        if not title:
            continue
        published = None
        if pub_m:
            try:
                from email.utils import parsedate_to_datetime
                published = parsedate_to_datetime(pub_m.group(1))
            except Exception:
                published = None
        items.append(NewsItem(
            title=title,
            link=link_m.group(1).strip() if link_m else "",
            published=published,
            source=re.sub(r"<!\[CDATA\[|\]\]>", "", src_m.group(1)).strip() if src_m else "",
        ))
    return items


def _dedupe(items: list[NewsItem], threshold: float = 0.85) -> list[NewsItem]:
    """Remove near-duplicate headlines via SequenceMatcher."""
    kept: list[NewsItem] = []
    seen_hashes: set[str] = set()
    for it in items:
        norm = re.sub(r"\W+", " ", it.title.lower()).strip()
        h = hashlib.md5(norm.encode()).hexdigest()
        if h in seen_hashes:
            continue
        is_dup = False
        for k in kept:
            if SequenceMatcher(None, norm, re.sub(r"\W+", " ", k.title.lower())).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(it)
            seen_hashes.add(h)
    return kept


def _recency_weight(published: Optional[datetime]) -> float:
    """
    1.0 if ≤24h old, 0.5 if ≤72h, 0.1 if ≤7d, 0 otherwise.
    None timestamps default to 0.3 (assume mid-range).
    """
    if published is None:
        return 0.3
    try:
        now = datetime.now(timezone.utc)
        pub = published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_hrs = (now - pub).total_seconds() / 3600
        if age_hrs <= 24:
            return 1.0
        if age_hrs <= 72:
            return 0.5
        if age_hrs <= 168:  # 7 days
            return 0.1
        return 0.0
    except Exception:
        return 0.3


def _impact_multiplier(title: str) -> tuple[float, str]:
    """Return (multiplier, tier) for a headline."""
    t = title.lower()
    if any(kw in t for kw in HIGH_IMPACT):
        return 1.5, "HIGH"
    if any(kw in t for kw in MED_IMPACT):
        return 1.0, "MED"
    if any(kw in t for kw in LOW_IMPACT):
        return 0.6, "LOW"
    return 0.8, "GEN"


# ───────────────────────── Classification ─────────────────────────
def _classify_batch_finbert(texts: list[str]) -> list[tuple[float, float, float]]:
    """
    Run FinBERT on a (potentially large) list of texts. Returns list of
    (pos, neg, neu) probabilities, same length/order as `texts`.

    Internally chunked to BATCH_SIZE so peak memory stays bounded.
    Returns [] if the model is unavailable; callers should fall back to
    rule-based classification in that case.
    """
    if not texts:
        return []
    model, tok = _load_model()
    if model is None or tok is None:
        return []

    try:
        import torch
    except Exception as e:
        log.warning("torch unavailable at inference time: %s", e)
        return []

    pos_i = _label_index.get("positive", 0)
    neg_i = _label_index.get("negative", 1)
    neu_i = _label_index.get("neutral", 2)

    out: list[tuple[float, float, float]] = []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                chunk = texts[i:i + BATCH_SIZE]
                try:
                    enc = tok(
                        chunk, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt",
                    )
                    logits = model(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    for row in probs:
                        out.append((
                            float(row[pos_i]),
                            float(row[neg_i]),
                            float(row[neu_i]),
                        ))
                except Exception as e:
                    # Per-chunk failure → neutral fallback for just that chunk
                    log.warning("FinBERT chunk %d failed: %s — using neutral fallback", i, e)
                    out.extend((0.15, 0.15, 0.70) for _ in chunk)
    except Exception as e:
        log.warning("FinBERT inference loop failed: %s", e)
        return []
    return out


def _classify_rule_based(title: str) -> tuple[float, float, float]:
    """Cheap keyword sentiment for fallback. Returns (pos, neg, neu)."""
    words = set(re.findall(r"\w+", title.lower()))
    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)
    if pos_hits == 0 and neg_hits == 0:
        return (0.15, 0.15, 0.70)
    total = pos_hits + neg_hits
    p = pos_hits / total * 0.8 + 0.1
    n = neg_hits / total * 0.8 + 0.1
    neu = 1.0 - p - n
    return (p, n, max(0.0, neu))


# ───────────────────────── Aggregation ─────────────────────────
def _aggregate_ticker(
    ticker: str,
    items_with_probs: list[tuple[NewsItem, tuple[float, float, float]]],
    method: str,
) -> TickerNewsScore:
    """Combine per-headline probs into a single TickerNewsScore."""
    if not items_with_probs:
        return TickerNewsScore(
            ticker=ticker, score_0_100=50.0, raw_score=0.0,
            headline_count=0, method="no_news",
        )

    weighted_sum = 0.0
    total_weight = 0.0
    top: list[dict[str, Any]] = []

    for item, (p_pos, p_neg, _p_neu) in items_with_probs:
        recency = _recency_weight(item.published)
        impact, tier = _impact_multiplier(item.title)
        weight = recency * impact
        item_score = p_pos - p_neg  # in [-1, +1]
        weighted_sum += item_score * weight
        total_weight += weight
        top.append({
            "title": item.title,
            "source": item.source,
            "tier": tier,
            "recency_weight": round(recency, 2),
            "pos": round(p_pos, 3),
            "neg": round(p_neg, 3),
            "score": round(item_score, 3),
        })

    raw = (weighted_sum / total_weight) * 100.0 if total_weight > 0 else 0.0
    raw = max(-80.0, min(80.0, raw))
    score_0_100 = 50.0 + raw * 0.5  # [-80,80] → [10,90]

    top = sorted(top, key=lambda x: abs(x["score"]), reverse=True)[:5]

    return TickerNewsScore(
        ticker=ticker,
        score_0_100=round(score_0_100, 2),
        raw_score=round(raw, 2),
        headline_count=len(items_with_probs),
        top_headlines=top,
        method=method,
    )


# ───────────────────────── Thread-safe failure tracker ─────────────────────
class _CircuitBreaker:
    """Tracks consecutive RSS failures across worker threads."""
    def __init__(self, limit: int):
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._tripped = False
        self._limit = limit

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """Returns True if this failure trips the breaker."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._limit and not self._tripped:
                self._tripped = True
                return True
            return False

    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped


def _fetch_one_ticker_rss(
    display: str,
    company_name: Optional[str],
    breaker: _CircuitBreaker,
) -> tuple[str, list[NewsItem]]:
    """Worker for parallel RSS fetch. Respects the circuit breaker."""
    if breaker.is_tripped():
        return display, []

    query = _clean_company_name(company_name) if company_name else display
    try:
        raw_items = _google_news_rss(query, max_items=15)
    except Exception as e:
        log.debug("unexpected RSS exception for %s: %s", display, e)
        raw_items = []

    if raw_items:
        breaker.record_success()
        items = _dedupe(raw_items)[:10]
        return display, items
    else:
        if breaker.record_failure():
            log.warning(
                "RSS circuit breaker TRIPPED after %d consecutive empty fetches. "
                "Remaining tickers will receive neutral scores.",
                RSS_FAILURE_CIRCUIT_BREAKER,
            )
        return display, []


# ───────────────────────── Public API ─────────────────────────
def score_ticker_news(
    ticker_display: str,
    company_name: Optional[str] = None,
    use_finbert: bool = True,
) -> TickerNewsScore:
    """
    Compute NewsScore for one ticker. Kept for backwards-compat / smoke tests.
    For large universes, prefer score_universe_news (batched + parallel).
    """
    _, items = _fetch_one_ticker_rss(
        ticker_display, company_name, _CircuitBreaker(RSS_FAILURE_CIRCUIT_BREAKER),
    )
    if not items:
        return TickerNewsScore(
            ticker=ticker_display, score_0_100=50.0, raw_score=0.0,
            headline_count=0, method="no_news",
        )

    titles = [i.title for i in items]
    probs: list[tuple[float, float, float]] = []
    method = "rule_fallback"
    if use_finbert:
        probs = _classify_batch_finbert(titles)
        if probs:
            method = "finbert"
    if not probs:
        probs = [_classify_rule_based(t) for t in titles]

    return _aggregate_ticker(
        ticker_display, list(zip(items, probs)), method,
    )


def score_universe_news(
    tickers: list[tuple[str, Optional[str]]],
    use_finbert: bool = True,
) -> dict[str, TickerNewsScore]:
    """
    Score an entire list of (display_ticker, company_name) tuples.

    Architecture (NEWS-F2 + NEWS-F3):
      Phase A  [parallel 8 workers]    Fetch all RSS feeds concurrently.
                                       Circuit breaker aborts early if we hit
                                       sustained Google News rate-limiting.
      Phase B  [single-threaded batch] Flatten all headlines into one flat
                                       list, run FinBERT inference in chunks
                                       of BATCH_SIZE. ~3x faster than
                                       per-ticker inference.
      Phase C  [fast in-memory]        Re-distribute probs back to tickers
                                       and aggregate.

    For 500 tickers × ~10 headlines each (~5000 headlines total), expected
    runtime on a GitHub Actions ubuntu-latest runner:
      - Phase A: ~2-3 min  (was ~10-12 min serial)
      - Phase B: ~1-2 min  (was ~4-6 min per-ticker)
      - Phase C: <5s
    """
    if not tickers:
        return {}

    log.info("News stage: %d tickers, use_finbert=%s, workers=%d",
             len(tickers), use_finbert, RSS_WORKERS)

    # Pre-warm the FinBERT model BEFORE spawning workers, so the eager load
    # happens in the main thread. Makes failures visible up front.
    if use_finbert:
        _load_model()

    breaker = _CircuitBreaker(RSS_FAILURE_CIRCUIT_BREAKER)

    # ── Phase A: parallel RSS fetch ──────────────────────────────────
    ticker_items: dict[str, list[NewsItem]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=RSS_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one_ticker_rss, disp, name, breaker): disp
            for disp, name in tickers
        }
        for idx, fut in enumerate(as_completed(futures), 1):
            disp = futures[fut]
            try:
                _, items = fut.result()
                ticker_items[disp] = items
            except Exception as e:
                log.warning("RSS worker failed for %s: %s", disp, e)
                ticker_items[disp] = []
            if idx % 50 == 0:
                log.info("RSS progress: %d/%d (elapsed %.1fs)",
                         idx, len(tickers), time.time() - t0)
    log.info("Phase A (RSS) done in %.1fs. Tickers with headlines: %d/%d",
             time.time() - t0, sum(1 for v in ticker_items.values() if v), len(tickers))

    # ── Phase B: flatten + classify ──────────────────────────────────
    # Build parallel arrays so we can re-zip after inference
    flat_owners: list[str] = []      # ticker each headline belongs to
    flat_items: list[NewsItem] = []
    for disp, items in ticker_items.items():
        for it in items:
            flat_owners.append(disp)
            flat_items.append(it)

    method = "rule_fallback"
    probs: list[tuple[float, float, float]] = []

    if flat_items:
        t1 = time.time()
        if use_finbert:
            probs = _classify_batch_finbert([it.title for it in flat_items])
            if probs and len(probs) == len(flat_items):
                method = "finbert"
            else:
                log.warning("FinBERT returned %d probs for %d texts — falling back to rules",
                            len(probs), len(flat_items))
                probs = []
        if not probs:
            probs = [_classify_rule_based(it.title) for it in flat_items]
            method = "rule_fallback"
        log.info("Phase B (inference) done in %.1fs — %d headlines, method=%s",
                 time.time() - t1, len(flat_items), method)

    # ── Phase C: per-ticker aggregation ──────────────────────────────
    grouped: dict[str, list[tuple[NewsItem, tuple[float, float, float]]]] = {
        disp: [] for disp, _ in tickers
    }
    for disp, item, p in zip(flat_owners, flat_items, probs):
        grouped.setdefault(disp, []).append((item, p))

    results: dict[str, TickerNewsScore] = {}
    for disp, _ in tickers:
        results[disp] = _aggregate_ticker(disp, grouped.get(disp, []), method)

    # Summary stats
    non_neutral = sum(1 for r in results.values() if abs(r.raw_score) > 5)
    tripped_note = " [circuit breaker tripped]" if breaker.is_tripped() else ""
    log.info("News stage DONE — %d tickers scored, %d non-neutral%s",
             len(results), non_neutral, tripped_note)
    return results


if __name__ == "__main__":
    # Quick smoke test — does RSS come back at all?
    import json
    r = score_ticker_news("RELIANCE", "Reliance Industries", use_finbert=False)
    print(json.dumps({
        "ticker": r.ticker,
        "score_0_100": r.score_0_100,
        "headline_count": r.headline_count,
        "method": r.method,
        "sample_titles": [h["title"] for h in r.top_headlines[:3]],
    }, indent=2, default=str))
