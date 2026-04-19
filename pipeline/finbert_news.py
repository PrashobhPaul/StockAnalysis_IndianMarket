"""
pipeline/finbert_news.py — FinBERT sentiment + impact-tier overlay.

Per-ticker news flow:
    1. Fetch headlines (Google News RSS) for each symbol
    2. Dedup near-duplicates (SequenceMatcher ≥ 0.85)
    3. Apply recency decay (24h = 1.0, 72h = 0.5, 7d = 0.1, older = 0)
    4. Classify each headline with FinBERT → (positive, negative, neutral) probs
    5. Apply impact-tier keyword multiplier
         HIGH  (merger, acquisition, bankruptcy, fraud, raid)   × 1.5
         MED   (results, guidance, order, contract)             × 1.0
         LOW   (analyst, target, report)                        × 0.6
    6. Aggregate per ticker → NewsScore in [-100, +100]
    7. Rescale to 0-100 for the final composite (50 = neutral)

Degradation:
    - If transformers/torch import fails → fall back to pure keyword sentiment.
    - If RSS fetch fails for a ticker → score = 50 (neutral) with note.
    - Never raises. Every exception is logged and neutralized.

The model is cached via HF_HOME environment variable (set in daily.yml to
.cache/huggingface, which actions/cache@v4 persists across runs).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import quote_plus

from pipeline.utils import get_logger, http_get

log = get_logger("finbert_news")

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
MAX_LENGTH = 256

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
    "growth", "bullish", "outperform", "buy", "upgraded", "raises",
    "approval", "wins", "contract", "order", "milestone",
}
NEGATIVE_WORDS = {
    "miss", "misses", "slump", "fall", "falls", "plunge", "plunges",
    "downgrade", "downgraded", "loss", "losses", "warning", "weak",
    "bearish", "underperform", "sell", "cut", "cuts", "lowers",
    "probe", "raid", "fraud", "delay", "recall", "scam",
}


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
    method: str = "finbert"         # or "rule_fallback" or "no_news"


# ───────────────────────── Model loading (lazy) ─────────────────────────
_model = None
_tokenizer = None
_model_load_failed = False


def _load_model():
    """Lazy-load FinBERT. Sets `_model_load_failed=True` to skip future attempts."""
    global _model, _tokenizer, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model, _tokenizer

    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F401  (ensures torch is importable)

        log.info("Loading FinBERT model: %s", MODEL_NAME)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        mdl.eval()
        log.info("FinBERT loaded in %.1fs", time.time() - t0)
        _model, _tokenizer = mdl, tok
        return _model, _tokenizer
    except Exception as e:
        log.warning("FinBERT load failed (%s) — will use rule-based fallback", e)
        _model_load_failed = True
        return None, None


# ───────────────────────── News fetching ─────────────────────────
def _google_news_rss(query: str, max_items: int = 20) -> list[NewsItem]:
    """
    Fetch headlines from Google News RSS. Returns list of NewsItem.
    Empty list on any failure — never raises.
    """
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{query} when:7d")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        resp = http_get(url, timeout=15)
        if not resp or resp.status_code != 200:
            return []
        xml = resp.text
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
    Run FinBERT on a batch. Returns list of (pos, neg, neu) probabilities.
    Empty list if model unavailable.
    """
    model, tok = _load_model()
    if model is None or tok is None:
        return []
    try:
        import torch
        with torch.no_grad():
            out: list[tuple[float, float, float]] = []
            for i in range(0, len(texts), BATCH_SIZE):
                chunk = texts[i:i + BATCH_SIZE]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # FinBERT label order: [positive, negative, neutral]
                for row in probs:
                    out.append((float(row[0]), float(row[1]), float(row[2])))
            return out
    except Exception as e:
        log.warning("FinBERT inference failed: %s", e)
        return []


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
def score_ticker_news(
    ticker_display: str,
    company_name: Optional[str] = None,
    use_finbert: bool = True,
) -> TickerNewsScore:
    """
    Compute NewsScore for one ticker.

    Args:
        ticker_display: human-readable symbol (e.g. "RELIANCE")
        company_name:   optional full name for better RSS matches
        use_finbert:    set False for the smoke test / fast path
    """
    query = company_name or ticker_display
    raw_items = _google_news_rss(query, max_items=15)
    if not raw_items:
        return TickerNewsScore(
            ticker=ticker_display, score_0_100=50.0, raw_score=0.0,
            headline_count=0, method="no_news",
        )

    items = _dedupe(raw_items)[:10]
    titles = [i.title for i in items]

    probs: list[tuple[float, float, float]] = []
    method = "rule_fallback"
    if use_finbert:
        probs = _classify_batch_finbert(titles)
        if probs:
            method = "finbert"
    if not probs:
        probs = [_classify_rule_based(t) for t in titles]

    weighted_sum = 0.0
    total_weight = 0.0
    top: list[dict[str, Any]] = []

    for item, (p_pos, p_neg, _p_neu) in zip(items, probs):
        recency = _recency_weight(item.published)
        impact, tier = _impact_multiplier(item.title)
        weight = recency * impact
        # Per-item score in [-1, +1]
        item_score = p_pos - p_neg
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

    if total_weight > 0:
        raw = (weighted_sum / total_weight) * 100.0  # -100 .. +100
    else:
        raw = 0.0

    # Rescale to 0-100 (50 = neutral). Cap ±80 to prevent single strong
    # headline from fully saturating.
    raw = max(-80.0, min(80.0, raw))
    score_0_100 = 50.0 + raw * 0.5  # [-80,80] → [10,90]

    top = sorted(top, key=lambda x: abs(x["score"]), reverse=True)[:5]

    return TickerNewsScore(
        ticker=ticker_display,
        score_0_100=round(score_0_100, 2),
        raw_score=round(raw, 2),
        headline_count=len(items),
        top_headlines=top,
        method=method,
    )


def score_universe_news(
    tickers: list[tuple[str, Optional[str]]],
    use_finbert: bool = True,
) -> dict[str, TickerNewsScore]:
    """
    Score a list of (display_ticker, company_name) tuples.

    Not parallelized — Google News has tight rate limits and FinBERT
    inference is CPU-bound; a single-threaded sweep of 500 tickers
    finishes in ~8-12 minutes which fits the daily.yml budget.
    """
    results: dict[str, TickerNewsScore] = {}
    for idx, (tk, name) in enumerate(tickers, 1):
        try:
            results[tk] = score_ticker_news(tk, name, use_finbert=use_finbert)
        except Exception as e:
            log.warning("news scoring failed for %s: %s", tk, e)
            results[tk] = TickerNewsScore(
                ticker=tk, score_0_100=50.0, raw_score=0.0,
                headline_count=0, method="error",
            )
        if idx % 25 == 0:
            log.info("news progress: %d/%d", idx, len(tickers))
    return results


if __name__ == "__main__":
    # Quick smoke test
    import json
    r = score_ticker_news("RELIANCE", "Reliance Industries", use_finbert=False)
    print(json.dumps(r.__dict__, indent=2, default=str))
