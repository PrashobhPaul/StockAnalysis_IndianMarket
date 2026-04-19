/**
 * ProfitPilot v3 Service Worker
 *
 * Strategy: network-first for predictions.json and news_cache.json (they
 * change every day — stale caches were the v2 bug that caused the splash
 * screen hang). Cache-first for static shell (index.html, manifest, icons).
 *
 * Version bumped every deploy so the browser evicts the old SW reliably.
 */
const SW_VERSION = "v3.0.0";
const STATIC_CACHE = `pp-static-${SW_VERSION}`;
const RUNTIME_CACHE = `pp-runtime-${SW_VERSION}`;

const STATIC_ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

// JSON endpoints: never cache, always fetch fresh (with cache fallback on error)
const NETWORK_FIRST_PATTERNS = [
  /predictions\.json/,
  /news_cache\.json/,
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => cache.addAll(STATIC_ASSETS).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== STATIC_CACHE && k !== RUNTIME_CACHE)
            .map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const req = event.request;

  // Only handle GETs over same-origin
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const isNetworkFirst = NETWORK_FIRST_PATTERNS.some(rx => rx.test(url.pathname));

  if (isNetworkFirst) {
    event.respondWith(networkFirst(req));
  } else {
    event.respondWith(cacheFirst(req));
  }
});

async function networkFirst(req) {
  try {
    const fresh = await fetch(req, { cache: "no-store" });
    if (fresh && fresh.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      cache.put(req, fresh.clone()).catch(() => {});
    }
    return fresh;
  } catch (e) {
    const cached = await caches.match(req);
    if (cached) return cached;
    return new Response(
      JSON.stringify({ error: "offline", message: String(e) }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }
}

async function cacheFirst(req) {
  const cached = await caches.match(req);
  if (cached) {
    // Refresh in background
    fetch(req).then(res => {
      if (res && res.ok) {
        caches.open(STATIC_CACHE).then(c => c.put(req, res.clone())).catch(() => {});
      }
    }).catch(() => {});
    return cached;
  }
  try {
    const res = await fetch(req);
    if (res && res.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(req, res.clone()).catch(() => {});
    }
    return res;
  } catch (e) {
    // Shell offline fallback
    if (req.mode === "navigate") {
      const fallback = await caches.match("./index.html");
      if (fallback) return fallback;
    }
    return new Response("Offline", { status: 503 });
  }
}

// Allow instant updates via postMessage({ type: "SKIP_WAITING" })
self.addEventListener("message", event => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
