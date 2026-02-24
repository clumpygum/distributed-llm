"""
cache.py

QueryCache — Thread-safe LRU cache with TTL, semantic similarity lookup,
per-entry routing history, predictive device routing, duplicate-insert
prevention, pattern-level invalidation, warm-up, and disk persistence.

Imported by:
  query_routing_engine.py  →  from cache import CacheEntry, CacheLookupResult, QueryCache
  router.py                →  via query_routing_engine / QueryRouter._cache

Predictive routing
------------------
Rather than blindly returning "nano" on every cache hit, the cache tracks a
routing_history per entry — a list of (device, confidence, method) tuples from
every real routing decision made for that query.  On lookup, a weighted-majority
vote over that history (with recency decay) predicts the best device.  The
caller receives both the CacheEntry and the CacheLookupResult which includes:
  - predicted_device  : "nano" | "orin"
  - predicted_confidence : float in [0, 1]
  - use_hybrid_fallback : True if confidence is below the ambiguity threshold,
                          signalling the caller should re-run the hybrid router
"""

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum weighted-vote confidence to trust the cache prediction.
# Below this the caller should fall back to the hybrid router.
PREDICTION_CONFIDENCE_THRESHOLD = 0.60

# How strongly to decay older routing history entries (per position from newest).
# 0.85 means: newest weight=1.0, second=0.85, third=0.72, ...
RECENCY_DECAY = 0.85


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RoutingRecord:
    """A single routing decision recorded for a cached query."""
    device: str           # "nano" | "orin"
    confidence: float     # original router confidence
    method: str           # e.g. "hybrid", "heuristic", "token"
    timestamp: str        # isoformat — for persistence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "confidence": self.confidence,
            "method": self.method,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingRecord":
        return cls(
            device=d["device"],
            confidence=float(d["confidence"]),
            method=d.get("method", "unknown"),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class CacheEntry:
    query: str
    query_hash: str
    context_key: str
    embedding: Optional[np.ndarray]
    timestamp: datetime
    device_used: str                          # device from the most recent routing
    response_time: Optional[float] = None
    hit_count: int = 0
    routing_history: List[RoutingRecord] = field(default_factory=list)  # NEW

    # Keep history bounded so memory doesn't grow unbounded
    MAX_HISTORY = 20

    def record_routing(self, device: str, confidence: float, method: str) -> None:
        """Append a new routing decision to history, capped at MAX_HISTORY."""
        self.routing_history.append(RoutingRecord(
            device=device,
            confidence=confidence,
            method=method,
            timestamp=datetime.now().isoformat(),
        ))
        if len(self.routing_history) > self.MAX_HISTORY:
            self.routing_history = self.routing_history[-self.MAX_HISTORY:]
        self.device_used = device

    def predict_device(self) -> Tuple[str, float]:
        """
        Weighted-majority vote over routing_history with recency decay.

        Returns (predicted_device, confidence) where confidence is the
        fractional vote share of the winning device in [0, 1].
        Falls back to device_used if history is empty.
        """
        if not self.routing_history:
            return self.device_used, 0.5

        nano_score = 0.0
        orin_score = 0.0
        total_weight = 0.0

        # Iterate newest-first so decay applies correctly
        for i, record in enumerate(reversed(self.routing_history)):
            weight = (RECENCY_DECAY ** i) * record.confidence
            if record.device == "orin":
                orin_score += weight
            else:
                nano_score += weight
            total_weight += weight

        if total_weight < 1e-9:
            return self.device_used, 0.5

        if orin_score >= nano_score:
            predicted = "orin"
            confidence = orin_score / total_weight
        else:
            predicted = "nano"
            confidence = nano_score / total_weight

        return predicted, float(min(confidence, 1.0))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_hash": self.query_hash,
            "context_key": self.context_key,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "device_used": self.device_used,
            "response_time": self.response_time,
            "hit_count": self.hit_count,
            "routing_history": [r.to_dict() for r in self.routing_history],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        emb = np.array(d["embedding"], dtype=np.float32) if d.get("embedding") else None
        history = [
            RoutingRecord.from_dict(r)
            for r in d.get("routing_history", [])
        ]
        return cls(
            query=d["query"],
            query_hash=d["query_hash"],
            context_key=d["context_key"],
            embedding=emb,
            timestamp=datetime.fromisoformat(d["timestamp"]),
            device_used=d["device_used"],
            response_time=d.get("response_time"),
            hit_count=d.get("hit_count", 0),
            routing_history=history,
        )


@dataclass
class CacheLookupResult:
    """
    Returned by QueryCache.lookup alongside the CacheEntry.

    predicted_device    — best device based on routing history vote
    predicted_confidence — vote share of winning device [0, 1]
    use_hybrid_fallback  — True if confidence < PREDICTION_CONFIDENCE_THRESHOLD,
                           caller should re-run hybrid router instead of trusting cache
    """
    entry: CacheEntry
    predicted_device: str
    predicted_confidence: float
    use_hybrid_fallback: bool


# =============================================================================
# QueryCache
# =============================================================================

class QueryCache:
    """
    Thread-safe LRU cache with TTL, semantic similarity, and predictive routing.

    Internal store is an OrderedDict keyed by query_hash:
      - O(1) exact-match lookup
      - O(1) LRU promotion via move_to_end
      - Duplicate inserts refresh in-place (no duplicates under concurrency)
      - Stale entries are preferred for eviction over hot valid ones
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 300,
        similarity_threshold: float = 0.85,
        use_semantic: bool = True,
        prediction_confidence_threshold: float = PREDICTION_CONFIDENCE_THRESHOLD,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        self.prediction_confidence_threshold = prediction_confidence_threshold

        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self._hits = 0
        self._attempts = 0
        self._evictions = 0
        self._hybrid_fallbacks = 0   # how often prediction confidence was too low

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: str,
        context_key: str,
        q_emb: Optional[np.ndarray] = None,
    ) -> Optional[CacheLookupResult]:
        """
        Return a CacheLookupResult (entry + device prediction) or None on miss.
        Checks exact hash match first, then semantic similarity.

        The caller should check result.use_hybrid_fallback — if True, the
        prediction confidence is low and the hybrid router should re-route.
        """
        self._attempts += 1
        self._evict_expired()

        qhash = self._make_hash(query, context_key)
        entry = None

        with self._lock:
            # 1. Exact match — O(1)
            candidate = self._store.get(qhash)
            if candidate is not None and candidate.context_key == context_key:
                if self._is_valid(candidate):
                    candidate.hit_count += 1
                    self._store.move_to_end(qhash)
                    self._hits += 1
                    entry = candidate
                else:
                    self._delete_entry(qhash)

            # 2. Semantic match — snapshot candidates, release lock before numpy ops
            if entry is None:
                if not self.use_semantic or q_emb is None:
                    return None
                candidates: List[Tuple[np.ndarray, str]] = [
                    (e.embedding.copy(), h)
                    for h, e in self._store.items()
                    if e.context_key == context_key
                    and e.embedding is not None
                    and self._is_valid(e)
                ]

        # Cosine similarity outside the lock
        if entry is None and q_emb is not None:
            best_hash: Optional[str] = None
            best_sim = 0.0
            norm_q = float(np.linalg.norm(q_emb))
            if norm_q < 1e-9:
                return None

            for emb_copy, h in candidates:
                norm_c = float(np.linalg.norm(emb_copy))
                if norm_c < 1e-9:
                    continue
                sim = float(np.dot(q_emb, emb_copy) / (norm_q * norm_c))
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim, best_hash = sim, h

            if best_hash is None:
                return None

            with self._lock:
                candidate = self._store.get(best_hash)
                if candidate is None or not self._is_valid(candidate):
                    return None
                candidate.hit_count += 1
                self._store.move_to_end(best_hash)
                self._hits += 1
                entry = candidate

        if entry is None:
            return None

        # Build prediction from routing history
        predicted_device, predicted_confidence = entry.predict_device()
        use_fallback = predicted_confidence < self.prediction_confidence_threshold

        if use_fallback:
            self._hybrid_fallbacks += 1
            logger.debug(
                "Cache hit but low prediction confidence=%.2f for %r — hybrid fallback",
                predicted_confidence, query[:60],
            )

        return CacheLookupResult(
            entry=entry,
            predicted_device=predicted_device,
            predicted_confidence=predicted_confidence,
            use_hybrid_fallback=use_fallback,
        )

    def insert(
        self,
        query: str,
        context_key: str,
        device: str,
        confidence: float = 1.0,
        method: str = "unknown",
        q_emb: Optional[np.ndarray] = None,
        response_time: Optional[float] = None,
    ) -> None:
        """
        Insert or refresh an entry, recording the routing decision in history.
        If the hash already exists, the entry is updated in-place (no duplicate).
        """
        qhash = self._make_hash(query, context_key)

        with self._lock:
            if qhash in self._store:
                existing = self._store[qhash]
                existing.timestamp = datetime.now()
                existing.record_routing(device, confidence, method)
                if q_emb is not None:
                    existing.embedding = q_emb.copy()
                if response_time is not None:
                    existing.response_time = response_time
                self._store.move_to_end(qhash)
                return

            if len(self._store) >= self.max_size:
                self._evict_one()

            entry = CacheEntry(
                query=query,
                query_hash=qhash,
                context_key=context_key,
                embedding=q_emb.copy() if q_emb is not None else None,
                timestamp=datetime.now(),
                device_used=device,
                response_time=response_time,
            )
            entry.record_routing(device, confidence, method)
            self._store[qhash] = entry
            self._store.move_to_end(qhash)

    def invalidate(
        self,
        context_key: Optional[str] = None,
        query_pattern: Optional[str] = None,
    ) -> int:
        """
        Remove entries matching context_key and/or a query_pattern regex.
        Passing neither clears everything.
        Returns number of entries removed.
        """
        pattern = re.compile(query_pattern, re.IGNORECASE) if query_pattern else None
        removed = 0

        with self._lock:
            to_delete = [
                h for h, e in self._store.items()
                if (context_key is None or e.context_key == context_key)
                and (pattern is None or pattern.search(e.query))
            ]
            for h in to_delete:
                self._delete_entry(h)
                removed += 1

        logger.debug("Cache invalidate removed %d entries", removed)
        return removed

    def warm_up(
        self,
        pairs: List[Tuple[str, str, str]],   # (query, context_key, device)
        embedder: Any = None,
    ) -> None:
        """
        Pre-populate the cache before live traffic arrives.
        Optionally encodes embeddings if a SentenceTransformer is provided.
        """
        texts = [q for q, _, _ in pairs]
        embeddings: List[Optional[np.ndarray]] = [None] * len(pairs)

        if embedder is not None:
            try:
                vecs = embedder.encode(texts)
                embeddings = list(vecs)
            except Exception as exc:
                logger.warning("warm_up: embedding failed, skipping vectors: %s", exc)

        for (query, ctx_key, device), emb in zip(pairs, embeddings):
            self.insert(query, ctx_key, device, q_emb=emb)

        logger.info("Cache warmed up with %d entries", len(pairs))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise non-expired entries to a JSON file."""
        self._evict_expired()
        with self._lock:
            data = [e.to_dict() for e in self._store.values()]
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Cache saved %d entries → %s", len(data), path)

    def load(self, path: str) -> int:
        """
        Restore entries from a JSON file.
        Expired entries are discarded on load.
        Returns the number of entries loaded.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("Cache load: file not found — %s", path)
            return 0

        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Cache load: JSON parse error: %s", exc)
            return 0

        loaded = 0
        with self._lock:
            for d in raw:
                try:
                    entry = CacheEntry.from_dict(d)
                except Exception as exc:
                    logger.warning("Cache load: skipping malformed entry: %s", exc)
                    continue
                if not self._is_valid(entry):
                    continue
                self._store[entry.query_hash] = entry
                loaded += 1

        logger.info("Cache loaded %d entries from %s", loaded, path)
        return loaded

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of cache health metrics."""
        with self._lock:
            valid = sum(1 for e in self._store.values() if self._is_valid(e))
            stale = len(self._store) - valid
            top_hot = sorted(
                self._store.values(), key=lambda e: e.hit_count, reverse=True
            )[:5]

        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "valid": valid,
            "stale": stale,
            "hits": self._hits,
            "attempts": self._attempts,
            "hit_rate": round(self._hits / max(self._attempts, 1), 4),
            "evictions": self._evictions,
            "hybrid_fallbacks": self._hybrid_fallbacks,
            "top_queries": [
                {
                    "query": e.query[:60],
                    "hits": e.hit_count,
                    "predicted_device": e.predict_device()[0],
                    "predicted_confidence": round(e.predict_device()[1], 3),
                    "history_len": len(e.routing_history),
                }
                for e in top_hot
            ],
        }

    def clear(self) -> None:
        """Wipe the entire cache and reset all stats."""
        with self._lock:
            for e in self._store.values():
                if e.embedding is not None:
                    del e.embedding
            self._store.clear()
            self._hits = 0
            self._attempts = 0
            self._evictions = 0
            self._hybrid_fallbacks = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hash(query: str, context_key: str) -> str:
        key = f"{context_key}||{query.lower().strip()}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _is_valid(self, entry: CacheEntry) -> bool:
        return (datetime.now() - entry.timestamp).total_seconds() <= self.ttl_seconds

    def _delete_entry(self, h: str) -> None:
        """Remove an entry and free its embedding. Must be called with lock held."""
        entry = self._store.pop(h, None)
        if entry is not None and entry.embedding is not None:
            del entry.embedding

    def _evict_expired(self) -> None:
        """Remove all TTL-expired entries."""
        with self._lock:
            expired = [h for h, e in self._store.items() if not self._is_valid(e)]
            for h in expired:
                self._delete_entry(h)
                self._evictions += 1

    def _evict_one(self) -> None:
        """
        Evict one entry when at capacity.
        Prefers stale entries; falls back to LRU (front of OrderedDict).
        Must be called with lock held.
        """
        for h, e in list(self._store.items()):
            if not self._is_valid(e):
                self._delete_entry(h)
                self._evictions += 1
                return
        if self._store:
            h = next(iter(self._store))
            self._delete_entry(h)
            self._evictions += 1