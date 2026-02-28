"""
query_routing_engine.py

Dynamic Query Router for Distributed Edge Chatbots on NVIDIA Jetson Platforms
- Multiple routing strategies: token / semantic (trained centroids) / heuristic / hybrid / perf
- Cache backed by QueryCache from cache.py:
    * Predictive routing via weighted routing history
    * Context-aware override (heavy context overrides cached nano prediction)
    * Hybrid re-route on low prediction confidence
- Thread-safe implementation for production use
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from cache import CacheEntry, CacheLookupResult, QueryCache
except ImportError as _e:
    raise ImportError(
        "Could not import 'cache.py'. Ensure cache.py is in the same directory "
        f"as query_routing_engine.py (or on PYTHONPATH). Original error: {_e}"
    ) from _e

logger = logging.getLogger(__name__)

# Optional deps
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Semantic routing disabled.")

try:
    from litellm import token_counter
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm not available. Token-based routing will use approximation.")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class RoutingDecision:
    device: Literal["nano", "orin"]
    confidence: float
    method: str
    reasoning: str
    complexity_score: Optional[float] = None
    cache_hit: bool = False


# =============================================================================
# Base Strategy
# =============================================================================

class BaseRouter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        raise NotImplementedError


# =============================================================================
# Token Strategy
# =============================================================================

class TokenBasedRouter(BaseRouter):
    """Token-counting router. device = ORIN if tokens > threshold else NANO."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = int(config.get("token_threshold", 1000))
        self.model = config.get("model", "meta-llama/Llama-2-7b-hf")

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        full_text = f"{context}\n{query}" if context else query

        if LITELLM_AVAILABLE:
            tok = int(token_counter(model=self.model, text=full_text))
        else:
            tok = max(1, len(full_text) // 4)

        device = "orin" if tok > self.threshold else "nano"
        conf = float(min(abs(tok - self.threshold) / max(self.threshold, 1), 1.0))

        return RoutingDecision(
            device=device,
            confidence=conf,
            method="token",
            reasoning=f"tokens={tok} threshold={self.threshold}",
            complexity_score=float(tok),
        )


# =============================================================================
# Semantic Strategy
# =============================================================================

class SemanticRouter(BaseRouter):
    """
    Trained semantic router using centroids.
    Falls back to token router on ambiguity or irrelevance.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for SemanticRouter")

        model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)
        self.label_path = config.get("semantic_label_path", "")
        self.margin_threshold = float(config.get("semantic_margin_threshold", 0.03))
        self.min_similarity = float(config.get("semantic_min_similarity", 0.15))
        self._token_fallback = TokenBasedRouter(config)
        self.nano_center, self.orin_center = self._load_and_build_centroids(self.label_path)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _load_and_build_centroids(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not label_path or not os.path.exists(label_path):
            simple_seed = [
                "Hello", "What is 2+2?", "Define machine learning", "What is the weather today?",
            ]
            complex_seed = [
                "Write a Python function to solve knapsack and explain time complexity",
                "Analyze the economic impact of climate policy with trade-offs",
                "Draft a detailed report with methodology and evaluation plan",
                "Explain quantum computing implications for cryptography in depth",
            ]
            return (
                np.mean(self.embedder.encode(simple_seed), axis=0),
                np.mean(self.embedder.encode(complex_seed), axis=0),
            )

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nano_texts, orin_texts = [], []
        for row in data:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text:
                continue
            if label == "nano":
                nano_texts.append(text)
            elif label == "orin":
                orin_texts.append(text)

        if len(nano_texts) < 3 or len(orin_texts) < 3:
            raise ValueError(
                f"Need >=3 samples per class. Got nano={len(nano_texts)} orin={len(orin_texts)}"
            )

        return (
            np.mean(self.embedder.encode(nano_texts), axis=0),
            np.mean(self.embedder.encode(orin_texts), axis=0),
        )

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        q_emb = self.embedder.encode([query])[0]
        sim_nano = self._cosine_similarity(q_emb, self.nano_center)
        sim_orin = self._cosine_similarity(q_emb, self.orin_center)

        if sim_nano < self.min_similarity and sim_orin < self.min_similarity:
            d = self._token_fallback.route(query, context)
            return RoutingDecision(
                device=d.device,
                confidence=d.confidence * 0.5,
                method="semantic_fallback_irrelevant",
                reasoning=f"low similarity (n={sim_nano:.2f}, o={sim_orin:.2f}) -> {d.reasoning}",
                complexity_score=float(sim_orin),
            )

        margin = abs(sim_orin - sim_nano)
        if margin < self.margin_threshold:
            d = self._token_fallback.route(query, context)
            return RoutingDecision(
                device=d.device,
                confidence=float(margin),
                method="semantic_fallback_ambiguous",
                reasoning=f"ambiguous margin={margin:.3f} (n={sim_nano:.2f}, o={sim_orin:.2f}) -> {d.reasoning}",
                complexity_score=float(sim_orin),
            )

        device = "orin" if sim_orin > sim_nano else "nano"
        return RoutingDecision(
            device=device,
            confidence=float(min(1.0, margin / 0.2)),
            method="semantic",
            reasoning=f"sim_nano={sim_nano:.3f} sim_orin={sim_orin:.3f} margin={margin:.3f}",
            complexity_score=float(sim_orin),
        )


# =============================================================================
# Heuristic Strategy
# =============================================================================

class HeuristicRouter(BaseRouter):
    """Conservative rule-based router with pre-compiled regex for performance."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.long_text_threshold = int(config.get("heuristic_long_chars", 250))
        self.multi_question_threshold = int(config.get("heuristic_multi_qmarks", 3))
        self.code_markers_needed = int(config.get("heuristic_code_markers_needed", 2))
        self.context_chars_threshold = int(config.get("heuristic_context_chars", 800))
        self._token_fallback = TokenBasedRouter(config)

        raw_complex_patterns = {
            "code_build_debug": [
                r"\b(write|implement|code|program|script|build|refactor|debug|fix)\b",
                r"\b(traceback|exception|error|segfault|timeout|hanging)\b",
                r"\b(api|flask|fastapi|docker|kubernetes|ssh|tunnel|nginx)\b",
                r"\b(system design|architecture|distributed|scalability|load balanc)\b",
            ],
            "math_cs_theory": [
                r"\b(prove|lemma|theorem|corollary|derivative|integral|gradient)\b",
                r"\b(time complexity|space complexity|big[- ]o)\b",
                r"\b(dynamic programming|dp|graph|dijkstra|bfs|dfs)\b",
                r"(?:\b|^)a\*(?:\s|$|\W)",
            ],
            "reasoning_comparison": [
                r"\b(compare|contrast|difference between|pros and cons|vs\.?|versus)\b",
                r"\b(evaluate|assess|critique|analyze)\b",
            ],
            "long_form_generation": [
                r"\b(essay|report|proposal|research paper|literature review|methodology)\b",
                r"\b(comprehensive|in-depth|step[- ]by[- ]step|detailed)\b",
                r"\b(summarize|synthesis)\b.*\b(everything|all|so far|entire)\b",
                r"\b(transcript|debate|dialogue|format as json|markdown table)\b",
            ],
            "data_engineering": [
                r"\b(etl|pipeline|spark|hadoop|presto|sql|csv|excel|dataframe|dataset)\b",
                r"\b(deduplicate|normalize|clean|transform|parse|extract)\b",
            ],
            "medical_analysis": [
                r"\b(symptom|diagnosis|treatment|therapy|prognosis|chronic|severe)\b",
                r"\b(pain|migraine|dizziness|fatigue|nausea|inflammation|anxiety|depression)\b",
                r"\b(dietary|meal|training|exercise|recovery)\b.*\b(plan|schedule|regimen)\b",
                r"\b(mental health|psycholog|counseling|therapist|physician|hospital)\b",
            ],
            "context_heavy": [
                r"\b(using (all|the) (context|history|above)|based on (the|our) (conversation|context))\b",
                r"\b(continue|expand|build on|follow up)\b.*\b(previous|earlier|above)\b",
            ],
        }

        raw_simple_patterns = {
            "greeting": [
                r"\b(hi|hello|hey|yo|sup)\b",
                r"\bgood (morning|afternoon|evening)\b",
                r"\b(thanks|thank you)\b",
            ],
            "general_knowledge": [
                r"\b(what is|who is|where is|when is|when did|how many|capital of)\b",
                r"\b(tell me a joke|fun fact|random fact)\b",
                r"\b(how to|how do i|can you tell me)\b",
            ],
            "wellness_tips": [
                r"\b(benefits? of|tips? for|advice on)\b",
                r"\b(daily intake|how often|how much)\b",
                r"\b(healthy|good)\b.*\b(habit|routine|lifestyle)\b",
            ],
            "short_definition": [
                r"\b(define|meaning of|definition of)\b",
                r"\bwhat does\b.*\bmean\b",
            ],
            "tiny_math": [
                r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*\??\s*$",
                r"^\s*what is\s+\d+\s*[\+\-\*/]\s*\d+\s*\??\s*$",
            ],
        }

        self.complex_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in raw_complex_patterns.items()
        }
        self.simple_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in raw_simple_patterns.items()
        }

        self._code_markers = [
            "```", "def ", "class ", "import ", "Traceback", "Exception", "ModuleNotFoundError",
            "SELECT ", "WITH ", "FROM ", "JOIN ", ";", "{", "}", "->", "::", "==", "!=",
        ]

    @staticmethod
    def _any_match(query_lower: str, patterns: list) -> bool:
        return any(pat.search(query_lower) for pat in patterns)

    def _match_category(self, query_lower: str, buckets: Dict[str, list]) -> Tuple[bool, Optional[str]]:
        for cat, pats in buckets.items():
            if self._any_match(query_lower, pats):
                return True, cat
        return False, None

    def _code_signal_count(self, query: str) -> int:
        return sum(1 for m in self._code_markers if m in query)

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        q = (query or "").strip()
        ql = q.lower()

        is_complex, cat = self._match_category(ql, self.complex_patterns)
        if is_complex:
            return RoutingDecision(device="orin", confidence=0.92, method="heuristic",
                                   reasoning=f"complex pattern={cat}")

        if len(q) >= self.long_text_threshold:
            return RoutingDecision(device="orin", confidence=0.80, method="heuristic",
                                   reasoning=f"long query chars={len(q)}")

        if q.count("?") >= self.multi_question_threshold:
            return RoutingDecision(device="orin", confidence=0.80, method="heuristic",
                                   reasoning=f"multi-question count={q.count('?')}")

        if self._code_signal_count(q) >= self.code_markers_needed:
            return RoutingDecision(device="orin", confidence=0.88, method="heuristic",
                                   reasoning="code/debug markers detected")

        if context and len(context) >= self.context_chars_threshold:
            return RoutingDecision(device="orin", confidence=0.75, method="heuristic",
                                   reasoning=f"large context chars={len(context)}")

        is_simple, scat = self._match_category(ql, self.simple_patterns)
        if is_simple:
            return RoutingDecision(device="nano", confidence=0.90, method="heuristic",
                                   reasoning=f"simple pattern={scat}")

        if len(ql.split()) <= 15 and len(q) <= 100:
            return RoutingDecision(device="nano", confidence=0.75, method="heuristic",
                                   reasoning="short everyday query")

        d = self._token_fallback.route(query, context)
        return RoutingDecision(
            device=d.device,
            confidence=float(d.confidence * 0.5),
            method="heuristic_fallback",
            reasoning=f"no heuristic match -> {d.reasoning}",
            complexity_score=d.complexity_score,
        )


# =============================================================================
# Hybrid Strategy
# =============================================================================

class HybridRouter(BaseRouter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weights = config.get("weights", {"token": 0.35, "semantic": 0.35, "heuristic": 0.30})
        self.routers = {
            "token": TokenBasedRouter(config),
            "semantic": SemanticRouter(config) if SENTENCE_TRANSFORMERS_AVAILABLE else None,
            "heuristic": HeuristicRouter(config),
        }

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        decisions: Dict[str, RoutingDecision] = {
            name: r.route(query, context)
            for name, r in self.routers.items()
            if r is not None
        }

        nano_score = orin_score = 0.0
        parts = []

        for name, d in decisions.items():
            w = float(self.weights.get(name, 0.0))
            vote = w * float(d.confidence)
            if d.device == "orin":
                orin_score += vote
            else:
                nano_score += vote
            parts.append(f"{name}:{d.device} conf={d.confidence:.2f} w={w:.2f}")

        if orin_score > nano_score:
            final, margin = "orin", orin_score - nano_score
        else:
            final, margin = "nano", nano_score - orin_score

        total = nano_score + orin_score
        conf = float(min(max(margin / total if total > 1e-12 else 0.5, 0.0), 1.0))

        return RoutingDecision(
            device=final,
            confidence=conf,
            method="hybrid",
            reasoning=f"nano_score={nano_score:.3f} orin_score={orin_score:.3f} | " + " | ".join(parts),
        )


# =============================================================================
# Perf Strategy
# =============================================================================

class PerformanceAwareRouter(BaseRouter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window = int(config.get("perf_window", 30))
        self.fail_penalty = float(config.get("perf_fail_penalty", 3000.0))
        self.stats: Dict[str, deque] = {
            "nano": deque(maxlen=self.window),
            "orin": deque(maxlen=self.window),
        }

    def update(self, device: str, latency_ms: float, tokens: int, ok: bool = True) -> None:
        if device in self.stats:
            self.stats[device].append((float(latency_ms), int(tokens), 1 if ok else 0))

    def _score(self, device: str) -> float:
        data = list(self.stats[device])
        if not data:
            return float("inf")
        total_lat = sum(x[0] for x in data)
        total_tok = sum(x[1] for x in data)
        ok_sum = sum(x[2] for x in data)
        fail_rate = 1.0 - (ok_sum / len(data))
        if total_tok == 0:
            return float(total_lat / len(data) + self.fail_penalty * fail_rate)
        return float(total_lat / total_tok + self.fail_penalty * fail_rate)

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        nano_s, orin_s = self._score("nano"), self._score("orin")
        if nano_s == float("inf") and orin_s == float("inf"):
            return RoutingDecision(device="nano", confidence=0.2, method="perf",
                                   reasoning="no perf stats yet -> default nano")
        device = "orin" if orin_s < nano_s else "nano"
        return RoutingDecision(
            device=device,
            confidence=0.70,
            method="perf",
            reasoning=f"scores nano={nano_s:.2f} orin={orin_s:.2f} -> {device}",
        )


# =============================================================================
# QueryRouter
# =============================================================================

class QueryRouter:
    """
    Strategy selector backed by QueryCache (from cache.py).

    Cache hit routing logic:
      1. Heavy context + cached nano prediction → hybrid re-route
         (long conversation makes a previously-simple query complex)
      2. Low prediction confidence (<threshold) → hybrid re-route
         (mixed routing history, let the router break the tie)
      3. High-confidence prediction, normal context → return predicted device
         (could be nano OR orin depending on routing history)
    """

    AVAILABLE_STRATEGIES = {
        "token": TokenBasedRouter,
        "semantic": SemanticRouter,
        "heuristic": HeuristicRouter,
        "hybrid": HybridRouter,
        "perf": PerformanceAwareRouter,
    }

    def __init__(self, strategy: str = "token", config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy={strategy}. Available={list(self.AVAILABLE_STRATEGIES)}")

        self.strategy_name = strategy
        self.router = self.AVAILABLE_STRATEGIES[strategy](self.config)

        self.cache_enabled = bool(self.config.get("cache_enabled", True))

        self._cache = QueryCache(
            max_size=int(self.config.get("cache_max_size", 500)),
            ttl_seconds=int(self.config.get("cache_ttl_seconds", 3600)),
            similarity_threshold=float(self.config.get("cache_similarity_threshold", 0.85)),
            use_semantic=bool(self.config.get("use_semantic_cache", True)),
            prediction_confidence_threshold=float(
                self.config.get("prediction_confidence_threshold", 0.70)
            ),
        )

        # Shared embedder — encoded once per query, reused for lookup + insert
        self.cache_embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.get("use_semantic_cache", True):
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self.cache_embedder = SentenceTransformer(model_name)

    @property
    def strategy(self) -> str:
        return self.strategy_name

    def _default_config(self) -> Dict[str, Any]:
        return {
            # ── Routing ───────────────────────────────────────────────────────
            # token_threshold=1000 matches Ng Mu Rong FYP benchmarks:
            # Nano degrades nonlinearly past 1000-1500 tokens.
            "token_threshold": 1000,
            "model": "meta-llama/Llama-2-7b-hf",
            "embedding_model": "all-MiniLM-L6-v2",
            "semantic_label_path": "src/tests/semantic_labels.json",
            "semantic_margin_threshold": 0.03,
            "semantic_min_similarity": 0.15,
            # heuristic_long_chars: 800 chars ≈ 200 tokens — clearly a long query
            "heuristic_long_chars": 800,
            "heuristic_multi_qmarks": 2,
            "heuristic_code_markers_needed": 2,
            # heuristic_context_chars: 3200 chars ≈ 800 tokens — Nano sweet spot
            # upper bound per Ng Mu Rong benchmarks (800-1200 token range).
            # Previously 800 chars (~200 tokens) which was routing to Orin far
            # too aggressively for context that Nano handles comfortably.
            "heuristic_context_chars": 3200,
            # Semantic gets the highest weight — it is the main FYP contribution.
            # Token is a reliable fallback. Heuristic is fast and domain-specific.
            "weights": {"token": 0.25, "semantic": 0.45, "heuristic": 0.30},
            # ── Cache ─────────────────────────────────────────────────────────
            # Set cache_enabled=False during benchmarking runs so routing
            # accuracy is measured cleanly against every query (no cache bypasses).
            # Enable for production deployment.
            "cache_enabled": False,
            "cache_ttl_seconds": 3600,
            "cache_max_size": 500,
            "cache_similarity_threshold": 0.85,
            "use_semantic_cache": True,
            "prediction_confidence_threshold": 0.70,
            # ── Perf ──────────────────────────────────────────────────────────
            "perf_window": 30,
            "perf_fail_penalty": 3000.0,
        }

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route_query(
        self,
        query: str,
        context: Optional[str] = None,
        context_key: Optional[str] = None,
    ) -> RoutingDecision:
        ctxk = context_key or "default"

        # Encode once — reused for both cache lookup and insert
        q_emb: Optional[np.ndarray] = None
        if self.cache_enabled and self.cache_embedder is not None:
            try:
                q_emb = self.cache_embedder.encode([query])[0]
            except Exception as exc:
                logger.warning("Cache embedding failed, continuing without: %s", exc)

        if self.cache_enabled:
            result = self._cache.lookup(query, ctxk, q_emb)
            if result is not None:
                context_len = len(context) if context else 0
                context_threshold = int(self.config.get("heuristic_context_chars", 800))

                # ── Re-route condition 1: heavy context overrides cached nano ──
                # The query may have been simple before, but a long conversation
                # now makes it complex — trust the router over the cache.
                context_override = (
                    context_len >= context_threshold
                    and result.predicted_device == "nano"
                )

                # ── Re-route condition 2: low prediction confidence ────────────
                # Routing history is mixed — hybrid router breaks the tie and
                # adds the new decision back to history to build confidence.
                low_confidence = result.use_hybrid_fallback

                if context_override or low_confidence:
                    reason = (
                        f"context_len={context_len}>={context_threshold} overrides cached nano"
                        if context_override
                        else f"low prediction confidence={result.predicted_confidence:.2f}"
                    )
                    logger.debug("Cache hit but re-routing via hybrid: %s", reason)

                    decision = self.router.route(query, context)
                    self._cache.insert(
                        query, ctxk,
                        device=decision.device,
                        confidence=decision.confidence,
                        method=decision.method,
                        q_emb=q_emb,
                    )
                    decision.reasoning = (
                        f"cache hit (hybrid re-route: {reason}) | " + decision.reasoning
                    )
                    decision.cache_hit = True
                    return decision

                # ── High-confidence prediction, normal context ─────────────────
                # Return the device predicted by routing history (nano or orin).
                age = int((datetime.now() - result.entry.timestamp).total_seconds())
                return RoutingDecision(
                    device=result.predicted_device,
                    confidence=result.predicted_confidence,
                    method=f"{self.strategy_name}_cached",
                    reasoning=(
                        f"cache hit age={age}s hits={result.entry.hit_count} "
                        f"predicted={result.predicted_device} "
                        f"conf={result.predicted_confidence:.2f} "
                        f"context_len={context_len} "
                        f"history={len(result.entry.routing_history)}"
                    ),
                    cache_hit=True,
                )

        # Cache miss — full routing, insert result into cache
        decision = self.router.route(query, context)

        if self.cache_enabled:
            self._cache.insert(
                query, ctxk,
                device=decision.device,
                confidence=decision.confidence,
                method=decision.method,
                q_emb=q_emb,
            )

        return decision

    # ------------------------------------------------------------------
    # Cache passthrough helpers
    # ------------------------------------------------------------------

    def warm_up_cache(self, pairs: List[Tuple[str, str, str]]) -> None:
        """Pre-populate the cache. pairs = [(query, context_key, device), ...]"""
        self._cache.warm_up(pairs, embedder=self.cache_embedder)

    def save_cache(self, path: str) -> None:
        """Persist cache to disk."""
        self._cache.save(path)

    def load_cache(self, path: str) -> int:
        """Restore cache from disk. Returns number of entries loaded."""
        return self._cache.load(path)

    def invalidate_cache(
        self,
        context_key: Optional[str] = None,
        query_pattern: Optional[str] = None,
    ) -> int:
        """Remove entries by context_key and/or query regex pattern."""
        return self._cache.invalidate(context_key=context_key, query_pattern=query_pattern)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache health metrics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Wipe the entire cache and reset stats."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Perf feedback + strategy switching
    # ------------------------------------------------------------------

    def update_perf(self, device: str, latency_ms: float, tokens: int, ok: bool = True) -> None:
        if hasattr(self.router, "update"):
            self.router.update(device=device, latency_ms=latency_ms, tokens=tokens, ok=ok)

    def change_strategy(self, strategy: str) -> None:
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy={strategy}")
        self.strategy_name = strategy
        self.router = self.AVAILABLE_STRATEGIES[strategy](self.config)


# =============================================================================
# Smoke test
# =============================================================================

# =============================================================================
# Shared configs — import these in router.py, chatbot_tester.py, etc.
# =============================================================================

# Use during benchmarking: cache off so every query hits the router and
# routing accuracy is measured cleanly with no cache bypasses.
BENCHMARK_CFG = {
    "token_threshold": 1000,
    "model": "meta-llama/Llama-2-7b-hf",
    "embedding_model": "all-MiniLM-L6-v2",
    "semantic_label_path": "src/tests/semantic_labels.json",
    "semantic_margin_threshold": 0.03,
    "semantic_min_similarity": 0.15,
    "heuristic_long_chars": 800,         # ~200 tokens
    "heuristic_multi_qmarks": 2,
    "heuristic_code_markers_needed": 2,
    "heuristic_context_chars": 3200,     # ~800 tokens — Nano sweet spot upper bound
    "weights": {"token": 0.25, "semantic": 0.45, "heuristic": 0.30},
    "cache_enabled": False,              # OFF — clean routing accuracy measurement
    "perf_window": 30,
    "perf_fail_penalty": 3000.0,
}

# Use in production: cache on, responses served from history on repeat queries.
PRODUCTION_CFG = {
    **BENCHMARK_CFG,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "cache_max_size": 500,
    "cache_similarity_threshold": 0.85,
    "use_semantic_cache": True,
    "prediction_confidence_threshold": 0.70,
    "enable_response_cache": True,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = BENCHMARK_CFG

    qr = QueryRouter(strategy="hybrid", config=cfg)

    qr.warm_up_cache([
        ("hello", "demo", "nano"),
        ("what is 2+2", "demo", "nano"),
    ])

    tests = [
        "hello",
        "what is 2+2",
        "Explain quantum computing and its implications for cryptography",
    ]

    for t in tests:
        d = qr.route_query(t, context_key="demo")
        print(f"{t!r:55} => {d.device:4}  [{d.method}]  {d.reasoning}")

    print("\nCache stats:", qr.get_cache_stats())

    print("\n--- Second pass (predictive routing from history) ---")
    for t in tests:
        d = qr.route_query(t, context_key="demo")
        print(f"{t!r:55} => {d.device:4}  [{d.method}]  cache_hit={d.cache_hit}")

    qr.save_cache("/tmp/qr_cache.json")
    print("\nCache saved to /tmp/qr_cache.json")