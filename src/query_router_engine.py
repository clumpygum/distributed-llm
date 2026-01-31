"""
query_routing_engine.py

Dynamic Query Router for Distributed Edge Chatbots on NVIDIA Jetson Platforms
- Multiple routing strategies: token / semantic (trained centroids) / heuristic / hybrid / perf
- Optional query cache (exact + semantic similarity within same context_key)
- Designed to plug into your existing Router wrapper (route_query + update_perf)

Key design changes vs your draft:
1) SemanticRouter now supports TRAINED centroids from a label file:
      [{"text": "...", "label": "nano"|"orin"}, ...]
   -> routes by closest centroid + optional margin fallback to token.
2) HeuristicRouter expanded and made more tunable + safer fallbacks.
3) HybridRouter uses "vote + weighted confidence", with a default “confidence-as-margin”
   normalization so it doesn’t always collapse to one device.
"""

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

# Optional deps
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic routing disabled.")

try:
    from litellm import token_counter
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not available. Token-based routing will use approximation.")


# ----------------------------
# Data Models
# ----------------------------

@dataclass
class RoutingDecision:
    device: Literal["nano", "orin"]
    confidence: float
    method: str
    reasoning: str
    complexity_score: Optional[float] = None
    cache_hit: bool = False


@dataclass
class CachedQuery:
    query: str
    query_hash: str
    context_key: str
    embedding: Optional[np.ndarray]
    timestamp: datetime
    device_used: str
    response_time: Optional[float] = None


# ----------------------------
# Base Strategy
# ----------------------------

class BaseRouter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        raise NotImplementedError


# ----------------------------
# Token Strategy
# ----------------------------

class TokenBasedRouter(BaseRouter):
    """
    Token-counting router.
    device = ORIN if tokens > threshold else NANO
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = int(config.get("token_threshold", 1000))
        self.model = config.get("model", "meta-llama/Llama-2-7b-hf")

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        full_text = f"{context}\n{query}" if context else query

        if LITELLM_AVAILABLE:
            tok = int(token_counter(model=self.model, text=full_text))
        else:
            # rough approx: ~4 chars/token
            tok = max(1, len(full_text) // 4)

        device = "orin" if tok > self.threshold else "nano"

        # confidence: distance from threshold scaled into [0,1]
        conf = abs(tok - self.threshold) / max(self.threshold, 1)
        conf = float(min(conf, 1.0))

        return RoutingDecision(
            device=device,
            confidence=conf,
            method="token",
            reasoning=f"tokens={tok} threshold={self.threshold}",
            complexity_score=float(tok),
        )


# ----------------------------
# Semantic Strategy (trained centroids)
# ----------------------------

class SemanticRouter(BaseRouter):
    """
    Trained semantic router using centroids:
      - load labeled data from config["semantic_label_path"]
      - compute nano_center and orin_center
      - route by higher cosine similarity
      - optional margin fallback to token if ambiguous

    Label file format (JSON):
      [
        {"text": "...", "label": "nano"},
        {"text": "...", "label": "orin"}
      ]
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for SemanticRouter")

        model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)

        self.label_path = config.get("semantic_label_path", "")
        self.margin_threshold = float(config.get("semantic_margin_threshold", 0.03))
        self._token_fallback = TokenBasedRouter(config)

        self.nano_center, self.orin_center = self._load_and_build_centroids(self.label_path)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def _load_and_build_centroids(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # If no label file, fall back to a tiny seed set (still works, but not recommended).
        if not label_path or not os.path.exists(label_path):
            simple_seed = [
                "Hello",
                "What is 2+2?",
                "Define machine learning",
                "What is the weather today?",
            ]
            complex_seed = [
                "Write a Python function to solve knapsack and explain time complexity",
                "Analyze the economic impact of climate policy with trade-offs",
                "Draft a detailed report with methodology and evaluation plan",
                "Explain quantum computing implications for cryptography in depth",
            ]
            nano_center = np.mean(self.embedder.encode(simple_seed), axis=0)
            orin_center = np.mean(self.embedder.encode(complex_seed), axis=0)
            return nano_center, orin_center

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
                f"Need >=3 samples per class for semantic centroids. "
                f"Got nano={len(nano_texts)} orin={len(orin_texts)}"
            )

        nano_center = np.mean(self.embedder.encode(nano_texts), axis=0)
        orin_center = np.mean(self.embedder.encode(orin_texts), axis=0)
        return nano_center, orin_center

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        q_emb = self.embedder.encode([query])[0]
        sim_nano = self._cosine_similarity(q_emb, self.nano_center)
        sim_orin = self._cosine_similarity(q_emb, self.orin_center)

        margin = abs(sim_orin - sim_nano)

        # ambiguous => token fallback
        if margin < self.margin_threshold:
            d = self._token_fallback.route(query, context)
            return RoutingDecision(
                device=d.device,
                confidence=float(margin),
                method="semantic_fallback_token",
                reasoning=f"ambiguous semantic nano={sim_nano:.3f} orin={sim_orin:.3f} margin={margin:.3f} -> {d.reasoning}",
                complexity_score=float(sim_orin),
            )

        device = "orin" if sim_orin > sim_nano else "nano"

        # confidence: normalize margin into [0,1] (0.2 is a typical “strong separation”)
        conf = float(min(1.0, margin / 0.2))

        return RoutingDecision(
            device=device,
            confidence=conf,
            method="semantic",
            reasoning=f"sim_nano={sim_nano:.3f} sim_orin={sim_orin:.3f} margin={margin:.3f}",
            complexity_score=float(sim_orin),
        )


# ----------------------------
# Heuristic Strategy (more comprehensive)
# ----------------------------

class HeuristicRouter(BaseRouter):
    """
    Conservative rule-based router:
      - strong complex signal => ORIN
      - strong simple signal => NANO
      - else fallback to token

    Tunables:
      heuristic_long_chars (default 220)
      heuristic_multi_qmarks (default 2)
      heuristic_code_markers_needed (default 2)
      heuristic_context_chars (default 800)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.long_text_threshold = int(config.get("heuristic_long_chars", 220))
        self.multi_question_threshold = int(config.get("heuristic_multi_qmarks", 2))
        self.code_markers_needed = int(config.get("heuristic_code_markers_needed", 2))
        self.context_chars_threshold = int(config.get("heuristic_context_chars", 800))

        self._token_fallback = TokenBasedRouter(config)

        # IMPORTANT: keep patterns fairly broad but not over-trigger happy
        self.complex_patterns = {
            "code_build_debug": [
                r"\b(write|implement|code|program|script|build|refactor)\b",
                r"\b(debug|fix|traceback|exception|error|segfault|timeout|hanging)\b",
                r"\b(api|flask|fastapi|docker|kubernetes|ssh|tunnel|port-forward|nginx)\b",
                r"\b(system design|architecture|distributed|scalability|load balanc)\b",
            ],
            "math_cs_theory": [
                r"\b(prove|lemma|theorem|corollary)\b",
                r"\b(derivative|integral|gradient|jacobian)\b",
                r"\b(time complexity|space complexity|big[- ]o)\b",
                r"\b(dynamic programming|dp|graph|dijkstra|a\*|bfs|dfs)\b",
            ],
            "long_form_generation": [
                r"\b(essay|report|proposal|research paper|literature review|methodology)\b",
                r"\b(comprehensive|in-depth|step by step|detailed)\b",
                r"\b(summarize|synthesis)\b.*\b(everything|all|so far|entire)\b",
                r"\b(transcript|debate|dialogue)\b",
            ],
            "data_engineering": [
                r"\b(etl|pipeline|spark|hadoop|presto|sql)\b",
                r"\b(csv|excel|dataframe|dataset)\b",
                r"\b(deduplicate|normalize|clean|transform|parse|extract)\b",
            ],
            "context_heavy": [
                r"\b(using (all|the) (context|history|above)|based on (the|our) (conversation|context))\b",
                r"\b(continue|expand|build on|follow up)\b.*\b(previous|earlier|above)\b",
            ],
        }

        self.simple_patterns = {
            "greeting": [
                r"^(hi|hello|hey|yo|sup)\b",
                r"^good (morning|afternoon|evening)\b",
                r"^(thanks|thank you)\b",
            ],
            "short_definition": [
                r"^define\b",
                r"^what does .+ mean\??$",
            ],
            "short_wh": [
                r"^(what|who|when|where)\b.*\?$",
            ],
            "tiny_math": [
                r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*\??\s*$",
                r"^\s*what is\s+\d+\s*[\+\-\*/]\s*\d+\s*\??\s*$",
            ],
        }

        self._code_markers = [
            "```", "def ", "class ", "import ", "Traceback", "Exception", "ModuleNotFoundError",
            "SELECT ", "WITH ", "FROM ", "JOIN ", ";", "{", "}", "->", "::", "==", "!=",
        ]

    @staticmethod
    def _any_match(query_lower: str, patterns: list) -> bool:
        for pat in patterns:
            if re.search(pat, query_lower):
                return True
        return False

    def _match_category(self, query_lower: str, buckets: Dict[str, list]) -> Tuple[bool, Optional[str]]:
        for cat, pats in buckets.items():
            if self._any_match(query_lower, pats):
                return True, cat
        return False, None

    def _code_signal_count(self, query: str) -> int:
        cnt = 0
        for m in self._code_markers:
            if m in query:
                cnt += 1
        return cnt

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        q = (query or "").strip()
        ql = q.lower()

        # 1) strong complex patterns
        is_complex, cat = self._match_category(ql, self.complex_patterns)
        if is_complex:
            return RoutingDecision(
                device="orin",
                confidence=0.92,
                method="heuristic",
                reasoning=f"complex pattern={cat}",
            )

        # 2) medium complex signals
        if len(q) >= self.long_text_threshold:
            return RoutingDecision(
                device="orin",
                confidence=0.80,
                method="heuristic",
                reasoning=f"long query chars={len(q)}",
            )

        if q.count("?") >= self.multi_question_threshold:
            return RoutingDecision(
                device="orin",
                confidence=0.80,
                method="heuristic",
                reasoning=f"multi-question count={q.count('?')}",
            )

        if self._code_signal_count(q) >= self.code_markers_needed:
            return RoutingDecision(
                device="orin",
                confidence=0.88,
                method="heuristic",
                reasoning="code/debug markers detected",
            )

        if context and len(context) >= self.context_chars_threshold:
            return RoutingDecision(
                device="orin",
                confidence=0.75,
                method="heuristic",
                reasoning=f"large context chars={len(context)}",
            )

        # 3) strong simple patterns
        is_simple, scat = self._match_category(ql, self.simple_patterns)
        if is_simple:
            return RoutingDecision(
                device="nano",
                confidence=0.90,
                method="heuristic",
                reasoning=f"simple pattern={scat}",
            )

        # 4) very short and harmless => nano
        if len(ql.split()) <= 6 and len(q) <= 40:
            return RoutingDecision(
                device="nano",
                confidence=0.70,
                method="heuristic",
                reasoning="very short query",
            )

        # 5) fallback
        d = self._token_fallback.route(query, context)
        return RoutingDecision(
            device=d.device,
            confidence=float(d.confidence * 0.5),
            method="heuristic_fallback",
            reasoning=f"no heuristic match -> {d.reasoning}",
            complexity_score=d.complexity_score,
        )


# ----------------------------
# Hybrid Strategy
# ----------------------------

class HybridRouter(BaseRouter):
    """
    Weighted ensemble:
      - token / semantic / heuristic each produce (device, confidence)
      - we convert to weighted votes (confidence * weight)
      - device with higher vote wins

    Important: semantic can be missing (no sentence-transformers) => skipped
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weights = config.get("weights", {"token": 0.35, "semantic": 0.35, "heuristic": 0.30})

        self.routers = {
            "token": TokenBasedRouter(config),
            "semantic": SemanticRouter(config) if SENTENCE_TRANSFORMERS_AVAILABLE else None,
            "heuristic": HeuristicRouter(config),
        }

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        decisions: Dict[str, RoutingDecision] = {}
        for name, r in self.routers.items():
            if r is not None:
                decisions[name] = r.route(query, context)

        nano_score = 0.0
        orin_score = 0.0
        parts = []

        for name, d in decisions.items():
            w = float(self.weights.get(name, 0.0))
            vote = w * float(d.confidence)

            if d.device == "orin":
                orin_score += vote
            else:
                nano_score += vote

            parts.append(f"{name}:{d.device} conf={d.confidence:.2f} w={w:.2f}")

        # tie-break: prefer nano on ties (energy saving), but you can flip this.
        if orin_score > nano_score:
            final = "orin"
            margin = orin_score - nano_score
        else:
            final = "nano"
            margin = nano_score - orin_score

        total = nano_score + orin_score
        conf = (margin / total) if total > 1e-12 else 0.5
        conf = float(min(max(conf, 0.0), 1.0))

        return RoutingDecision(
            device=final,
            confidence=conf,
            method="hybrid",
            reasoning=f"nano_score={nano_score:.3f} orin_score={orin_score:.3f} | " + " | ".join(parts),
        )


# ----------------------------
# Perf Strategy
# ----------------------------

class PerformanceAwareRouter(BaseRouter):
    """
    Online router:
      score(device) = latency_per_token + fail_penalty * fail_rate
      choose lower score

    You must call QueryRouter.update_perf(...) after each request.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window = int(config.get("perf_window", 30))
        self.fail_penalty = float(config.get("perf_fail_penalty", 3000.0))
        self.stats = {
            "nano": deque(maxlen=self.window),  # (lat_ms, tok, ok_flag)
            "orin": deque(maxlen=self.window),
        }

    def update(self, device: str, latency_ms: float, tokens: int, ok: bool = True) -> None:
        if device in self.stats and tokens and tokens > 0:
            self.stats[device].append((float(latency_ms), int(tokens), 1 if ok else 0))

    def _score(self, device: str) -> float:
        data = list(self.stats[device])
        if not data:
            return float("inf")

        total_lat = sum(x[0] for x in data)
        total_tok = sum(x[1] for x in data) or 1
        ok_sum = sum(x[2] for x in data)
        fail_rate = 1.0 - (ok_sum / len(data))

        lat_per_tok = total_lat / total_tok
        return float(lat_per_tok + self.fail_penalty * fail_rate)

    def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        nano_s = self._score("nano")
        orin_s = self._score("orin")

        if nano_s == float("inf") and orin_s == float("inf"):
            return RoutingDecision(
                device="nano",
                confidence=0.2,
                method="perf",
                reasoning="no perf stats yet -> default nano",
            )

        device = "orin" if orin_s < nano_s else "nano"
        return RoutingDecision(
            device=device,
            confidence=0.70,
            method="perf",
            reasoning=f"scores nano={nano_s:.2f} orin={orin_s:.2f} -> {device}",
        )


# ----------------------------
# QueryRouter (strategy selector + cache)
# ----------------------------

class QueryRouter:
    """
    Strategy selector + optional cache.
    Exposes:
      - route_query(query, context, context_key) -> RoutingDecision
      - update_perf(device, latency_ms, tokens, ok)
      - change_strategy(strategy)
      - clear_cache()
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
            raise ValueError(f"Unknown strategy={strategy}. Available={list(self.AVAILABLE_STRATEGIES.keys())}")

        self.strategy_name = strategy
        self.router = self.AVAILABLE_STRATEGIES[strategy](self.config)

        # cache config
        self.cache_enabled = bool(self.config.get("cache_enabled", True))
        self.cache_ttl = int(self.config.get("cache_ttl_seconds", 300))
        self.cache_max_size = int(self.config.get("cache_max_size", 100))
        self.similarity_threshold = float(self.config.get("cache_similarity_threshold", 0.85))
        self.cache_force_nano = bool(self.config.get("cache_force_nano", True))
        self.use_semantic_cache = bool(self.config.get("use_semantic_cache", True))

        self.query_cache: deque = deque(maxlen=self.cache_max_size)

        # embedder for semantic cache
        self.cache_embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.use_semantic_cache:
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self.cache_embedder = SentenceTransformer(model_name)

    @property
    def strategy(self) -> str:
        # compatibility with older Router wrapper
        return self.strategy_name

    def _default_config(self) -> Dict[str, Any]:
        return {
            # token
            "token_threshold": 1000,
            "model": "meta-llama/Llama-2-7b-hf",

            # semantic
            "embedding_model": "all-MiniLM-L6-v2",
            "semantic_label_path": "src/tests/semantic_labels.json",
            "semantic_margin_threshold": 0.03,

            # heuristic tuning
            "heuristic_long_chars": 220,
            "heuristic_multi_qmarks": 2,
            "heuristic_code_markers_needed": 2,
            "heuristic_context_chars": 800,

            # hybrid weights
            "weights": {"token": 0.35, "semantic": 0.35, "heuristic": 0.30},

            # cache
            "cache_enabled": True,
            "cache_ttl_seconds": 300,
            "cache_max_size": 100,
            "cache_similarity_threshold": 0.85,
            "use_semantic_cache": True,
            "cache_force_nano": True,

            # perf
            "perf_window": 30,
            "perf_fail_penalty": 3000.0,
        }

    def _hash_query(self, query: str, context_key: str) -> str:
        key = f"{context_key}||{query.lower().strip()}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _is_cache_valid(self, cq: CachedQuery) -> bool:
        return (datetime.now() - cq.timestamp).total_seconds() <= self.cache_ttl

    def _clean_expired_cache(self) -> None:
        if not self.cache_enabled:
            return
        valid = [x for x in self.query_cache if self._is_cache_valid(x)]
        self.query_cache.clear()
        self.query_cache.extend(valid)

    def _find_cached(self, query: str, context_key: str) -> Optional[CachedQuery]:
        if not self.cache_enabled or not self.query_cache:
            return None

        qh = self._hash_query(query, context_key)

        # exact match first
        for cached in self.query_cache:
            if cached.context_key != context_key:
                continue
            if not self._is_cache_valid(cached):
                continue
            if cached.query_hash == qh:
                return cached

        # semantic match next
        if self.cache_embedder is None:
            return None

        q_emb = self.cache_embedder.encode([query])[0]
        best = None
        best_sim = 0.0

        for cached in self.query_cache:
            if cached.context_key != context_key:
                continue
            if not self._is_cache_valid(cached):
                continue
            if cached.embedding is None:
                continue

            denom = (np.linalg.norm(q_emb) * np.linalg.norm(cached.embedding)) + 1e-12
            sim = float(np.dot(q_emb, cached.embedding) / denom)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best = cached

        return best

    def _add_cache(self, query: str, context_key: str, decision: RoutingDecision) -> None:
        if not self.cache_enabled:
            return

        emb = self.cache_embedder.encode([query])[0] if self.cache_embedder is not None else None
        self.query_cache.append(CachedQuery(
            query=query,
            query_hash=self._hash_query(query, context_key),
            context_key=context_key,
            embedding=emb,
            timestamp=datetime.now(),
            device_used=decision.device,
            response_time=None,
        ))

    def route_query(self, query: str, context: Optional[str] = None, context_key: Optional[str] = None) -> RoutingDecision:
        """
        query: current user query
        context: optional context string (can be your concatenated conversation history)
        context_key: stable short key for a conversation (so cache won't leak between convos)
        """
        ctxk = context_key or "default"
        self._clean_expired_cache()

        cached = self._find_cached(query, ctxk)
        if cached is not None and self.cache_force_nano:
            return RoutingDecision(
                device="nano",
                confidence=1.0,
                method=f"{self.strategy_name}_cached",
                reasoning=f"cache hit age={(datetime.now() - cached.timestamp).seconds}s original={cached.device_used} -> nano",
                cache_hit=True,
            )

        d = self.router.route(query, context)
        self._add_cache(query, ctxk, d)
        return d

    def update_perf(self, device: str, latency_ms: float, tokens: int, ok: bool = True) -> None:
        # only affects perf router
        if hasattr(self.router, "update"):
            self.router.update(device=device, latency_ms=latency_ms, tokens=tokens, ok=ok)

    def clear_cache(self) -> None:
        self.query_cache.clear()

    def change_strategy(self, strategy: str) -> None:
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy={strategy}")
        self.strategy_name = strategy
        self.router = self.AVAILABLE_STRATEGIES[strategy](self.config)


# ----------------------------
# Quick demo
# ----------------------------
if __name__ == "__main__":
    cfg = {
        "token_threshold": 1200,
        "cache_enabled": True,
        "use_semantic_cache": True,
        "cache_force_nano": True,
        "semantic_label_path": "src/tests/semantic_labels.json",
        "semantic_margin_threshold": 0.03,
        "weights": {"token": 0.35, "semantic": 0.35, "heuristic": 0.30},
    }

    qr = QueryRouter(strategy="hybrid", config=cfg)
    tests = [
        "hello",
        "what is 2+2",
        "Explain quantum computing and its implications for cryptography",
        "Write a Python function to implement quicksort and explain time complexity",
        "Summarize everything we've discussed so far in one comprehensive report.",
    ]

    for t in tests:
        d = qr.route_query(t, context_key="demo")
        print(t, "=>", d.device, d.method, d.reasoning)