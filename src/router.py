import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from token_counter import TokenCounter
from query_router_engine import QueryRouter  # fixed: was query_router_engine
from models.orin import Orin
from models.nano import Nano


class Router:
    def __init__(self, strategy: str = "token", config: Optional[Dict[str, Any]] = None, threshold_fallback: int = 100):
        """
        strategy:           "token" | "semantic" | "heuristic" | "hybrid" | "perf"
        config:             dict for QueryRouter + caching knobs
        threshold_fallback: token count used only if QueryRouter itself raises
        """
        self.token_counter = TokenCounter()
        self.orin = Orin()
        self.nano = Nano()

        self.threshold_fallback = threshold_fallback
        self.config = config or {}
        self.query_router = QueryRouter(strategy=strategy, config=self.config)

        # Response cache — delegate to QueryRouter's built-in QueryCache
        # so we don't maintain a second LRU store.
        self.enable_response_cache = bool(self.config.get("enable_response_cache", False))

        # Context hashing for cache key correctness (last-k turns)
        self.cache_last_k = int(self.config.get("cache_last_k", 6))

        # Optional failover if the chosen device returns an error
        self.enable_failover = bool(self.config.get("enable_failover", True))

    # ------------------------------------------------------------------
    # Back-compat API
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: int) -> None:
        """Keep API compatible with the original Router."""
        self.threshold_fallback = threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text(self, response: Any) -> Optional[str]:
        """Normalise any device response shape into a plain string."""
        if response is None:
            return None

        if isinstance(response, str):
            return response.strip() or None

        if isinstance(response, dict):
            # Common happy-path keys
            for key in ("response", "content", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                # nested: {"message": {"content": "..."}}
                if isinstance(val, dict):
                    inner = val.get("content")
                    if isinstance(inner, str) and inner.strip():
                        return inner.strip()

            # Error shape — surface a short description rather than returning None
            if "error" in response:
                parts = [
                    str(response.get("error", "")).strip(),
                    str(response.get("detail", "")).strip(),
                    str(response.get("body", "")).strip(),
                ]
                combined = " ".join(p for p in parts if p)
                return combined[:300] if combined else None

        return None

    def _history_to_query_and_context(
        self, conversation_history: List[Dict]
    ) -> Tuple[str, Optional[str], str]:
        """
        Split conversation_history into:
          query    — latest user message text
          context  — everything before that message as a formatted string
          ctx_hash — SHA-256 prefix of last-k turns (for cache keying)
        """
        if not conversation_history:
            return "", None, "nohist"

        # Locate last user turn
        last_user_idx: Optional[int] = None
        for i in range(len(conversation_history) - 1, -1, -1):
            m = conversation_history[i]
            if isinstance(m, dict) and m.get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            query = ""
            ctx_msgs = conversation_history
        else:
            query = (conversation_history[last_user_idx].get("content") or "").strip()
            ctx_msgs = conversation_history[:last_user_idx]

        # Build context string
        context_lines = [
            f"{(m.get('role') or '').strip()}: {(m.get('content') or '').strip()}"
            for m in ctx_msgs
            if isinstance(m, dict) and (m.get("content") or "").strip()
        ]
        context = "\n".join(context_lines) if context_lines else None

        # Stable cache key from last-k turns
        compact = [
            f"{m.get('role', '')}:{(m.get('content') or '').strip()}"
            for m in conversation_history[-self.cache_last_k:]
            if isinstance(m, dict)
        ]
        ctx_hash = hashlib.sha256("\n".join(compact).encode("utf-8")).hexdigest()[:16]

        return query, context, ctx_hash

    def _is_error(self, raw: Any) -> bool:
        return isinstance(raw, dict) and "error" in raw

    def _run_device(
        self, device: str, conversation_history: List[Dict]
    ) -> Tuple[Any, str, float]:
        """
        Run inference on the chosen device.
        Returns (raw_response, device_name, latency_ms).
        """
        start = datetime.now()

        if device == "orin":
            print("Processing query on Orin")
            raw = self.orin.process(conversation_history)
            which = "orin"
        else:
            print("Processing query on Nano")
            raw = self.nano.process(conversation_history)
            which = "nano"

        lat_ms = (datetime.now() - start).total_seconds() * 1000.0
        return raw, which, lat_ms

    # ------------------------------------------------------------------
    # Response cache helpers (backed by QueryRouter's QueryCache)
    # ------------------------------------------------------------------

    def _response_cache_key(self, ctx_hash: str, query: str) -> str:
        """Unique key combining routing strategy, context hash, and query."""
        return f"{self.query_router.strategy}|{ctx_hash}|{query.lower().strip()}"

    def _get_response_cache(self, key: str) -> Optional[Dict]:
        """Retrieve from QueryRouter's cache store."""
        if not self.enable_response_cache:
            return None
        entry = self.query_router._cache.lookup(key, context_key="response_cache")
        if entry is None:
            return None
        # We stash the payload JSON-encoded in entry.query field (small hack —
        # no extra store needed).
        try:
            import json
            return json.loads(entry.query)
        except Exception:
            return None

    def _set_response_cache(self, key: str, payload: Dict) -> None:
        """Store in QueryRouter's cache store."""
        if not self.enable_response_cache:
            return
        try:
            import json
            self.query_router._cache.insert(
                query=json.dumps(payload),   # payload serialised as the "query"
                context_key="response_cache",
                device="nano",  # cache hits always route to nano
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_query(self, conversation_history: List[Dict]) -> Tuple[Dict, int, str]:
        """
        Main entry point.  Returns:
          (response_dict, response_tokens, device_string)

        response_dict keys:
          response, raw, cache_hit, routing_overhead_ms,
          routing_method, routing_confidence, routing_reasoning, ok
        """
        query, context, ctx_hash = self._history_to_query_and_context(conversation_history)

        # ── 0) Response cache check ────────────────────────────────────
        if self.enable_response_cache:
            cache_key = self._response_cache_key(ctx_hash, query)
            cached = self._get_response_cache(cache_key)
            if cached is not None:
                text = cached.get("text", "")
                raw = cached.get("raw")
                # Cache hits ALWAYS route to nano.
                response_tokens = self.token_counter.count_tokens(
                    {"role": "assistant", "content": text}
                )
                return {
                    "response": text,
                    "raw": raw,
                    "cache_hit": True,
                    "routing_method": "response_cache",
                    "routing_confidence": 1.0,
                    "routing_reasoning": "response cache hit -> nano",
                    "routing_overhead_ms": 0.0,
                    "ok": True,
                }, response_tokens, "nano"

        # ── 1) Routing decision ────────────────────────────────────────
        t0 = time.time()
        routing_method = "unknown"
        routing_confidence = 0.0
        routing_reasoning = ""
        device = "nano"  # safe default

        try:
            decision = self.query_router.route_query(
                query=query, context=context, context_key=ctx_hash
            )
            device = decision.device
            routing_method = decision.method
            routing_confidence = float(decision.confidence)
            routing_reasoning = decision.reasoning
            print(
                f"Routing decision: {device.upper()} | method={decision.method} "
                f"| conf={decision.confidence:.3f}"
            )
            print(f"Reason: {decision.reasoning}")
        except Exception as exc:
            ctx_size = self.token_counter.get_context_size(conversation_history)
            device = "orin" if ctx_size > self.threshold_fallback else "nano"
            routing_method = "fallback_ctx_size"
            routing_confidence = 0.2
            routing_reasoning = (
                f"router failed: {exc}; ctx_size={ctx_size}, "
                f"threshold_fallback={self.threshold_fallback}"
            )
            print(
                f"Routing decision failed ({exc}). "
                f"Falling back to ctx_size={ctx_size} -> {device.upper()}"
            )

        routing_overhead_ms = (time.time() - t0) * 1000.0

        # ── 2) Run device (+ optional failover) ───────────────────────
        raw, which, lat_ms = self._run_device(device, conversation_history)

        if self.enable_failover and self._is_error(raw):
            other = "orin" if which == "nano" else "nano"
            print(f"Primary device {which.upper()} failed — failing over to {other.upper()}")
            raw2, which2, lat2 = self._run_device(other, conversation_history)
            if not self._is_error(raw2):
                raw, which, lat_ms = raw2, which2, lat2

        # ── 3) Normalise output + token count ─────────────────────────
        text = self._extract_text(raw) or "No response available"
        response_tokens = self.token_counter.count_tokens(
            {"role": "assistant", "content": text}
        )
        ok = not self._is_error(raw)

        # ── 4) Feed latency back to perf-aware router ─────────────────
        try:
            self.query_router.update_perf(which, lat_ms, response_tokens, ok=ok)
        except Exception:
            pass

        # ── 5) Store in response cache ─────────────────────────────────
        if self.enable_response_cache:
            self._set_response_cache(
                self._response_cache_key(ctx_hash, query),
                {"text": text, "raw": raw, "device": which},
            )

        return {
            "response": text,
            "raw": raw,
            "cache_hit": False,
            "routing_overhead_ms": round(routing_overhead_ms, 2),
            "routing_method": routing_method,
            "routing_confidence": round(routing_confidence, 4),
            "routing_reasoning": routing_reasoning,
            "ok": ok,
        }, response_tokens, which