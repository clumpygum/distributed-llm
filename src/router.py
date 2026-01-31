import hashlib
import time
from datetime import datetime

from token_counter import TokenCounter
from query_router_engine import QueryRouter
from models.orin import Orin
from models.nano import Nano


class _LRUCacheTTL:
    # Simple LRU + TTL cache for responses (no external deps)
    def __init__(self, maxsize=300, ttl_s=6 * 3600):
        from collections import OrderedDict
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self.store = OrderedDict()

    def get(self, key):
        now = time.time()
        if key not in self.store:
            return None
        val, exp = self.store[key]
        if exp < now:
            self.store.pop(key, None)
            return None
        self.store.move_to_end(key)
        return val

    def set(self, key, val):
        now = time.time()
        exp = now + self.ttl_s
        if key in self.store:
            self.store.pop(key, None)
        self.store[key] = (val, exp)
        self.store.move_to_end(key)
        while len(self.store) > self.maxsize:
            self.store.popitem(last=False)


class Router:
    def __init__(self, strategy="token", config=None, threshold_fallback=100):
        """
        strategy: "token" | "semantic" | "heuristic" | "hybrid" | "perf"
        config: dict for QueryRouter + caching knobs
        threshold_fallback: used only if QueryRouter fails
        """
        self.token_counter = TokenCounter()
        self.orin = Orin()
        self.nano = Nano()

        self.threshold_fallback = threshold_fallback
        self.config = config or {}
        self.query_router = QueryRouter(strategy=strategy, config=self.config)

        # Response cache (optional)
        self.enable_response_cache = bool(self.config.get("enable_response_cache", False))
        if self.enable_response_cache:
            self.response_cache = _LRUCacheTTL(
                maxsize=int(self.config.get("response_cache_size", 300)),
                ttl_s=int(self.config.get("response_cache_ttl_s", 6 * 3600)),
            )
        else:
            self.response_cache = None

        # Optional failover if chosen device fails
        self.enable_failover = bool(self.config.get("enable_failover", True))

        # Context hashing for cache correctness (last-k turns)
        self.cache_last_k = int(self.config.get("cache_last_k", 6))

    def set_threshold(self, threshold):
        # keep API compatible with old Router
        self.threshold_fallback = threshold

    def _extract_text(self, response):
        if response is None:
            return None

        if isinstance(response, str):
            return response.strip() or None

        if isinstance(response, dict):
            if isinstance(response.get("response"), str) and response["response"].strip():
                return response["response"].strip()

            msg = response.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            if isinstance(response.get("content"), str) and response["content"].strip():
                return response["content"].strip()

            if isinstance(response.get("message"), str) and response["message"].strip():
                return response["message"].strip()

            if "error" in response:
                err = str(response.get("error", "")).strip()
                detail = str(response.get("detail", "")).strip()
                body = str(response.get("body", "")).strip()
                combined = " ".join(x for x in [err, detail, body] if x)
                return combined[:300] if combined else None

        return None

    def _history_to_query_and_context(self, conversation_history):
        """
        Correctly split:
          - query: latest user msg
          - context: everything BEFORE that user msg
          - ctx_hash: hash of last-k turns for caching
        """
        if not conversation_history:
            return "", None, "nohist"

        # find last user index
        last_user_idx = None
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

        # context string
        context_lines = []
        for m in ctx_msgs:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if content:
                context_lines.append(f"{role}: {content}")
        context = "\n".join(context_lines) if context_lines else None

        # hash last-k turns (user+assistant) to keep cache safe
        compact = []
        for m in conversation_history[-self.cache_last_k :]:
            if isinstance(m, dict):
                compact.append(f"{m.get('role','')}:{(m.get('content') or '').strip()}")
        ctx_hash = hashlib.sha256("\n".join(compact).encode("utf-8")).hexdigest()[:16]

        return query, context, ctx_hash

    def _is_error(self, raw):
        # Treat dict with "error" as failure
        return isinstance(raw, dict) and ("error" in raw)

    def _run_device(self, device, conversation_history):
        """
        Returns: raw_response, which_device, latency_ms
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

        end = datetime.now()
        lat_ms = (end - start).total_seconds() * 1000.0
        return raw, which, lat_ms

    def route_query(self, conversation_history):
        """
        main.py expects:
          (response_dict_with_response_key, response_tokens, device_string)
        """
        query, context, ctx_hash = self._history_to_query_and_context(conversation_history)

        # Response cache (optional)
        cache_hit = False
        if self.enable_response_cache and self.response_cache is not None:
            cache_key = (self.query_router.strategy, ctx_hash, query.lower())
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                text = cached["text"]
                raw = cached.get("raw")
                which = cached.get("device", "nano")  # MUST be "nano" or "orin" for your metrics

                # Count tokens for cached response so main.py + tester stay consistent
                response_tokens = self.token_counter.count_tokens({"role": "assistant", "content": text})

                return {
                    "response": text,
                    "raw": raw,
                    "cache_hit": True,
                    "routing_method": "response_cache",
                    "routing_confidence": 1.0,
                    "routing_reasoning": "Exact response-cache hit",
                    "routing_overhead_ms": 0.0,
                    "ok": True,
                }, response_tokens, which
        # 1) Decide device (measure routing overhead)
        t0 = time.time()
        routing_method = "unknown"
        routing_confidence = 0.0
        routing_reasoning = ""
        try:
            decision = self.query_router.route_query(query=query, context=context, context_key=ctx_hash)
            device = decision.device
            routing_method = decision.method
            routing_confidence = float(decision.confidence)
            routing_reasoning = decision.reasoning
            print(f"Routing decision: {device.upper()} | method={decision.method} | conf={decision.confidence:.3f}")
            print(f"Reason: {decision.reasoning}")
        except Exception as e:
            ctx_size = self.token_counter.get_context_size(conversation_history)
            device = "orin" if ctx_size > self.threshold_fallback else "nano"
            routing_method = "fallback_ctx_size"
            routing_confidence = 0.2
            routing_reasoning = f"router failed: {e}; ctx_size={ctx_size}, threshold_fallback={self.threshold_fallback}"
            print(f"Routing decision failed ({e}). Falling back to ctx_size={ctx_size} -> {device.upper()}")
        routing_overhead_ms = (time.time() - t0) * 1000.0

        # 2) Run chosen device (+ optional failover)
        raw, which, lat_ms = self._run_device(device, conversation_history)

        if self.enable_failover and self._is_error(raw):
            other = "orin" if which == "nano" else "nano"
            print(f"Primary device failed on {which.upper()}, failover -> {other.upper()}")
            raw2, which2, lat2 = self._run_device(other, conversation_history)
            # If failover succeeded, use it. Otherwise keep original error.
            if not self._is_error(raw2):
                raw, which, lat_ms = raw2, which2, lat2

        # 3) Normalize output + count tokens
        text = self._extract_text(raw) or "No response available"
        response_tokens = self.token_counter.count_tokens({"role": "assistant", "content": text})
        ok = not self._is_error(raw)

        # 4) Update performance-aware stats if available
        # (You’ll implement QueryRouter.update_perf in query_router_engine.py)
        try:
            if hasattr(self.query_router, "update_perf"):
                self.query_router.update_perf(which, lat_ms, response_tokens, ok=ok)
        except Exception:
            pass

        # 5) Save to response cache
        if self.enable_response_cache and self.response_cache is not None:
            cache_key = (self.query_router.strategy, ctx_hash, query.lower())
            self.response_cache.set(cache_key, {"text": text, "raw": raw, "device": which})

        # Return shape compatible with main/chat code
        return {
            "response": text,
            "raw": raw,
            "cache_hit": cache_hit,
            "routing_overhead_ms": round(routing_overhead_ms, 2),
            "routing_method": routing_method,
            "routing_confidence": round(float(routing_confidence), 4),
            "routing_reasoning": routing_reasoning,
            "ok": ok,
        }, response_tokens, which