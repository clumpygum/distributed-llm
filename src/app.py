from flask import Flask, request, jsonify
from flask_cors import CORS
from router import Router

app = Flask(__name__)
CORS(app)

# Default configuration
# FIX 4: Restored heuristic weight to its meaningful default (was 0.0, which
# silently disabled it in hybrid mode).
base_config = {
    "cache_enabled": True,
    "enable_response_cache": True,
    "enable_failover": True,
    "weights": {"token": 0.25, "semantic": 0.45, "heuristic": 0.30}
}

# --- GLOBAL STATE ---
current_strategy_name = "hybrid"
active_router = Router(strategy=current_strategy_name, config=base_config)

# FIX 1: Per-session conversation history stored server-side.
# Keyed by session_id (sent from the frontend, defaults to "default").
# Capped at the last 10 messages so memory doesn't grow unbounded.
HISTORY_LIMIT = 10
conversation_histories: dict = {}


@app.route('/chat', methods=['POST'])
def chat():
    global active_router, current_strategy_name

    data = request.json
    user_input = data.get('message', '')
    requested_strategy = data.get('strategy', 'hybrid')
    session_id = data.get('session_id', 'default')   # FIX 1: accept session_id

    # Map React dropdown names to Python strategy names
    if requested_strategy == 'token-counting':
        requested_strategy = 'token'

    if not user_input.strip():
        return jsonify({"error": "No message provided"}), 400

    # FIX 3: Use change_strategy() instead of re-initialising the whole Router.
    # This preserves the PerformanceAwareRouter's rolling latency stats and the
    # routing cache that were built up during the session.
    if requested_strategy != current_strategy_name:
        print(f"🔄 Switching strategy: {current_strategy_name} -> {requested_strategy}")
        try:
            active_router.query_router.change_strategy(requested_strategy)
            current_strategy_name = requested_strategy
        except Exception as e:
            print(f"Failed to switch strategy: {e}")
            return jsonify({"error": f"Failed to switch strategy: {e}"}), 500

    # FIX 1: Retrieve (or create) this session's history, append the new user
    # message, then pass the full history to route_query so the router has
    # proper multi-turn context for routing decisions and response generation.
    history = conversation_histories.setdefault(session_id, [])
    history.append({"role": "user", "content": user_input})

    try:
        response_data, tokens, device = active_router.route_query(history)

        # Unpack response dict
        if isinstance(response_data, dict):
            reply_text = response_data.get("response", "")
            reasoning  = response_data.get("routing_reasoning", f"Method: {requested_strategy}")
            # FIX 2: Surface the extra metadata the router always populates
            method     = response_data.get("routing_method", requested_strategy)
            confidence = response_data.get("routing_confidence", 0.0)
            cache_hit  = response_data.get("cache_hit", False)
        else:
            reply_text = str(response_data)
            reasoning  = "Direct response"
            method     = requested_strategy
            confidence = 0.0
            cache_hit  = False

        # FIX 1: Append assistant reply to history and trim to the last N messages
        history.append({"role": "assistant", "content": reply_text})
        conversation_histories[session_id] = history[-HISTORY_LIMIT:]

        return jsonify({
            "reply":      reply_text,
            "device":     device,
            "reasoning":  reasoning,
            "method":     method,        # FIX 2
            "confidence": confidence,    # FIX 2
            "cache_hit":  cache_hit,     # FIX 2
            "tokens":     tokens,
        })

    except Exception as e:
        print(f"❌ Error during routing: {e}")
        # Remove the user message we just appended so history stays consistent
        if history and history[-1]["role"] == "user":
            history.pop()
        return jsonify({
            "reply":      "System Error: The router encountered an issue.",
            "device":     "error",
            "reasoning":  str(e),
            "method":     requested_strategy,
            "confidence": 0.0,
            "cache_hit":  False,
            "tokens":     0,
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Return the conversation history for a given session (useful for debugging)."""
    session_id = request.args.get('session_id', 'default')
    return jsonify(conversation_histories.get(session_id, []))


@app.route('/history', methods=['DELETE'])
def clear_history():
    """Clear the conversation history for a session (e.g. when user clicks 'New Chat')."""
    session_id = request.args.get('session_id', 'default')
    conversation_histories.pop(session_id, None)
    return jsonify({"cleared": session_id})


if __name__ == '__main__':
    print("🚀 API running on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)