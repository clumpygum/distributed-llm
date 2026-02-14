from flask import Flask, request, jsonify
from flask_cors import CORS
from router import Router

app = Flask(__name__)
CORS(app)

# Default configuration
base_config = {
    "cache_enabled": True,
    "enable_response_cache": True,
    "enable_failover": True,
    "weights": {"token": 0.4, "semantic": 0.6, "heuristic": 0.0} 
}

# --- GLOBAL STATE ---
# We track the strategy name here since the Router object might not expose it
current_strategy_name = "hybrid"
active_router = Router(strategy=current_strategy_name, config=base_config)

@app.route('/chat', methods=['POST'])
def chat():
    # Use 'global' so we can modify the router instance when the user switches dropdowns
    global active_router, current_strategy_name

    data = request.json
    user_input = data.get('message', '')
    requested_strategy = data.get('strategy', 'hybrid') 
    
    # Map React names to Python names
    if requested_strategy == 'token-counting': requested_strategy = 'token'
    
    # 1. Check if we need to switch strategies
    # We compare against our global variable, NOT the router object
    if requested_strategy != current_strategy_name:
        print(f"🔄 Switching Strategy: {current_strategy_name} -> {requested_strategy}")
        try:
            # Re-initialize the router with the new strategy
            active_router = Router(strategy=requested_strategy, config=base_config)
            current_strategy_name = requested_strategy
        except Exception as e:
            print(f"Failed to switch strategy: {e}")
            return jsonify({"error": "Failed to switch strategy"}), 500

    # 2. Route the Query
    history = [{"role": "user", "content": user_input}]
    
    try:
        response_data, tokens, device = active_router.route_query(history)
        
        # Unpack response
        if isinstance(response_data, dict):
            reply_text = response_data.get("response", "")
            # Get reasoning or default to the method name
            reasoning = response_data.get("routing_reasoning", f"Method: {requested_strategy}")
        else:
            reply_text = str(response_data)
            reasoning = "Direct response"

        return jsonify({
            "reply": reply_text,
            "device": device,       
            "reasoning": reasoning,
            "tokens": tokens
        })
        
    except Exception as e:
        print(f"❌ Error during routing: {e}")
        # Return a clean error to the frontend so it doesn't just hang
        return jsonify({
            "reply": "System Error: The router encountered an issue.",
            "device": "error",
            "reasoning": str(e),
            "tokens": 0
        }), 500

if __name__ == '__main__':
    # Use port 8000 to be safe from AirPlay conflicts
    print("🚀 API running on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)