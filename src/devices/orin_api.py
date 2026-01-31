# src/devices/orin_api.py
# This file runs on the Orin server

import logging
import requests
from flask import Flask, request, jsonify

# Configure logging to capture everything into flask.log
logging.basicConfig(
    filename="flask.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)

# Constants
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3"  # Ensure this matches the model pulled on your Orin

@app.route("/")
def home():
    app.logger.info("Test again: Flask server is running!")
    return "Test again: Server is running!\n", 200

@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health check.
    The ServerManager uses this to confirm Flask is ready before sending queries.
    """
    return jsonify({"ok": True}), 200

@app.route("/query", methods=["POST"])
def process_query():
    app.logger.info("Processing query...")

    # Capture the request body
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    
    app.logger.info(f"Received query payload.")

    if not query:
        app.logger.error("No query provided")
        return jsonify({"error": "No query provided"}), 400

    # 1. Format the conversation history
    # Converts list of dicts to a simple string prompt for Ollama
    if isinstance(query, list):
        formatted_query = "\n".join(
            [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in query]
        )
    else:
        formatted_query = str(query)

    # 2. Run the Ollama model via HTTP (Much faster than subprocess)
    payload = {
        "model": MODEL_NAME,
        "prompt": formatted_query,
        "stream": False
    }

    app.logger.info("Sending request to local Ollama API...")
    
    try:
        # connect_timeout=5s, read_timeout=180s (inference can be slow)
        r = requests.post(OLLAMA_URL, json=payload, timeout=(5, 180))
        r.raise_for_status()
        
        # Ollama returns: {"response": "actual text", ...}
        ollama_resp = r.json()
        response_text = ollama_resp.get("response", "")

        app.logger.info(f"Ollama success. Generated {len(response_text)} chars.")
        return jsonify({"response": response_text})

    except requests.Timeout:
        app.logger.error("Ollama request timed out.")
        return jsonify({"error": "Ollama timed out processing the request."}), 504
        
    except requests.RequestException as e:
        app.logger.error(f"Ollama request failed: {str(e)}")
        return jsonify({"error": f"Ollama connection failed: {str(e)}"}), 500
    
    except Exception as e:
        app.logger.exception("Unexpected error during processing")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    # threaded=True allows handling the health check while a query might be processing
    app.run(host="0.0.0.0", port=5000, threaded=True)