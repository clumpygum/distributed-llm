# This file runs on the nano server

import logging
import requests
from flask import Flask, request, jsonify

logging.basicConfig(
    filename="flask.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "phi3-mini-nano:latest"

# Keep responses bounded for benchmarking
DEFAULT_NUM_PREDICT = 64
DEFAULT_TEMPERATURE = 0.0


@app.route("/")
def home():
    app.logger.info("GET / - Server alive")
    return "Test again: Server is running!\n", 200


@app.route("/health", methods=["GET"])
def health():
    # Lightweight health check (no inference)
    return jsonify({"ok": True}), 200


@app.route("/query", methods=["POST"])
def process_query():
    app.logger.info("Processing query...")

    data = request.get_json(silent=True) or {}
    query = data.get("query", None)

    app.logger.info(f"Received query: {query}")

    if not query:
        app.logger.error("No query provided")
        return jsonify({"error": "No query provided"}), 400

    # query is expected to be a list of {"role","content"}
    if isinstance(query, list):
        formatted_query = "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}" for m in query
        ).strip()
    elif isinstance(query, str):
        formatted_query = query.strip()
    else:
        return jsonify({"error": "Invalid query format. Expect list[role/content] or string."}), 400

    if not formatted_query:
        return jsonify({"error": "Empty query after formatting."}), 400

    # Ollama payload (NON-STREAMING + bounded output)
    payload = {
        "model": MODEL_NAME,
        "prompt": formatted_query,
        "stream": False,
        "options": {
            "num_predict": int(data.get("num_predict", DEFAULT_NUM_PREDICT)),
            "temperature": float(data.get("temperature", DEFAULT_TEMPERATURE)),
        }
    }

    app.logger.info("Running Ollama model via HTTP API...")
    try:
        # connect timeout 5s, read timeout 180s
        r = requests.post(OLLAMA_URL, json=payload, timeout=(5, 180))
        r.raise_for_status()
        out = r.json()

        # Ollama returns {"response": "...", ...}
        resp_text = (out.get("response") or "").strip()

        app.logger.info(f"Ollama OK. chars={len(resp_text)}")
        return jsonify({"response": resp_text})

    except requests.Timeout:
        app.logger.exception("Ollama timed out")
        return jsonify({"error": "Ollama timed out"}), 504

    except Exception as e:
        # This logs stack trace into flask.log
        app.logger.exception("Ollama call failed")
        # Keep error short but useful
        return jsonify({"error": f"Ollama call failed: {str(e)}"}), 500


if __name__ == "__main__":
    # threaded=True helps Flask handle concurrent requests better
    app.run(host="0.0.0.0", port=5001, threaded=True)