# This file runs on the Orin server

import shlex
import sys
import logging
from flask import Flask, request, jsonify
import subprocess

# Configure logging to capture everything into flask.log
logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)

@app.route("/")
def home():
    app.logger.info("Test again: Flask server is running!")
    return "Test again: Server is running!\n"

@app.route("/query", methods=["POST"])
def process_query():
    app.logger.info("Processing query...")

    # Capture the request body (payload) from the incoming POST request
    data = request.json
    query = data.get("query", "")
    
    app.logger.info(f"Received query: {query}")

    if not query:
        app.logger.error("No query provided")
        return jsonify({"error": "No query provided"}), 400

    # Run the Ollama model locally
    app.logger.info("Running Ollama model (shlex)...")
    # Convert chat history to string format expected by ollama
    formatted_query = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in query]
    )
    
    result = subprocess.run(
        ["ollama", "run", "llama3", formatted_query],
        input= "",
        capture_output=True,
        text=True
    )
    

    app.logger.info(f"Ollama model result: {result.stdout}")
    app.logger.error(f"Ollama model error (stderr): {result.stderr}")
        
    return jsonify({"response": result.stdout})

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

