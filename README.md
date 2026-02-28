# Distributed LLM Chatbot on NVIDIA Jetson

## Overview

This repository contains a locally distributed LLM-based chatbot architecture designed for secure, efficient edge deployment. The system orchestrates inference across multiple NVIDIA Jetson devices (e.g., Nano + Orin) using a dynamic Query Routing Engine to optimize for latency, throughput, and hardware constraints.

---

## Core Contribution: Query Routing Engine

The routing engine dynamically determines the optimal hardware target for any given prompt. It balances the lower computational overhead of the Jetson Nano against the high-performance capabilities of the Jetson Orin.

### Routing Strategies

| Strategy | Logic | Use Case |
| :--- | :--- | :--- |
| **Token** | Routes based on estimated token/context length thresholds. | Distinguishing short queries from long-form documents. |
| **Heuristic** | Rule-based routing via query type or intent detection. | Directing coding tasks vs. casual conversation. |
| **Semantic** | Embedding similarity to predefined topic centroids. | Domain-specific expertise routing. |
| **Perf** | Real-time health metrics and recent latency history. | High-availability and automatic failover. |
| **Hybrid** | Weighted combination of Token and Semantic signals. | General-purpose optimization. |

### Hybrid Scoring Formula

The Hybrid strategy calculates a Capability Score ($S$) for each device based on query complexity ($C$), semantic similarity, and device health ($H$):

$$S = (w_1 \cdot C_{tokens}) + (w_2 \cdot \text{Sim}_{centroid}) + (w_3 \cdot H_{latency})$$

---

## Features

* **Multi-Device Edge Inference:** Seamlessly bridge Jetson Nano and Orin via local networking (LAN or SSH tunnels).
* **Intelligent Failover:** Automatic fallback to the secondary device if the primary node is unreachable or returns an error.
* **Dual-Layer Caching:**
    * **Exact Match:** Direct context hash lookup for identical queries.
    * **Semantic Cache:** Similarity-based lookup for conceptually equivalent queries within a specific context.
* **Benchmark Harness:** Evaluation suite that logs latency, token counts, and success rates to CSV for performance analysis.
* **Power Logging:** Support for collecting device-level power metrics during active inference (optional).

---

## Repository Structure

```text
src/
├── router/           # Core Query Routing Engine logic
│   └── query_routing_engine.py
├── devices/          # API wrappers for Nano and Orin (Ollama-based)
│   ├── nano_api.py
│   └── orin_api.py
└── server/           # Flask/FastAPI entry point
    └── app.py
tests/
└── routing_chatbot_tester.py  # Benchmarking and Evaluation harness
data/
├── query_sets/       # Test datasets (JSON/CSV)
└── results/          # Benchmark logs and measurement exports
```

## Setup and Installation
### 1. Host Environment

The host machine (where the router and benchmarking harness run) requires Python 3.10+.

```Bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Device Configuration

Ensure both Jetson devices are reachable on your network and have Ollama installed and running.
* Jetson Nano: 10.0.1.11 (Example IP)
* Jetson Orin: 10.0.1.8 (Example IP)

### 3. Environment Variables

Create a .env file in the root directory to store device IPs and configuration:

```Plaintext
NANO_IP="10.0.1.10"
ORIN_IP="10.0.1.8"
LOG_LEVEL="INFO"
```
## Running the System
### Start the API Server

Launch the routing server to handle incoming requests:

```Bash
PYTHONPATH=src python3 src/server/app.py
Run Benchmarking
```
To evaluate the efficiency of different routing strategies and thresholds:

```Bash
PYTHONPATH=src python3 src/tests/routing_chatbot_tester.py \
  --query-set general_knowledge \
  --thresholds 100 500 1000 2000 \
  --strategies token heuristic semantic hybrid perf \
  --cache-modes off on \
  --output-csv results_final.csv \
  --output-per-query-csv benchmark_per_query_all.csv
```
## Notes and Common Issues
* Large Files: Do not commit .venv/, local model weights, or large CSV result files. These should be managed via .gitignore.

* SSH Tunnels: If accessing devices over a public network, ensure SSH tunnels are established and ports are correctly forwarded.

* Ollama Bindings: Verify that Ollama is configured to listen on 0.0.0.0 to allow external requests from the router.

## Roadmap
* Implement auto-tuning feedback loops for routing weights.

* Add real-time latency distribution visualization to the frontend.

* Integrate jtop/tegrastats for automated power-per-query analysis.
 
* Improve semantic centroid training pipeline for domain-specific tasks.

## License
Distributed under the MIT License.
