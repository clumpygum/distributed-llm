"""
routing_chatbot_tester.py

Benchmark + routing-accuracy tester for your distributed LLM chatbot.

Key behavior change (to stop 3-hour runs):
- Threshold sweeps apply ONLY to token strategy.
- All other strategies (heuristic / semantic / hybrid / perf) run ONCE using a fixed token_threshold
  (by default: the last value in --thresholds).

What it measures:
- Per-query: device used, latency, response tokens, energy (from power logs), latency/token, energy/token
- Per-experiment summary: totals by device + overall
- Optional routing accuracy if your query set includes expected_device labels.

Assumptions (same as your setup):
- Run with PYTHONPATH=src
- Router(strategy=..., config=..., threshold_fallback=..., benchmark_mode=...)
- Router exposes router.nano.server_manager and router.orin.server_manager
- Devices have ~/murong/logging_power.py that writes ~/murong/power.log
"""

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pexpect

from router import Router
from query_router_engine import BENCHMARK_CFG, PRODUCTION_CFG
from query_sets import query_sets

print("RUNNING TESTER:", os.path.abspath(__file__), flush=True)


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class QueryItem:
    text: str
    expected_device: Optional[str] = None  # "nano" | "orin" | None


@dataclass
class RunConfig:
    query_set_name: str
    thresholds: List[int]                 # used for token sweeps
    strategies: List[str]
    cache_modes: List[str]                # ["off", "on"]
    fixed_threshold_for_non_token: int    # used for heuristic/semantic/hybrid/perf
    output_csv: str
    output_per_query_csv: str


@dataclass
class SSHConfig:
    nano_ip: str
    orin_ip: str
    nano_ssh_user: str
    orin_ssh_user: str
    nano_ssh_port: int
    orin_ssh_port: int


# -----------------------------
# Query set loading
# -----------------------------

def normalize_query_set(raw_items: Any) -> List[QueryItem]:
    """
    Supports query_sets in either format:
      1) list[str]
      2) list[dict] where dict contains:
           - "query" or "text"
           - optional "expected_device" or "label" ("nano"/"orin")
    """
    out: List[QueryItem] = []

    if not isinstance(raw_items, list):
        raise ValueError("query_sets[<name>] must be a list")

    for x in raw_items:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(QueryItem(text=s, expected_device=None))
        elif isinstance(x, dict):
            q = (x.get("query") or x.get("text") or "").strip()
            if not q:
                continue
            exp = (x.get("expected_device") or x.get("label") or None)
            if isinstance(exp, str):
                exp = exp.lower().strip()
                if exp not in ("nano", "orin"):
                    exp = None
            else:
                exp = None
            out.append(QueryItem(text=q, expected_device=exp))

    if not out:
        raise ValueError("Query set is empty after normalization")

    return out


# -----------------------------
# SSH helpers + power logging
# -----------------------------

def _ssh_spawn(user: str, ip: str, port: int, timeout: int = 30) -> pexpect.spawn:
    cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"{user}@{ip} -p {port}"
    )
    return pexpect.spawn(cmd, encoding="utf-8", timeout=timeout)


def start_power_logging(device: str, ssh: SSHConfig) -> None:
    if device == "nano":
        user, ip, port = ssh.nano_ssh_user, ssh.nano_ip, ssh.nano_ssh_port
    elif device == "orin":
        user, ip, port = ssh.orin_ssh_user, ssh.orin_ip, ssh.orin_ssh_port
    else:
        raise ValueError("device must be nano/orin")

    try:
        child = _ssh_spawn(user, ip, port, timeout=30)
        child.expect(r"\$")

        # Sync device clock to host clock so [start_time, end_time] matches power.log timestamps.
        # If sudo prompts for a password, we skip (avoid hanging).
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        child.sendline(f"sudo timedatectl set-time \"{current_time}\" || true")

        idx = child.expect([r"\$", r"[Pp]assword", pexpect.TIMEOUT], timeout=10)
        if idx == 1:
            print(f"[power] {device} sudo password prompt detected; skipping time sync")
            child.sendcontrol("c")
            child.expect(r"\$", timeout=10)
        elif idx == 2:
            print(f"[power] {device} time sync command timed out; continuing")

        child.sendline("cd ~/murong")
        child.expect(r"\$")

        child.sendline("pkill -f \"python3 -u logging_power.py\" || true")
        child.expect(r"\$")

        child.sendline("nohup python3 -u logging_power.py > power.log 2>&1 & echo $!")
        child.expect(r"\d+")
        pid = child.match.group(0)
        print(f"[power] {device} logger PID: {pid}")

        child.sendline("exit")
        child.expect(pexpect.EOF)
    except (pexpect.TIMEOUT, pexpect.EOF):
        print(f"[power] {device} start failed (timeout/EOF)")


def stop_power_logging(device: str, ssh: SSHConfig) -> None:
    if device == "nano":
        user, ip, port = ssh.nano_ssh_user, ssh.nano_ip, ssh.nano_ssh_port
    elif device == "orin":
        user, ip, port = ssh.orin_ssh_user, ssh.orin_ip, ssh.orin_ssh_port
    else:
        raise ValueError("device must be nano/orin")

    cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"{user}@{ip} -p {port} "
        "'pkill -f \"python3 -u logging_power.py\" || true'"
    )
    subprocess.run(cmd, shell=True, capture_output=True, text=True)


def retrieve_power_log(device: str, ssh: SSHConfig, out_path: str) -> None:
    if device == "nano":
        user, ip, port = ssh.nano_ssh_user, ssh.nano_ip, ssh.nano_ssh_port
    elif device == "orin":
        user, ip, port = ssh.orin_ssh_user, ssh.orin_ip, ssh.orin_ssh_port
    else:
        raise ValueError("device must be nano/orin")

    remote_log = f"{user}@{ip}:/home/{user}/murong/power.log"
    scp_cmd = (
        f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-P {port} {remote_log} {out_path}"
    )
    child = pexpect.spawn(scp_cmd, encoding="utf-8", timeout=120)
    child.wait()


def parse_power_log(path: str) -> Dict[datetime, int]:
    power_data: Dict[datetime, int] = {}
    if not os.path.exists(path):
        return power_data

    with open(path, "r") as f:
        first = f.readline()
        # If first line looks like data, rewind.
        if ":" in first and any(ch.isdigit() for ch in first[:10]):
            f.seek(0)

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            ts_str, p_str = parts[0].strip(), parts[1].strip()

            ts = None
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    ts = datetime.strptime(ts_str, fmt)
                    break
                except ValueError:
                    pass
            if ts is None:
                continue

            try:
                p = int(p_str)
            except ValueError:
                continue

            power_data[ts] = p

    return power_data


def energy_for_window(power_data: Dict[datetime, int], start: datetime, end: datetime) -> float:
    """
    Proper integration: power samples are mW; timestamps give dt in seconds.
    mW * s = mJ
    """
    pts = sorted((t, p) for t, p in power_data.items() if start <= t <= end)
    if len(pts) < 2:
        return 0.0

    e_mj = 0.0
    for (t0, p0), (t1, _) in zip(pts, pts[1:]):
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        e_mj += p0 * dt
    return e_mj


# -----------------------------
# Router helpers
# -----------------------------

def build_router_config(cache_enabled: bool, token_threshold: int) -> Dict[str, Any]:
    """
    Build config from the canonical BENCHMARK_CFG / PRODUCTION_CFG defined in
    query_routing_engine.py, then override token_threshold for the sweep.

    cache_enabled=False  →  BENCHMARK_CFG  (cache off, clean accuracy measurement)
    cache_enabled=True   →  PRODUCTION_CFG (cache on, predictive routing)
    """
    base = PRODUCTION_CFG if cache_enabled else BENCHMARK_CFG
    return {**base, "token_threshold": token_threshold}


def try_clear_cache(router: Router) -> None:
    qr = getattr(router, "query_router", None)
    if qr is None:
        return
    if hasattr(qr, "clear_cache"):
        try:
            qr.clear_cache()
        except Exception:
            pass


def warmup(router: Router) -> None:
    # one tiny call to reduce cold-start overhead inside the experiment
    tmp_hist = [{"role": "user", "content": "Reply with exactly: OK"}]
    try:
        router.route_query(tmp_hist)
    except Exception:
        pass


def compute_accuracy(rows: List[Dict[str, Any]]) -> Optional[float]:
    labeled = [r for r in rows if r.get("expected_device") in ("nano", "orin")]
    if not labeled:
        return None
    correct = sum(1 for r in labeled if r.get("device_used") == r.get("expected_device"))
    return correct / len(labeled)


# -----------------------------
# CSV helpers
# -----------------------------

def ensure_csv_headers(path: str, headers: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(headers)


def append_csv_row(path: str, headers: List[str], row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(h, "") for h in headers])


# -----------------------------
# Core benchmark runner
# -----------------------------

def run_experiment(query_items: List[QueryItem], run_cfg: RunConfig, ssh_cfg: SSHConfig) -> None:
    summary_headers = [
        "query_set", "strategy", "cache_mode", "token_threshold",
        "routing_accuracy",
        "nano_total_latency_ms", "nano_total_energy_mJ", "nano_avg_power_mW", "nano_total_tokens",
        "nano_latency_per_token_ms", "nano_energy_per_token_mJ",
        "orin_total_latency_ms", "orin_total_energy_mJ", "orin_avg_power_mW", "orin_total_tokens",
        "orin_latency_per_token_ms", "orin_energy_per_token_mJ",
        "overall_total_latency_ms", "overall_total_energy_mJ", "overall_total_tokens",
        "overall_latency_per_token_ms", "overall_energy_per_token_mJ",
    ]

    per_query_headers = [
        "query_set", "strategy", "cache_mode", "token_threshold",
        "query_index", "query_text", "expected_device",
        "device_used", "cache_hit",
        "routing_method", "routing_confidence", "routing_reasoning", "routing_overhead_ms",
        "start_time", "end_time", "latency_ms", "response_tokens",
        "energy_mJ", "latency_per_token_ms", "energy_per_token_mJ",
    ]

    ensure_csv_headers(run_cfg.output_csv, summary_headers)
    ensure_csv_headers(run_cfg.output_per_query_csv, per_query_headers)

    # Start power logging for the whole session (stable + cheaper than per-experiment)
    start_power_logging("nano", ssh_cfg)
    start_power_logging("orin", ssh_cfg)

    all_rows: List[Dict[str, Any]] = []

    TOKEN_SWEEP_STRATEGIES = {"token"}

    for strategy in run_cfg.strategies:
        for cache_mode in run_cfg.cache_modes:
            cache_enabled = (cache_mode.lower() == "on")
            # cache_enabled=False → benchmark_mode=True  (BENCHMARK_CFG, cache off)
            # cache_enabled=True  → benchmark_mode=False (PRODUCTION_CFG, cache on)
            benchmark_mode = not cache_enabled

            thresholds_to_run = (
                run_cfg.thresholds
                if strategy in TOKEN_SWEEP_STRATEGIES
                else [run_cfg.fixed_threshold_for_non_token]
            )

            for threshold in thresholds_to_run:
                config = build_router_config(cache_enabled=cache_enabled, token_threshold=threshold)

                try:
                    router = Router(
                        strategy=strategy,
                        config=config,
                        threshold_fallback=threshold,
                        benchmark_mode=benchmark_mode,
                    )
                except Exception as e:
                    print(f"[skip] strategy={strategy} cache={cache_mode} thr={threshold} -> {e}")
                    continue

                print(
                    f"[run] strategy={strategy} cache={cache_mode} "
                    f"benchmark_mode={benchmark_mode} threshold={threshold}"
                )

                # Start servers (idempotent)
                try:
                    router.nano.server_manager.start_server()
                except Exception:
                    pass
                try:
                    router.orin.server_manager.start_server()
                except Exception:
                    pass

                # Fairness: clear cache between experiments
                try_clear_cache(router)

                # warmup once per experiment config
                warmup(router)

                conversation_history: List[Dict[str, str]] = []
                per_rows: List[Dict[str, Any]] = []

                for i, qi in enumerate(query_items):
                    conversation_history.append({"role": "user", "content": qi.text})

                    start_time = datetime.now()
                    try:
                        response, response_tokens, device_used = router.route_query(conversation_history)
                    except Exception as e:
                        end_time = datetime.now()
                        latency_ms = int((end_time - start_time).total_seconds() * 1000)

                        row = {
                            "query_set": run_cfg.query_set_name,
                            "strategy": strategy,
                            "cache_mode": cache_mode,
                            "token_threshold": threshold,
                            "query_index": i,
                            "query_text": qi.text,
                            "expected_device": qi.expected_device,
                            "device_used": "error",
                            "cache_hit": "",
                            "routing_method": "",
                            "routing_confidence": "",
                            "routing_reasoning": "",
                            "routing_overhead_ms": "",
                            "start_time": start_time,
                            "end_time": end_time,
                            "latency_ms": latency_ms,
                            "response_tokens": 0,
                            "energy_mJ": 0.0,
                            "latency_per_token_ms": "",
                            "energy_per_token_mJ": "",
                        }
                        per_rows.append(row)
                        print(f"[err] strategy={strategy} i={i}: {e}")
                        continue

                    end_time = datetime.now()
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)

                    # Keep conversation continuity + extract routing metadata (best-effort)
                    assistant_text = ""
                    cache_hit_val = ""
                    routing_method = ""
                    routing_confidence = ""
                    routing_reasoning = ""
                    routing_overhead_ms = ""

                    if isinstance(response, dict):
                        assistant_text = str(response.get("response", ""))
                        cache_hit_val = response.get("cache_hit", "")
                        routing_method = response.get("routing_method", "")
                        routing_confidence = response.get("routing_confidence", "")
                        routing_reasoning = response.get("routing_reasoning", "")
                        routing_overhead_ms = response.get("routing_overhead_ms", "")
                    else:
                        assistant_text = str(response)

                    conversation_history.append({"role": "assistant", "content": assistant_text})

                    row = {
                        "query_set": run_cfg.query_set_name,
                        "strategy": strategy,
                        "cache_mode": cache_mode,
                        "token_threshold": threshold,
                        "query_index": i,
                        "query_text": qi.text,
                        "expected_device": qi.expected_device,
                        "device_used": device_used,
                        "cache_hit": cache_hit_val,
                        "routing_method": routing_method,
                        "routing_confidence": routing_confidence,
                        "routing_reasoning": routing_reasoning,
                        "routing_overhead_ms": routing_overhead_ms,
                        "start_time": start_time,
                        "end_time": end_time,
                        "latency_ms": latency_ms,
                        "response_tokens": int(response_tokens or 0),
                        "energy_mJ": None,  # fill after pulling logs
                        "latency_per_token_ms": None,
                        "energy_per_token_mJ": None,
                    }
                    per_rows.append(row)

                all_rows.extend(per_rows)

                # Stop servers after each experiment config to reduce state carryover
                try:
                    router.nano.server_manager.stop_server()
                except Exception:
                    pass
                try:
                    router.orin.server_manager.stop_server()
                except Exception:
                    pass

    # Stop power logging and retrieve logs
    stop_power_logging("nano", ssh_cfg)
    stop_power_logging("orin", ssh_cfg)

    nano_log_local = "nano_power.log"
    orin_log_local = "orin_power.log"
    retrieve_power_log("nano", ssh_cfg, nano_log_local)
    retrieve_power_log("orin", ssh_cfg, orin_log_local)

    nano_power = parse_power_log(nano_log_local)
    orin_power = parse_power_log(orin_log_local)

    # Fill per-query energy + derived metrics, then write per-query CSV
    for row in all_rows:
        dev = row.get("device_used")
        if dev not in ("nano", "orin"):
            row["energy_mJ"] = 0.0
            row["latency_per_token_ms"] = ""
            row["energy_per_token_mJ"] = ""
        else:
            st = row["start_time"]
            en = row["end_time"]
            pdat = nano_power if dev == "nano" else orin_power

            e = energy_for_window(pdat, st, en)
            row["energy_mJ"] = round(e, 3)

            toks = int(row.get("response_tokens") or 0)
            lat = int(row.get("latency_ms") or 0)
            row["latency_per_token_ms"] = (lat / toks) if toks > 0 else ""
            row["energy_per_token_mJ"] = (e / toks) if toks > 0 else ""

        # Convert datetimes for CSV
        row["start_time"] = row["start_time"].isoformat(sep=" ")
        row["end_time"] = row["end_time"].isoformat(sep=" ")

        append_csv_row(run_cfg.output_per_query_csv, per_query_headers, row)

    # Aggregate summary per experiment key
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (r.get("strategy"), r.get("cache_mode"), int(r.get("token_threshold")))
        grouped.setdefault(key, []).append(r)

    for (strategy, cache_mode, threshold), rows in grouped.items():
        acc = compute_accuracy(rows)
        acc_out = "" if acc is None else round(acc, 4)

        def agg_for_device(dev: str) -> Tuple[int, float, int]:
            lat_sum = sum(int(x.get("latency_ms") or 0) for x in rows if x.get("device_used") == dev)
            e_sum = sum(float(x.get("energy_mJ") or 0.0) for x in rows if x.get("device_used") == dev)
            t_sum = sum(int(x.get("response_tokens") or 0) for x in rows if x.get("device_used") == dev)
            return lat_sum, e_sum, t_sum

        nano_lat, nano_e, nano_t = agg_for_device("nano")
        orin_lat, orin_e, orin_t = agg_for_device("orin")

        # avg power (mW): total_energy(mJ) / total_time(s)
        nano_avg_p = (nano_e / (nano_lat / 1000)) if nano_lat > 0 else 0.0
        orin_avg_p = (orin_e / (orin_lat / 1000)) if orin_lat > 0 else 0.0

        nano_lpt = (nano_lat / nano_t) if nano_t > 0 else ""
        nano_ept = (nano_e / nano_t) if nano_t > 0 else ""
        orin_lpt = (orin_lat / orin_t) if orin_t > 0 else ""
        orin_ept = (orin_e / orin_t) if orin_t > 0 else ""

        overall_lat = nano_lat + orin_lat
        overall_e = nano_e + orin_e
        overall_t = nano_t + orin_t
        overall_lpt = (overall_lat / overall_t) if overall_t > 0 else ""
        overall_ept = (overall_e / overall_t) if overall_t > 0 else ""

        summary_row = {
            "query_set": run_cfg.query_set_name,
            "strategy": strategy,
            "cache_mode": cache_mode,
            "token_threshold": threshold,
            "routing_accuracy": acc_out,

            "nano_total_latency_ms": nano_lat,
            "nano_total_energy_mJ": round(nano_e, 3),
            "nano_avg_power_mW": round(nano_avg_p, 6),
            "nano_total_tokens": nano_t,
            "nano_latency_per_token_ms": "" if nano_lpt == "" else round(nano_lpt, 6),
            "nano_energy_per_token_mJ": "" if nano_ept == "" else round(nano_ept, 6),

            "orin_total_latency_ms": orin_lat,
            "orin_total_energy_mJ": round(orin_e, 3),
            "orin_avg_power_mW": round(orin_avg_p, 6),
            "orin_total_tokens": orin_t,
            "orin_latency_per_token_ms": "" if orin_lpt == "" else round(orin_lpt, 6),
            "orin_energy_per_token_mJ": "" if orin_ept == "" else round(orin_ept, 6),

            "overall_total_latency_ms": overall_lat,
            "overall_total_energy_mJ": round(overall_e, 3),
            "overall_total_tokens": overall_t,
            "overall_latency_per_token_ms": "" if overall_lpt == "" else round(overall_lpt, 6),
            "overall_energy_per_token_mJ": "" if overall_ept == "" else round(overall_ept, 6),
        }

        append_csv_row(run_cfg.output_csv, summary_headers, summary_row)

    print(f"[done] wrote summary -> {run_cfg.output_csv}")
    print(f"[done] wrote per-query -> {run_cfg.output_per_query_csv}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--query-set", required=True, help="Key in query_sets dict (e.g., general_knowledge)")
    p.add_argument("--thresholds", nargs="+", type=int, default=[4000],
                   help="Thresholds to sweep ONLY for token strategy (e.g., 100 1000 4000)")
    p.add_argument("--fixed-threshold", type=int, default=None,
                   help="Token threshold used for NON-token strategies (default: last value of --thresholds)")
    p.add_argument("--strategies", nargs="+", default=["token", "heuristic", "semantic", "hybrid"],
                   help="Strategies to test")
    p.add_argument("--cache-modes", nargs="+", default=["off"], choices=["off", "on"],
                   help="Cache modes to test. 'off' = benchmark_mode (clean accuracy), 'on' = production_mode (predictive cache)")

    p.add_argument("--output-csv", default="benchmark_results.csv")
    p.add_argument("--output-per-query-csv", default="benchmark_per_query.csv")

    p.add_argument("--nano-ip", required=True)
    p.add_argument("--orin-ip", required=True)
    p.add_argument("--nano-ssh-user", default="nano")
    p.add_argument("--orin-ssh-user", default="orin")
    p.add_argument("--nano-ssh-port", type=int, default=22)
    p.add_argument("--orin-ssh-port", type=int, default=22)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.query_set not in query_sets:
        raise ValueError(f"Unknown query set: {args.query_set}. Available: {list(query_sets.keys())}")

    query_items = normalize_query_set(query_sets[args.query_set])

    fixed_thr = args.fixed_threshold if args.fixed_threshold is not None else args.thresholds[-1]

    run_cfg = RunConfig(
        query_set_name=args.query_set,
        thresholds=args.thresholds,
        strategies=args.strategies,
        cache_modes=args.cache_modes,
        fixed_threshold_for_non_token=fixed_thr,
        output_csv=args.output_csv,
        output_per_query_csv=args.output_per_query_csv,
    )

    ssh_cfg = SSHConfig(
        nano_ip=args.nano_ip,
        orin_ip=args.orin_ip,
        nano_ssh_user=args.nano_ssh_user,
        orin_ssh_user=args.orin_ssh_user,
        nano_ssh_port=args.nano_ssh_port,
        orin_ssh_port=args.orin_ssh_port,
    )

    # Always start fresh to avoid header/column mismatch across versions
    for p in (run_cfg.output_csv, run_cfg.output_per_query_csv):
        if os.path.exists(p):
            os.remove(p)

    run_experiment(query_items, run_cfg, ssh_cfg)


if __name__ == "__main__":
    # Optional: local default run without CLI typing
    import sys
    sys.argv += [
        "--query-set", "general_knowledge",
        "--thresholds", "100", "500", "1000", "2000", "4000",
        "--fixed-threshold", "1000",              # non-token runs once at 1000
        "--strategies", "token", "heuristic", "semantic", "hybrid", "perf",
        "--cache-modes", "off",
        "--nano-ip", "10.0.1.11",
        "--orin-ip", "10.0.1.8",
        "--output-csv", "results_final.csv",
        "--output-per-query-csv", "benchmark_per_query_all.csv",
    ]
    main()