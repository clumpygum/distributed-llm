# This file runs on the servers
from jtop import jtop
from datetime import datetime
import time
import sys

def pick_total_power_mw(stats: dict) -> int:
    candidates = [
        stats.get("Power TOT", None),
        stats.get("Power VIN_SYS_5V0", None),
    ]
    for v in candidates:
        if v is None:
            continue
        try:
            return int(float(v))
        except Exception:
            pass
    for k, v in stats.items():
        if isinstance(k, str) and k.startswith("Power "):
            try:
                return int(float(v))
            except Exception:
                continue
    return 0

def main():
    with jtop() as jetson:
        while jetson.ok():
            t0 = time.time()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            total_mw = pick_total_power_mw(jetson.stats)

            try:
                print(f"{ts}: {total_mw}", flush=True)
            except BrokenPipeError:
                # e.g. when piped into `head`
                sys.exit(0)

            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0 - elapsed))

if __name__ == "__main__":
    main()