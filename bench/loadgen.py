#!/usr/bin/env python3
"""Load generator for the inference server (Python standard library only).

Two modes:

  Closed loop: N clients, each sends a request, waits for the answer, repeats.
    python3 bench/loadgen.py --clients 16 --duration 60

  Open loop: requests arrive at a fixed average rate no matter how fast the
  server answers. This is how you push the server past capacity.
    python3 bench/loadgen.py --rate 12 --duration 60

Latency percentiles are reported for successful (200) requests only.
Rejected requests (503) are reported separately with how fast they came back.
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Explain what a hash map is in two sentences.",
    "Write a haiku about the ocean.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "List three uses for a paperclip.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What does a CPU cache do?",
    "Give me a tip for staying focused while studying.",
    "Describe a sunset to someone who has never seen one.",
]


class Results:
    def __init__(self):
        self.lock = threading.Lock()
        self.ok_latencies = []
        self.reject_latencies = []
        self.tokens = 0
        self.timeouts = 0
        self.errors = 0

    def record(self, status, latency, n_generated):
        with self.lock:
            if status == 200:
                self.ok_latencies.append(latency)
                self.tokens += n_generated
            elif status == 503:
                self.reject_latencies.append(latency)
            elif status == 504:
                self.timeouts += 1
            else:
                self.errors += 1


def send(url, prompt, n_predict, timeout):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
            return resp.status, time.perf_counter() - start, payload.get("n_generated", 0)
    except urllib.error.HTTPError as e:
        e.read()
        return e.code, time.perf_counter() - start, 0
    except Exception:
        return -1, time.perf_counter() - start, 0


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run_closed(args, results, stop_at):
    def client():
        rng = random.Random()
        while time.perf_counter() < stop_at:
            status, latency, n = send(args.url, rng.choice(PROMPTS), args.n_predict, args.timeout)
            results.record(status, latency, n)

    threads = [threading.Thread(target=client) for _ in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_open(args, results, stop_at):
    rng = random.Random()
    threads = []

    def one():
        status, latency, n = send(args.url, rng.choice(PROMPTS), args.n_predict, args.timeout)
        results.record(status, latency, n)

    next_at = time.perf_counter()
    while next_at < stop_at:
        now = time.perf_counter()
        if next_at > now:
            time.sleep(next_at - now)
        t = threading.Thread(target=one)
        t.start()
        threads.append(t)
        next_at += rng.expovariate(args.rate)   # Poisson arrivals
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080/generate")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--clients", type=int, help="closed loop: number of concurrent clients")
    mode.add_argument("--rate", type=float, help="open loop: average requests per second")
    ap.add_argument("--duration", type=float, default=30, help="seconds to send load")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120, help="client-side timeout in seconds")
    ap.add_argument("--warmup", type=int, default=2, help="requests to send before measuring")
    ap.add_argument("--label", default="", help="name for this run in the output")
    ap.add_argument("--json", help="also write the summary to this file")
    args = ap.parse_args()

    for _ in range(args.warmup):
        send(args.url, PROMPTS[0], args.n_predict, args.timeout)

    results = Results()
    start = time.perf_counter()
    stop_at = start + args.duration
    if args.clients:
        run_closed(args, results, stop_at)
    else:
        run_open(args, results, stop_at)
    wall = time.perf_counter() - start

    ok = len(results.ok_latencies)
    rejected = len(results.reject_latencies)
    offered = ok + rejected + results.timeouts + results.errors
    summary = {
        "label": args.label,
        "mode": f"closed, {args.clients} clients" if args.clients else f"open, {args.rate} req/s",
        "n_predict": args.n_predict,
        "wall_seconds": round(wall, 2),
        "requests_sent": offered,
        "ok": ok,
        "rejected_503": rejected,
        "timeouts_504": results.timeouts,
        "errors": results.errors,
        "throughput_req_per_s": round(ok / wall, 3),
        "throughput_tokens_per_s": round(results.tokens / wall, 1),
        "latency_ok_p50_s": round(percentile(results.ok_latencies, 50), 3),
        "latency_ok_p95_s": round(percentile(results.ok_latencies, 95), 3),
        "latency_ok_p99_s": round(percentile(results.ok_latencies, 99), 3),
        "reject_latency_p50_ms": round(percentile(results.reject_latencies, 50) * 1000, 2),
        "reject_latency_max_ms": round(max(results.reject_latencies) * 1000, 2) if rejected else None,
    }

    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k:<{width}}  {v}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
