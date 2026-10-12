# LLM Inference Server

A C++ server that shares one LLM on one GPU across many concurrent clients. llama.cpp runs the model; this project is the serving layer around it: a bounded request queue, a dynamic batching scheduler, token streaming over HTTP and gRPC, overload rejection, and Prometheus metrics.

## Architecture

```text
HTTP clients (POST /generate)      gRPC clients (Inference.Generate)
            \                              /
             \                            /
        admit: queue full? -> reject now (503 / RESOURCE_EXHAUSTED)
                          |
                 bounded RequestQueue  (mutex + condition variable)
                          |
              scheduler thread: take up to batch_size requests,
              waiting at most batch_wait_ms for the batch to fill;
              drop requests already past their deadline
                          |
              llama.cpp: one context, one sequence per request,
              prefill all prompts, then decode one token per
              sequence per step until each finishes
                          |
         per-request promise (non-streaming) or TokenChannel (streaming)
                          |
                    back to the client
```

- **Frontends** (`src/server.cpp`, `src/grpc_server.cpp`) parse a request, stamp it with an enqueue time and deadline, and try to enqueue it. If the queue is full they reject immediately instead of making the client wait.
- **Request queue** (`src/request_queue.hpp`) is a bounded FIFO. Frontend threads produce, the scheduler thread consumes.
- **Scheduler** (`src/scheduler.cpp`) does dynamic batching: it runs as soon as `batch_size` requests are ready or `batch_wait_ms` has passed, whichever comes first.
- **Inference** (`src/inference.cpp`) runs a batch as multiple sequences in one llama.cpp context. Each sequence finishes on its own (end of generation, token limit, deadline, or client disconnect).
- **Metrics** (`src/metrics.cpp`) exposes counters, gauges, and histograms in Prometheus text format.

## Build

Requires CMake, a C++20 compiler, and (for the gRPC frontend) gRPC + protobuf, e.g. `brew install grpc`.

```bash
git clone https://github.com/ggml-org/llama.cpp   # vendored, gitignored
mkdir -p models   # put a GGUF model here, e.g. qwen2.5-0.5b-instruct-q4_k_m.gguf
cmake -S . -B build
cmake --build build -j
```

- macOS uses Metal automatically.
- NVIDIA GPUs: configure with `cmake -S . -B build -DGGML_CUDA=ON`. No source changes are needed; llama.cpp picks the backend.
- Without gRPC installed: `-DENABLE_GRPC=OFF`.

## Run

```bash
./build/main --batch-size 8 --batch-wait-ms 10 --max-queue-depth 64
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | GGUF model path |
| `--port` | 8080 | HTTP port |
| `--grpc-port` | 50051 | gRPC port |
| `--batch-size` | 4 | Max requests per batch |
| `--batch-wait-ms` | 10 | Max time to wait for a batch to fill |
| `--max-queue-depth` | 64 | Queue capacity before rejecting |
| `--request-timeout-ms` | 30000 | Deadline per request |
| `--http-threads` | 64 | HTTP worker threads (keep above expected client count) |

## API

**HTTP**

```bash
# whole response
curl localhost:8080/generate -d '{"prompt": "The capital of France is", "n_predict": 16}'
# {"response": " Paris. ...", "finish_reason": "length", "n_generated": 16}

# streamed, one JSON object per line
curl -N localhost:8080/generate -d '{"prompt": "Count: 1, 2,", "n_predict": 8, "stream": true}'
```

Status codes: `503` queue full (with `Retry-After`), `504` deadline passed, `400` bad request.

**gRPC** (`proto/inference.proto`): `Inference.Generate` streams `GenerateReply` messages. A small test client is built alongside the server:

```bash
./build/grpc_client "The capital of Japan is" 16
```

**Metrics**: `GET /metrics`

| Metric | Type |
| --- | --- |
| `llm_requests_accepted_total`, `llm_requests_rejected_total` | counter |
| `llm_requests_completed_total{reason}` | counter |
| `llm_requests_in_flight`, `llm_queue_depth`, `llm_queue_capacity` | gauge |
| `llm_tokens_generated_total`, `llm_batches_total` | counter |
| `llm_request_latency_seconds`, `llm_queue_wait_seconds`, `llm_time_to_first_token_seconds` | histogram |
| `llm_batch_size` | histogram |

Percentiles come from the histograms, e.g. p99 latency over 1 minute:

```promql
histogram_quantile(0.99, rate(llm_request_latency_seconds_bucket[1m]))
```

## Benchmarks

```bash
bench/run_benchmarks.sh 60      # 60s per run; results appended to bench/results.jsonl
```

The suite runs 16 closed-loop clients at batch size 1, 4 and 8, then sends open-loop traffic at 3x the measured capacity with a small queue. `bench/loadgen.py` can also be run directly (`--clients N` or `--rate R`).

Latency percentiles are for successful requests only; rejected requests are counted and timed separately.

## Known limitations

- Static batching: a batch runs until all of its sequences finish, so a new request waits for the current batch even if one sequence is long. Continuous batching (adding sequences mid-batch) is the next step.
- A new llama.cpp context is created per batch instead of reusing one with KV-cache slots.
- Greedy sampling only.
- Requests are FIFO; there is no priority or fairness between clients.
