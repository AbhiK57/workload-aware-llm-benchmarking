"""
benchmark_serving.py — CLI tool for benchmarking an already-running vLLM/SGLang server.

Unlike the notebook-based BenchmarkRunner (which manages the server lifecycle),
this script connects to an existing server and runs a dataset-driven benchmark
with configurable traffic patterns.

Usage:
    python -m llm_benchmark.benchmark_serving \\
        --base-url http://localhost:8000/v1 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --dataset-name sharegpt \\
        --num-prompts 500 \\
        --max-concurrency 64 \\
        --request-rate 10.0 \\
        --percentile-metrics ttft,tpot,itl,e2el \\
        --output-json results.json

Supported --percentile-metrics values (comma-separated, any order):
    ttft  Time to First Token
    tpot  Time Per Output Token  (≡ ITL for streaming clients)
    itl   Inter-Token Latency
    e2el  End-to-End Latency (total request time)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import List, Optional

import aiohttp
from transformers import AutoTokenizer

from .datasets import load_dataset_prompts, PromptSample
from .metrics import stream_chat_request, aggregate_metrics, BatchMetrics, RequestMetrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m llm_benchmark.benchmark_serving",
        description="Benchmark a running vLLM/SGLang server with configurable datasets and traffic patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server connection
    g = p.add_argument_group("Server")
    g.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="Server base URL")
    g.add_argument("--model", required=True,
                   help="Model name/ID served by the server")

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset-name", default="sonnet",
                   choices=["random", "sharegpt", "sonnet", "dolly"],
                   help="Dataset to benchmark")
    g.add_argument("--num-prompts", type=int, default=200,
                   help="Number of prompts to send")
    g.add_argument("--random-input-len", type=int, default=512,
                   help="Target input length (random dataset) or length filter (others)")
    g.add_argument("--random-output-len", type=int, default=128,
                   help="Target output length when dataset provides no output length")
    g.add_argument("--fixed-output-len", type=int, default=None,
                   help="Fix output length exactly (sets min_tokens=max_tokens). "
                        "Overrides --random-output-len and eliminates output variability.")
    g.add_argument("--prefix-len", type=int, default=0,
                   help="Shared prefix length for sonnet workload (0 = disabled)")
    g.add_argument("--seed", type=int, default=42,
                   help="RNG seed for dataset sampling")

    # Traffic simulation
    g = p.add_argument_group("Traffic")
    g.add_argument("--max-concurrency", type=int, default=64,
                   help="Maximum number of concurrent in-flight requests")
    g.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Request arrival rate in req/s. "
                        "Use 'inf' (default) to send all requests as fast as possible. "
                        "When finite, inter-arrival times are Poisson-distributed.")

    # Generation
    g = p.add_argument_group("Generation")
    g.add_argument("--temperature", type=float, default=0.2,
                   help="Sampling temperature")
    g.add_argument("--timeout", type=int, default=600,
                   help="Per-request timeout in seconds")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument(
        "--percentile-metrics",
        type=str,
        default=None,
        metavar="METRICS",
        help=(
            "Comma-separated list of metrics to show with full percentile breakdown "
            "(p50/p90/p95/p99). Supported: ttft,tpot,itl,e2el. "
            "Example: --percentile-metrics ttft,tpot,itl,e2el"
        ),
    )
    g.add_argument("--output-json", default=None, metavar="PATH",
                   help="Save results to this JSON file")

    return p


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

async def _run_benchmark(
    base_url: str,
    model: str,
    samples: List[PromptSample],
    tokenizer,
    max_concurrency: int,
    request_rate: float,
    temperature: float,
    timeout_s: int,
    fixed_output_len: Optional[int],
    seed: int,
) -> BatchMetrics:
    """
    Send all prompts to the server and collect per-request metrics.

    When *request_rate* is finite, requests are launched with
    Poisson-distributed inter-arrival times (mean = 1/rate seconds).
    Concurrency is still bounded by *max_concurrency* via a semaphore.
    """
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(session: aiohttp.ClientSession, idx: int, sample: PromptSample) -> RequestMetrics:
        async with sem:
            max_tok = fixed_output_len if fixed_output_len is not None else sample.output_tokens
            min_tok = fixed_output_len  # None or == max_tok → fixed length
            return await stream_chat_request(
                session=session,
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": sample.prompt},
                ],
                tokenizer=tokenizer,
                max_tokens=max_tok,
                temperature=temperature,
                request_id=f"req_{idx}",
                timeout_s=timeout_s,
                min_tokens=min_tok,
            )

    connector = aiohttp.TCPConnector(limit=min(max_concurrency, 512), force_close=True)
    t0 = time.perf_counter()

    tasks: List[asyncio.Task] = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, sample in enumerate(samples):
            if request_rate < float("inf") and i > 0:
                await asyncio.sleep(rng.expovariate(request_rate))
            tasks.append(asyncio.create_task(_one(session, i, sample)))

        raw = await asyncio.gather(*tasks, return_exceptions=True)

    t1 = time.perf_counter()

    valid: List[RequestMetrics] = []
    for i, m in enumerate(raw):
        if isinstance(m, Exception):
            valid.append(RequestMetrics(
                request_id=f"req_{i}",
                success=False,
                error_message=str(m),
            ))
        else:
            valid.append(m)

    return aggregate_metrics(valid, t1 - t0)


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

_VALID_METRICS = {"ttft", "tpot", "itl", "e2el"}


def _print_results(metrics: BatchMetrics, args: argparse.Namespace) -> None:
    width = 66
    print("\n" + "=" * width)
    print("BENCHMARK RESULTS")
    print("=" * width)
    print(f"  Requests:          {metrics.successful_requests}/{metrics.total_requests} successful")
    print(f"  Wall time:         {metrics.wall_time_s:.2f}s")
    print(f"  Throughput:        {metrics.throughput_output_tok_s:.1f} output tok/s")
    print(f"  Request rate:      {metrics.throughput_requests_s:.3f} req/s")
    print(f"  Avg input tokens:  {metrics.avg_input_tokens:.0f}")
    print(f"  Avg output tokens: {metrics.avg_output_tokens:.0f}")

    # Parse which metrics to expand
    if args.percentile_metrics is not None:
        requested = {m.strip().lower() for m in args.percentile_metrics.split(",")}
        unknown = requested - _VALID_METRICS
        if unknown:
            print(f"\n  [WARNING] Unknown --percentile-metrics values ignored: {unknown}")
            print(f"            Valid options: {', '.join(sorted(_VALID_METRICS))}")
        requested &= _VALID_METRICS
    else:
        requested = set()  # compact summary only

    def _pct_row(label: str, mean, p50, p90, p95, p99):
        print(f"\n  {label}:")
        print(f"    mean={mean:>8.2f}  p50={p50:>8.2f}  "
              f"p90={p90:>8.2f}  p95={p95:>8.2f}  p99={p99:>8.2f}  (ms)")

    if requested:
        if "ttft" in requested:
            _pct_row("TTFT  (Time to First Token)",
                     metrics.ttft_mean_ms, metrics.ttft_p50_ms,
                     metrics.ttft_p90_ms, metrics.ttft_p95_ms, metrics.ttft_p99_ms)

        if "tpot" in requested:
            # TPOT (Time Per Output Token) = ITL in a streaming client measurement.
            # Both capture inter-token intervals; the labels differ by convention.
            _pct_row("TPOT  (Time Per Output Token ≡ ITL)",
                     metrics.itl_mean_ms, metrics.itl_p50_ms,
                     metrics.itl_p90_ms, metrics.itl_p95_ms, metrics.itl_p99_ms)

        if "itl" in requested:
            _pct_row("ITL   (Inter-Token Latency)",
                     metrics.itl_mean_ms, metrics.itl_p50_ms,
                     metrics.itl_p90_ms, metrics.itl_p95_ms, metrics.itl_p99_ms)

        if "e2el" in requested:
            _pct_row("E2EL  (End-to-End Latency)",
                     metrics.request_latency_mean_ms, metrics.request_latency_p50_ms,
                     metrics.request_latency_p90_ms,
                     metrics.request_latency_p95_ms, metrics.request_latency_p99_ms)
    else:
        # Compact one-liner summary
        print(f"\n  TTFT  p50/p99: {metrics.ttft_p50_ms:>8.2f} / {metrics.ttft_p99_ms:>8.2f} ms")
        print(f"  ITL   p50/p99: {metrics.itl_p50_ms:>8.2f} / {metrics.itl_p99_ms:>8.2f} ms")
        print(f"  E2EL  p50/p99: {metrics.request_latency_p50_ms:>8.2f} / {metrics.request_latency_p99_ms:>8.2f} ms")

    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Resolve output_len: fixed-output-len overrides random-output-len
    output_len = args.fixed_output_len if args.fixed_output_len is not None else args.random_output_len

    print(f"Loading dataset '{args.dataset_name}' ({args.num_prompts} prompts)...")
    samples = load_dataset_prompts(
        dataset_name=args.dataset_name,
        num_prompts=args.num_prompts,
        tokenizer=tokenizer,
        input_len=args.random_input_len,
        output_len=output_len,
        prefix_len=args.prefix_len,
        seed=args.seed,
    )

    avg_in  = sum(s.input_tokens  for s in samples) / max(len(samples), 1)
    avg_out = sum(s.output_tokens for s in samples) / max(len(samples), 1)
    print(f"  Loaded {len(samples)} prompts — "
          f"avg input {avg_in:.0f} tok, avg output {avg_out:.0f} tok")

    rate_str = f"{args.request_rate:.1f} req/s" if args.request_rate < float("inf") else "unlimited"
    print(f"\nBenchmarking {args.base_url}")
    print(f"  Concurrency: {args.max_concurrency}  |  Rate: {rate_str}")
    if args.fixed_output_len is not None:
        print(f"  Fixed output length: {args.fixed_output_len} tokens (min_tokens=max_tokens)")

    metrics = asyncio.run(_run_benchmark(
        base_url=args.base_url,
        model=args.model,
        samples=samples,
        tokenizer=tokenizer,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        temperature=args.temperature,
        timeout_s=args.timeout,
        fixed_output_len=args.fixed_output_len,
        seed=args.seed,
    ))

    _print_results(metrics, args)

    if args.output_json:
        result_dict = metrics.to_dict()
        result_dict.pop("request_metrics", None)  # exclude per-request details by default
        result_dict["_config"] = {
            "base_url":        args.base_url,
            "model":           args.model,
            "dataset":         args.dataset_name,
            "num_prompts":     args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "request_rate":    args.request_rate if args.request_rate < float("inf") else "inf",
            "fixed_output_len": args.fixed_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "temperature":     args.temperature,
        }
        with open(args.output_json, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
