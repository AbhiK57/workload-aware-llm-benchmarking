"""
Benchmark runner for LLM inference testing.
Orchestrates workloads, configurations, and statistics collection.
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import nest_asyncio
import numpy as np
from transformers import AutoTokenizer

from .benchmark_core import detect_gpu, pick_dtype, BenchmarkConfig
from .server import ServerConfig, ServerProcess
from .workloads import WorkloadGenerator, WORKLOAD_CONFIGS
from .metrics import stream_chat_request, aggregate_metrics, BatchMetrics
from .stats import StatisticsReporter, collect_kv_cache_stats

# Apply nest_asyncio for Colab/Jupyter compatibility
nest_asyncio.apply()


@dataclass
class TestConfig:
    """Configuration for a benchmark test variant."""
    
    name: str
    description: str
    engine: str = "vllm"  # "vllm" or "sglang"
    spec_config: Optional[str] = None  # Speculative decoding config
    gpu_mem_util: float = 0.85  # Target GPU memory utilization (0.0-1.0) to balance performance and stability
    extra_args: List[str] = field(default_factory=list)
    best_workloads: List[str] = field(default_factory=list)


# Default test configurations for vLLM
VLLM_TEST_CONFIGS = [
    TestConfig(
        name="baseline",
        description="Standard vLLM inference without speculation",
        engine="vllm",
        spec_config=None,
        gpu_mem_util=0.90,
        best_workloads=["random_medium"],
    ),
    TestConfig(
        name="spec_draft_1b",
        description="Speculative decoding with 1B draft model",
        engine="vllm",
        spec_config='{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 3}',
        gpu_mem_util=0.85,  # Need room for both models
        best_workloads=["code", "random_medium"],
    ),
    TestConfig(
        name="spec_ngram",
        description="Speculative decoding with n-gram matching",
        engine="vllm",
        spec_config='{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}',
        gpu_mem_util=0.90,
        best_workloads=["repetitive", "shared_prefix"],
    ),
    TestConfig(
        name="no_chunked_prefill",
        description="Simulates contiguous prefill by maximizing token budget (forces HoL blocking)",
        engine="vllm",
        spec_config=None,
        gpu_mem_util=0.90,
        extra_args=[
            "--max-num-batched-tokens", "8192", 
        ],
        best_workloads=["mixed_context"],
    ),
]

# Default test configurations for SGLang
SGLANG_TEST_CONFIGS = [
    TestConfig(
        name="baseline",
        description="Standard SGLang with RadixAttention",
        engine="sglang",
        spec_config=None,
        gpu_mem_util=0.90,
        extra_args=["--enable-metrics"], # <-- ADDED
        best_workloads=["shared_prefix", "random_medium"],
    ),
    TestConfig(
        name="no_radix",
        description="SGLang without RadixAttention (prefix caching disabled)",
        engine="sglang",
        spec_config=None,
        gpu_mem_util=0.90,
        extra_args=["--disable-radix-cache", "--enable-metrics"],
        best_workloads=["random_short"],
    ),
    TestConfig(
        name="torch_compile",
        description="SGLang with torch.compile for compute-bound speedups",
        engine="sglang",
        spec_config=None,
        gpu_mem_util=0.90,
        extra_args=["--enable-torch-compile", "--enable-metrics"],
        best_workloads=["code", "random_medium"],
    ),
]

# Default batch sizes to test
DEFAULT_BATCH_SIZES = [1, 8, 32, 64]


async def run_workload_async(
    base_url: str,
    model: str,
    prompts: List[str],
    tokenizer,
    concurrency: int = 32,
    max_tokens: int = 128,
    temperature: float = 0.2,
    timeout_s: int = 90,
    global_timeout_s: Optional[int] = None,
    request_rate: float = float("inf"),
    min_tokens: Optional[int] = None,
    collect_kv_stats: bool = True,
) -> BatchMetrics:
    """
    Run a workload with concurrent requests.

    Args:
        base_url: Server base URL
        model: Model name
        prompts: List of prompts to process
        tokenizer: Tokenizer for token counting
        concurrency: Maximum concurrent requests
        max_tokens: Max tokens per request
        temperature: Sampling temperature
        timeout_s: Per-request timeout (default 300s = 5 min)
        global_timeout_s: Total timeout for all requests (default: auto-calculated)
        request_rate: Request arrival rate in req/s. Use float("inf") to send all
                      requests immediately (default). When finite, requests are
                      launched with Poisson-distributed inter-arrival times.
        min_tokens: If set, forces at least this many output tokens per request.
                    Setting min_tokens == max_tokens fixes output length exactly,
                    eliminating generation variability across prompts.

    Returns:
        BatchMetrics with aggregated results
    """
    # Calculate global timeout if not specified
    # Allow more time for larger batches, but cap it
    if global_timeout_s is None:
        # Base: 30s + 2s per request, max 300s
        global_timeout_s = min(30 + len(prompts) * 2, 300)

    sem = asyncio.Semaphore(concurrency)

    async def process_one(session: aiohttp.ClientSession, idx: int, prompt: str):
        async with sem:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            return await stream_chat_request(
                session=session,
                base_url=base_url,
                model=model,
                messages=messages,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=f"req_{idx}",
                timeout_s=timeout_s,
                min_tokens=min_tokens,
            )

    t0 = time.perf_counter()

    # Poll KV cache stats in the background while requests are in-flight
    kv_snapshots = []

    async def _poll_kv(interval: float = 1.0):
        """Collect KV cache stats periodically during the workload."""
        consecutive_failures = 0
        try:
            await asyncio.sleep(0.5)
            while True:
                try:
                    stats = await asyncio.to_thread(collect_kv_cache_stats, base_url)
                    if stats:
                        kv_snapshots.append(stats)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    consecutive_failures += 1

                if consecutive_failures >= 3:
                    print(f"\n  [WARNING] Systemic telemetry failure: KV cache metrics endpoint refused connection 3 times.")
                    break

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    kv_poll_task = asyncio.create_task(_poll_kv()) if collect_kv_stats else None

    # Use a single shared session with a connection-pooling connector
    # to avoid overwhelming the server with separate connections
    connector = aiohttp.TCPConnector(limit=min(concurrency, 128), force_close=True)

    all_tasks: List[asyncio.Task] = []
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, prompt in enumerate(prompts):
                # Poisson arrival: sleep for exponentially distributed interval
                if request_rate < float("inf") and i > 0:
                    await asyncio.sleep(random.expovariate(request_rate))
                all_tasks.append(asyncio.create_task(process_one(session, i, prompt)))

            # Wait for all in-flight requests, honouring the global timeout
            try:
                metrics_list = await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True),
                    timeout=global_timeout_s,
                )
            except asyncio.TimeoutError:
                from .metrics import RequestMetrics
                metrics_list = [
                    RequestMetrics(
                        request_id=f"req_{i}",
                        success=False,
                        error_message=f"Global timeout ({global_timeout_s}s) exceeded",
                    )
                    for i in range(len(prompts))
                ]
    finally:
        # Stop KV polling
        if kv_poll_task is not None:
            kv_poll_task.cancel()
            try:
                await kv_poll_task
            except asyncio.CancelledError:
                pass

    t1 = time.perf_counter()

    # Handle exceptions
    valid_metrics = []
    for i, m in enumerate(metrics_list):
        if isinstance(m, Exception):
            from .metrics import RequestMetrics
            valid_metrics.append(RequestMetrics(
                request_id=f"req_{i}",
                success=False,
                error_message=str(m),
            ))
        else:
            valid_metrics.append(m)

    wall_s = t1 - t0
    result = aggregate_metrics(valid_metrics, wall_s)

    # Attach peak KV cache snapshot (highest utilization seen mid-flight)
    if kv_snapshots:
        result.kv_cache_snapshot = max(kv_snapshots, key=lambda s: s.cache_utilization_pct)

    return result


def run_workload(
    base_url: str,
    model: str,
    prompts: List[str],
    tokenizer,
    concurrency: int = 32,
    max_tokens: int = 128,
    temperature: float = 0.2,
    request_rate: float = float("inf"),
    min_tokens: Optional[int] = None,
    collect_kv_stats: bool = True,
) -> BatchMetrics:
    "Synchronous wrapper for run_workload_async."
    return asyncio.run(run_workload_async(
        base_url=base_url,
        model=model,
        prompts=prompts,
        tokenizer=tokenizer,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        request_rate=request_rate,
        min_tokens=min_tokens,
        collect_kv_stats=collect_kv_stats,
    ))


class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    
    Coordinates running multiple configurations across multiple workloads
    and batch sizes, collecting comprehensive statistics.
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        test_configs: Optional[List[TestConfig]] = None,
        batch_sizes: Optional[List[int]] = None,
        workloads: Optional[List[str]] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            model_id: Target model to benchmark
            test_configs: List of test configurations (defaults to VLLM_TEST_CONFIGS)
            batch_sizes: List of batch sizes to test
            workloads: List of workload names to test
        """
        self.model_id = model_id
        self.test_configs = test_configs or VLLM_TEST_CONFIGS
        self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        self.workloads = workloads or list(WORKLOAD_CONFIGS.keys())
        
        # Initialize components
        self.gpu_info = detect_gpu()
        self.dtype = pick_dtype(self.gpu_info)
        self.tokenizer = None
        self.workload_generator = None
        self.stats_reporter = StatisticsReporter()
        
    def _initialize(self):
        """Initialize tokenizer and workload generator."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True
            )
        
        if self.workload_generator is None:
            self.workload_generator = WorkloadGenerator(self.model_id)
            self.workload_generator.prepare_workloads(
                n_random=max(self.batch_sizes),
                n_shared=max(self.batch_sizes),
            )
    
    def run(
        self,
        warmup_requests: int = 3,
        collect_kv_stats: bool = True,
        verbose: bool = True,
    ):
        """
        Run the full benchmark suite.
        
        Args:
            warmup_requests: Number of warmup requests before each config
            collect_kv_stats: Whether to collect KV cache statistics
            verbose: Print progress information
            
        Returns:
            pandas DataFrame with all results
        """
        self._initialize()
        
        if verbose:
            print("🖥️ GPU:", self.gpu_info)
            print(f"📦 Target model: {self.model_id}")
            print(f"🔧 Data type: {self.dtype}")
            print(f"📋 Workloads: {self.workloads}")
            print(f"📊 Batch sizes: {self.batch_sizes}")
        
        for config in self.test_configs:
            self._run_config(
                config,
                warmup_requests=warmup_requests,
                collect_kv_stats=collect_kv_stats,
                verbose=verbose,
            )
        
        # Compute speedups vs baseline
        self.stats_reporter.compute_speedups("baseline")
        
        # Print summary
        if verbose:
            self.stats_reporter.print_summary()
        
        return self.stats_reporter.to_dataframe()
    
    def _run_config(
        self,
        config: TestConfig,
        warmup_requests: int = 3,
        collect_kv_stats: bool = True,
        verbose: bool = True,
    ):
        """Run benchmarks for a single configuration."""
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔧 CONFIG: {config.name}")
            print(f"   {config.description}")
            print("=" * 60)
        
        # Build server config
        extra_args = list(config.extra_args)
        if config.spec_config:
            extra_args.extend(["--speculative-config", config.spec_config])
        
        server_cfg = ServerConfig(
            engine=config.engine,
            model_id=self.model_id,
            port=8000,
            dtype=self.dtype,
            gpu_mem_util=config.gpu_mem_util,
            max_model_len=8192,  # headroom for >=1500-tok prompts + 512-tok output
            extra_args=extra_args,
        )
        
        server = ServerProcess(server_cfg)
        server.start()
        
        try:
            # Warmup
            if warmup_requests > 0 and verbose:
                print("\n  🔥 Warming up...")
                warmup_prompts = self.workload_generator.get_workload("random_short")[:warmup_requests]
                _ = run_workload(
                    base_url=server_cfg.base_url(),
                    model=self.model_id,
                    prompts=warmup_prompts,
                    tokenizer=self.tokenizer,
                    concurrency=warmup_requests,
                    max_tokens=32,
                )
            
            # Run workloads
            for workload_name in self.workloads:
                self._run_workload(
                    config=config,
                    workload_name=workload_name,
                    server_cfg=server_cfg,
                    collect_kv_stats=collect_kv_stats,
                    verbose=verbose,
                )
        
        finally:
            server.stop()
    
    def _run_workload(
        self,
        config: TestConfig,
        workload_name: str,
        server_cfg: ServerConfig,
        collect_kv_stats: bool = True,
        verbose: bool = True,
    ):
        """Run a single workload across all batch sizes."""
        workload_cfg = WORKLOAD_CONFIGS.get(workload_name)
        if not workload_cfg:
            print(f"  ⚠️ Unknown workload: {workload_name}")
            return
        
        prompts = self.workload_generator.get_workload(workload_name)
        
        if verbose:
            print(f"\n  📋 Workload: {workload_name}")
            print(f"     {workload_cfg.description}")
        
        # Per-workload batch size limits to avoid OOM/hanging
        max_batch_by_workload = {
            "shared_prefix": 64,  # Long shared prefix exhausts KV cache at high batch
            "repetitive": 8,      # vLLM stalls with many concurrent long-output requests
            "long_context": 32,   # 500-1500 token prompts use significant KV cache
            "mixed_context": 64,  # Half long (>=1500 tok) prompts; cap matches long_context
        }
        max_batch = max_batch_by_workload.get(workload_name, float('inf'))
        
        for batch_size in self.batch_sizes:
            # Skip batch sizes that are too large for this workload
            if batch_size > max_batch:
                if verbose:
                    print(f"\n     Batch size: {batch_size}... ⏭️ Skipped (max {max_batch} for {workload_name})")
                continue
            
            batch_prompts = prompts[:batch_size]
            
            if verbose:
                print(f"\n     Batch size: {batch_size}...", end=" ", flush=True)
            
            try:
                batch_metrics = run_workload(
                    base_url=server_cfg.base_url(),
                    model=self.model_id,
                    prompts=batch_prompts,
                    tokenizer=self.tokenizer,
                    concurrency=batch_size,
                    max_tokens=workload_cfg.max_tokens,
                    temperature=workload_cfg.temperature,
                )
                
                # Collect KV cache stats
                base_url = server_cfg.base_url() if collect_kv_stats else ""
                
                # Record stats
                self.stats_reporter.add_run(
                    config_name=config.name,
                    workload_name=workload_name,
                    batch_size=batch_size,
                    batch_metrics=batch_metrics,
                    base_url=base_url,
                )
                
                if verbose:
                    success_rate = batch_metrics.successful_requests / max(batch_metrics.total_requests, 1) * 100
                    print(f"✅ {batch_metrics.throughput_output_tok_s} tok/s, "
                          f"TTFT p50: {batch_metrics.ttft_p50_ms:.1f}ms, "
                          f"Success: {success_rate:.0f}%")
            
            except Exception as e:
                self.stats_reporter.add_run(
                    config_name=config.name,
                    workload_name=workload_name,
                    batch_size=batch_size,
                    batch_metrics=None,
                    error_message=str(e),
                )
                
                if verbose:
                    print(f"❌ Error: {e}")
    
    def get_results(self):
        """Get results DataFrame."""
        return self.stats_reporter.to_dataframe()
    
    def save_results(self, prefix: str = "benchmark"):
        """Save results to CSV files."""
        return self.stats_reporter.save_results(prefix)


def benchmark(
    model_id: str = None,
    batch_sizes: List[int] = None,
    workloads: List[str] = None,
    configs: List[str] = None,
):
    """
    Convenience function to run a benchmark.
    
    Args:
        model_id: Model to benchmark (defaults to env var or Llama-3.2-3B)
        batch_sizes: List of batch sizes
        workloads: List of workload names
        configs: List of config names to test (filters TEST_CONFIGS)
        
    Returns:
        pandas DataFrame with results
    """
    model_id = model_id or os.environ.get("TARGET_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    
    # Filter configs if specified
    test_configs = VLLM_TEST_CONFIGS
    if configs:
        test_configs = [c for c in VLLM_TEST_CONFIGS if c.name in configs]
    
    runner = BenchmarkRunner(
        model_id=model_id,
        test_configs=test_configs,
        batch_sizes=batch_sizes,
        workloads=workloads,
    )
    
    return runner.run()


def quick_benchmark():
    """Run a quick benchmark with smaller batch sizes and fewer workloads."""
    return benchmark(
        batch_sizes=[1, 8, 32],
        workloads=["random_medium", "repetitive", "code"],
        configs=["baseline", "spec_ngram"],
    )
