"""
LLM Inference Benchmark Package

A modular benchmarking framework for comparing LLM inference engines
like vLLM and SGLang. Supports various workloads, speculative decoding
configurations, and comprehensive statistics collection.

Example usage:
    from llm_benchmark import benchmark, quick_benchmark
    
    # Run full benchmark
    df = benchmark()
    
    # Run quick benchmark
    df = quick_benchmark()
    
    # Custom benchmark
    from llm_benchmark import BenchmarkRunner, TestConfig
    
    runner = BenchmarkRunner(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        batch_sizes=[1, 8, 32],
        workloads=["random_short", "code"],
    )
    df = runner.run()
"""

from .benchmark_core import (
    detect_gpu,
    pick_dtype,
    percentiles_ms,
    format_duration,
    format_tokens,
    BenchmarkConfig,
)

from .server import (
    ServerConfig,
    ServerProcess,
)

from .workloads import (
    WorkloadGenerator,
    WorkloadConfig,
    WORKLOAD_CONFIGS,
)

from .metrics import (
    RequestMetrics,
    BatchMetrics,
    stream_chat_request,
    aggregate_metrics,
    tokenize_chat,
)

from .stats import (
    KVCacheStats,
    GoodputMetrics,
    BenchmarkRunStats,
    StatisticsReporter,
    fetch_server_metrics,
    collect_kv_cache_stats,
    compute_goodput,
)

from .runner import (
    TestConfig,
    BenchmarkRunner,
    VLLM_TEST_CONFIGS,
    SGLANG_TEST_CONFIGS,
    DEFAULT_BATCH_SIZES,
    benchmark,
    quick_benchmark,
    run_workload,
)

from .datasets import (
    PromptSample,
    load_dataset_prompts,
    load_random,
    load_sharegpt,
    load_sonnet,
    load_dolly,
)

__version__ = "0.1.0"
__all__ = [
    # Core utilities
    "detect_gpu",
    "pick_dtype",
    "percentiles_ms",
    "format_duration",
    "format_tokens",
    "BenchmarkConfig",

    # Server management
    "ServerConfig",
    "ServerProcess",

    # Workloads
    "WorkloadGenerator",
    "WorkloadConfig",
    "WORKLOAD_CONFIGS",

    # Metrics
    "RequestMetrics",
    "BatchMetrics",
    "stream_chat_request",
    "aggregate_metrics",
    "tokenize_chat",

    # Statistics
    "KVCacheStats",
    "GoodputMetrics",
    "BenchmarkRunStats",
    "StatisticsReporter",
    "fetch_server_metrics",
    "collect_kv_cache_stats",
    "compute_goodput",

    # Runner
    "TestConfig",
    "BenchmarkRunner",
    "VLLM_TEST_CONFIGS",
    "SGLANG_TEST_CONFIGS",
    "DEFAULT_BATCH_SIZES",
    "benchmark",
    "quick_benchmark",
    "run_workload",

    # Dataset loaders
    "PromptSample",
    "load_dataset_prompts",
    "load_random",
    "load_sharegpt",
    "load_sonnet",
    "load_dolly",
]
