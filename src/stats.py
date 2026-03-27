"""
Enhanced statistics reporting for LLM inference benchmarking.
Includes KV cache analysis, goodput metrics, and comprehensive reporting.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


@dataclass
class KVCacheStats:
    """
    KV Cache usage statistics from the inference server.
    
    These metrics help understand memory efficiency and potential bottlenecks.
    """
    
    # Cache utilization
    cache_utilization_pct: float = 0.0  # Percentage of KV cache in use
    num_blocks_used: int = 0
    num_blocks_total: int = 0
    
    # Memory info
    gpu_cache_memory_gb: float = 0.0
    cpu_cache_memory_gb: float = 0.0
    
    # Prefix caching (if enabled)
    prefix_cache_hit_rate: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    
    # Block management
    num_free_blocks: int = 0
    num_swapped_blocks: int = 0
    
    @classmethod
    def from_server_metrics(cls, metrics_dict: Dict[str, Any]) -> "KVCacheStats":
        """Parse KV cache stats from server /metrics endpoint."""
        stats = cls()

        def _get(*candidate_keys, default=None):
            """Look up a metric by trying multiple possible names."""
            for k in candidate_keys:
                if k in metrics_dict:
                    return metrics_dict[k]
            return default

        # GPU cache utilization (vLLM uses colon or underscore across versions;
        # SGLang exposes similar metrics with sglang: prefix)
        val = _get("vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc",
                    "sglang:gpu_cache_usage_perc", "sglang_gpu_cache_usage_perc")
        if val is not None:
            stats.cache_utilization_pct = float(val) * 100

        # num_gpu_blocks is a label on vllm:cache_config_info, not its own metric.
        # fetch_server_metrics promotes it to "vllm:cache_config_num_gpu_blocks".
        val = _get("vllm:cache_config_num_gpu_blocks",
                    "vllm:num_gpu_blocks", "vllm_num_gpu_blocks",
                    "sglang:cache_config_num_gpu_blocks",
                    "sglang:num_gpu_blocks", "sglang_num_gpu_blocks")
        if val is not None:
            stats.num_blocks_total = int(val)

        # vLLM does not expose num_gpu_blocks_used directly; derive it from
        # utilization percentage × total block count.
        if stats.num_blocks_total > 0 and stats.cache_utilization_pct > 0:
            stats.num_blocks_used = round(
                stats.cache_utilization_pct / 100.0 * stats.num_blocks_total
            )
            stats.num_free_blocks = stats.num_blocks_total - stats.num_blocks_used

        # Prefix caching hit rate.
        # vLLM 0.5+: vllm:gpu_prefix_cache_hit_rate (0.0–1.0 fraction)
        # Older vLLM / fallbacks kept for compatibility.
        val = _get("vllm:gpu_prefix_cache_hit_rate",
                    "vllm:prefix_cache_hit_rate", "vllm_prefix_cache_hit_rate",
                    "vllm:prefix_caching_block_hit_rate",
                    "sglang:cache_hit_rate", "sglang_cache_hit_rate",
                    "sglang:radix_cache_hit_rate", "sglang_radix_cache_hit_rate")
        if val is not None:
            stats.prefix_cache_hit_rate = float(val) * 100

        # Prefix cache query / hit counters.
        # vLLM 0.5+: vllm:cache_query_total / vllm:cache_query_hit
        val = _get("vllm:cache_query_total",
                    "vllm:prefix_cache_queries", "vllm_prefix_cache_queries",
                    "vllm:prefix_caching_num_queries_total",
                    "sglang:cache_queries_total", "sglang_cache_queries_total")
        if val is not None:
            stats.prefix_cache_queries = int(val)

        val = _get("vllm:cache_query_hit",
                    "vllm:prefix_cache_hits", "vllm_prefix_cache_hits",
                    "vllm:prefix_caching_num_hits_total",
                    "sglang:cache_hits_total", "sglang_cache_hits_total")
        if val is not None:
            stats.prefix_cache_hits = int(val)

        return stats


@dataclass
class GoodputMetrics:
    """
    Goodput metrics - measures useful/successful work done.
    
    Goodput is a more realistic measure of system performance than raw throughput
    because it only counts successfully completed requests.
    """
    
    # Raw throughput
    raw_throughput_tok_s: float = 0.0  # All tokens / time
    
    # Goodput (successful tokens only)
    goodput_tok_s: float = 0.0  # Successful output tokens / time
    goodput_requests_s: float = 0.0  # Successful requests / time
    
    # Quality metrics
    success_rate: float = 0.0  # Percentage of successful requests
    timeout_rate: float = 0.0  # Percentage of timeouts
    error_rate: float = 0.0  # Percentage of errors (non-timeout)
    
    # Efficiency
    effective_batch_utilization: float = 0.0  # How well batching is being used
    throughput_per_watt: float = 0.0  # If power metrics available
    
    # SLA compliance
    sla_ttft_target_ms: float = 500.0
    sla_compliance_rate: float = 0.0  # % of requests meeting TTFT target
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BenchmarkRunStats:
    """
    Comprehensive statistics for a benchmark run.
    
    Aggregates all metrics for a single configuration/workload combination.
    """
    
    # Run identification
    run_id: str = ""
    timestamp: str = ""
    config_name: str = ""
    workload_name: str = ""
    batch_size: int = 0
    
    # Core performance metrics
    throughput_output_tok_s: float = 0.0
    throughput_total_tok_s: float = 0.0
    
    # Latency metrics
    ttft_p50_ms: float = float("nan")
    ttft_p95_ms: float = float("nan")
    ttft_p99_ms: float = float("nan")
    itl_p50_ms: float = float("nan")
    itl_p95_ms: float = float("nan")
    itl_p99_ms: float = float("nan")
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    
    # Goodput metrics
    goodput: Optional[GoodputMetrics] = None
    
    # KV cache stats (if available)
    kv_cache: Optional[KVCacheStats] = None
    
    # Derived metrics
    speedup_vs_baseline: float = float("nan")
    
    # Error info
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame/CSV export."""
        result = {}
        for k, v in self.__dict__.items():
            if k == "goodput" and v:
                for gk, gv in v.to_dict().items():
                    result[f"goodput_{gk}"] = gv
            elif k == "kv_cache" and v:
                for ck, cv in v.__dict__.items():
                    result[f"kv_{ck}"] = cv
            elif k in ("goodput", "kv_cache"):
                continue
            else:
                result[k] = v
        return result


def fetch_server_metrics(base_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Fetch metrics from server's /metrics endpoint.
    
    Parses Prometheus-format metrics into a dictionary.
    """
    try:
        # Remove /v1 suffix if present
        metrics_url = base_url.replace("/v1", "") + "/metrics"
        response = requests.get(metrics_url, timeout=timeout)
        response.raise_for_status()
        
        metrics = {}
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Parse Prometheus format: metric_name{labels} value [timestamp]
            try:
                if "{" in line:
                    name, rest = line.split("{", 1)
                    labels_str, value = rest.rsplit("}", 1)
                    # value may be "0.5" or "0.5 1706000000123" (optional timestamp)
                    val = float(value.strip().split()[0])
                    name = name.strip()
                    metrics[f"{name}{{{labels_str}}}"] = val
                    # Also store by base name (without labels) for easy lookup
                    if name not in metrics:
                        metrics[name] = val
                    # vllm:cache_config_info carries num_gpu_blocks etc. as labels,
                    # not as a value.  Promote each numeric label to its own key so
                    # the rest of the code can look them up by name.
                    if "cache_config_info" in name:
                        eng = name.split(":")[0]  # "vllm" or "sglang"
                        for kv in labels_str.split(","):
                            kv = kv.strip()
                            if "=" not in kv:
                                continue
                            lk, lv = kv.split("=", 1)
                            try:
                                metrics[f"{eng}:cache_config_{lk.strip()}"] = float(
                                    lv.strip().strip('"')
                                )
                            except ValueError:
                                pass
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics[parts[0]] = float(parts[1])
            except (ValueError, IndexError):
                continue
                
        return metrics
    except Exception as e:
        return {"error": str(e)}


def collect_kv_cache_stats(base_url: str) -> Optional[KVCacheStats]:
    """Collect KV cache statistics from the server."""
    metrics = fetch_server_metrics(base_url)
    if "error" in metrics:
        return None
    stats = KVCacheStats.from_server_metrics(metrics)
    # If neither utilization nor block counts were populated the metrics
    # endpoint returned nothing useful (e.g. wrong metric names, server not
    # ready yet).  Return None so callers don't record misleading zeros.
    if stats.cache_utilization_pct == 0.0 and stats.num_blocks_total == 0:
        return None
    return stats


def compute_goodput(
    batch_metrics,
    sla_ttft_ms: float = 500.0
) -> GoodputMetrics:
    """
    Compute goodput metrics from batch metrics.
    
    Args:
        batch_metrics: BatchMetrics object with request details
        sla_ttft_ms: Target TTFT for SLA compliance calculation
        
    Returns:
        GoodputMetrics with computed values
    """
    if batch_metrics.total_requests == 0:
        return GoodputMetrics()
    
    # Success rates
    success_rate = batch_metrics.successful_requests / batch_metrics.total_requests
    
    # Count timeouts and other errors
    timeout_count = sum(
        1 for m in batch_metrics.request_metrics
        if not m.success and "timeout" in m.error_message.lower()
    )
    error_count = batch_metrics.failed_requests - timeout_count
    
    timeout_rate = timeout_count / batch_metrics.total_requests
    error_rate = error_count / batch_metrics.total_requests
    
    # SLA compliance (TTFT under target)
    sla_compliant = sum(
        1 for m in batch_metrics.request_metrics
        if m.success and not np.isnan(m.ttft_s) and m.ttft_s * 1000 <= sla_ttft_ms
    )
    sla_compliance = sla_compliant / max(batch_metrics.successful_requests, 1)
    
    # Effective batch utilization
    # (how many requests were processed concurrently on average)
    if batch_metrics.wall_time_s > 0 and batch_metrics.successful_requests > 0:
        avg_request_time = np.mean([
            m.total_s for m in batch_metrics.request_metrics if m.success
        ])
        concurrent_estimate = (avg_request_time * batch_metrics.successful_requests) / batch_metrics.wall_time_s
        batch_util = min(concurrent_estimate / batch_metrics.total_requests, 1.0)
    else:
        batch_util = 0.0
    
    return GoodputMetrics(
        raw_throughput_tok_s=batch_metrics.throughput_output_tok_s,
        goodput_tok_s=batch_metrics.goodput_tok_s,
        goodput_requests_s=batch_metrics.throughput_requests_s * success_rate,
        success_rate=round(success_rate * 100, 2),
        timeout_rate=round(timeout_rate * 100, 2),
        error_rate=round(error_rate * 100, 2),
        effective_batch_utilization=round(batch_util * 100, 2),
        sla_ttft_target_ms=sla_ttft_ms,
        sla_compliance_rate=round(sla_compliance * 100, 2),
    )


class StatisticsReporter:
    """
    Comprehensive statistics reporter for benchmark results.
    
    Collects, aggregates, and presents benchmark statistics in various formats.
    """
    
    def __init__(self):
        self.all_runs: List[BenchmarkRunStats] = []
        self.start_time = datetime.now()
    
    def add_run(
        self,
        config_name: str,
        workload_name: str,
        batch_size: int,
        batch_metrics,
        base_url: str = "",
        error_message: str = "",
    ) -> BenchmarkRunStats:
        """
        Add a benchmark run result and compute derived statistics.
        
        Args:
            config_name: Name of the configuration being tested
            workload_name: Name of the workload
            batch_size: Batch size used
            batch_metrics: BatchMetrics from the run
            base_url: Server URL for collecting KV cache stats
            error_message: Error message if run failed
            
        Returns:
            BenchmarkRunStats for this run
        """
        run_id = f"{config_name}_{workload_name}_{batch_size}_{int(time.time())}"
        
        # Use mid-flight KV snapshot from BatchMetrics if available (collected
        # while requests were in-flight, so utilization is non-zero).
        # Fall back to a post-completion fetch from the server.
        kv_stats = None
        if batch_metrics and getattr(batch_metrics, 'kv_cache_snapshot', None):
            kv_stats = batch_metrics.kv_cache_snapshot
        elif base_url and not error_message:
            kv_stats = collect_kv_cache_stats(base_url)
        
        # Compute goodput metrics
        goodput = None
        if batch_metrics and hasattr(batch_metrics, 'request_metrics'):
            goodput = compute_goodput(batch_metrics)
        
        stats = BenchmarkRunStats(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config_name=config_name,
            workload_name=workload_name,
            batch_size=batch_size,
            throughput_output_tok_s=getattr(batch_metrics, 'throughput_output_tok_s', 0.0),
            throughput_total_tok_s=getattr(batch_metrics, 'throughput_total_tok_s', 0.0),
            ttft_p50_ms=getattr(batch_metrics, 'ttft_p50_ms', float("nan")),
            ttft_p95_ms=getattr(batch_metrics, 'ttft_p95_ms', float("nan")),
            ttft_p99_ms=getattr(batch_metrics, 'ttft_p99_ms', float("nan")),
            itl_p50_ms=getattr(batch_metrics, 'itl_p50_ms', float("nan")),
            itl_p95_ms=getattr(batch_metrics, 'itl_p95_ms', float("nan")),
            itl_p99_ms=getattr(batch_metrics, 'itl_p99_ms', float("nan")),
            total_requests=getattr(batch_metrics, 'total_requests', 0),
            successful_requests=getattr(batch_metrics, 'successful_requests', 0),
            failed_requests=getattr(batch_metrics, 'failed_requests', 0),
            total_input_tokens=getattr(batch_metrics, 'total_input_tokens', 0),
            total_output_tokens=getattr(batch_metrics, 'total_output_tokens', 0),
            avg_input_tokens=getattr(batch_metrics, 'avg_input_tokens', 0.0),
            avg_output_tokens=getattr(batch_metrics, 'avg_output_tokens', 0.0),
            goodput=goodput,
            kv_cache=kv_stats,
            error_message=error_message,
        )
        
        self.all_runs.append(stats)
        return stats
    
    def compute_speedups(self, baseline_config: str = "baseline"):
        """Compute speedup vs baseline for all runs."""
        # Build baseline lookup
        baseline_lookup = {}
        for run in self.all_runs:
            if run.config_name == baseline_config:
                key = (run.workload_name, run.batch_size)
                baseline_lookup[key] = run.throughput_output_tok_s
        
        # Compute speedups
        for run in self.all_runs:
            key = (run.workload_name, run.batch_size)
            baseline = baseline_lookup.get(key)
            if baseline and baseline > 0:
                run.speedup_vs_baseline = round(run.throughput_output_tok_s / baseline, 3)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a pandas DataFrame."""
        rows = [run.to_dict() for run in self.all_runs]
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print a formatted summary of benchmark results."""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        df = self.to_dataframe()
        
        if df.empty:
            print("No results to display.")
            return
        
        # Core metrics table
        summary_cols = [
            "config_name", "workload_name", "batch_size",
            "throughput_output_tok_s", "ttft_p50_ms", "itl_p50_ms",
            "goodput_success_rate", "speedup_vs_baseline"
        ]
        available_cols = [c for c in summary_cols if c in df.columns]
        
        print("\n📈 Performance Summary:")
        print("-" * 80)
        print(df[available_cols].to_string(index=False))
        
        # Goodput summary
        if "goodput_success_rate" in df.columns:
            print("\n📊 Goodput Metrics:")
            print("-" * 80)
            goodput_cols = [c for c in df.columns if c.startswith("goodput_")]
            if goodput_cols:
                print(df[["config_name", "workload_name", "batch_size"] + goodput_cols].to_string(index=False))
        
        # KV cache summary
        kv_cols = [c for c in df.columns if c.startswith("kv_")]
        if kv_cols and not df[kv_cols].isna().all().all():
            print("\n💾 KV Cache Statistics:")
            print("-" * 80)
            print(df[["config_name", "workload_name", "batch_size"] + kv_cols].dropna(how='all', subset=kv_cols).to_string(index=False))
        
        # Best configurations per workload
        print("\n🏆 Best Configuration per Workload (by throughput):")
        print("-" * 80)
        for workload in df["workload_name"].unique():
            wl_df = df[df["workload_name"] == workload]
            if len(wl_df) > 0:
                best = wl_df.loc[wl_df["throughput_output_tok_s"].idxmax()]
                print(f"  {workload}: {best['config_name']} (batch={best['batch_size']}) "
                      f"- {best['throughput_output_tok_s']:.1f} tok/s")
    
    def save_results(self, prefix: str = "benchmark"):
        """Save results to CSV files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        df = self.to_dataframe()
        
        # Full results
        full_path = f"{prefix}_full_{timestamp}.csv"
        df.to_csv(full_path, index=False)
        print(f"\n✅ Saved full results to {full_path}")
        
        # Pivot tables per workload
        for workload in df["workload_name"].unique():
            wl_df = df[df["workload_name"] == workload]
            if len(wl_df) > 0 and "throughput_output_tok_s" in wl_df.columns:
                pivot = wl_df.pivot_table(
                    index="batch_size",
                    columns="config_name",
                    values="throughput_output_tok_s",
                    aggfunc="first"
                )
                pivot_path = f"{prefix}_{workload}_{timestamp}.csv"
                pivot.to_csv(pivot_path)
                print(f"✅ Saved {workload} pivot to {pivot_path}")
        
        return full_path
    
    def get_comparison_table(self, metric: str = "throughput_output_tok_s") -> pd.DataFrame:
        """
        Create a comparison table across configurations.
        
        Returns a pivot table with configs as columns, (workload, batch_size) as rows.
        """
        df = self.to_dataframe()
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        
        return df.pivot_table(
            index=["workload_name", "batch_size"],
            columns="config_name",
            values=metric,
            aggfunc="first"
        )
