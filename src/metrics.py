"""
Metrics collection for LLM inference benchmarking.
Handles streaming requests and captures detailed timing information.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np


@dataclass
class RequestMetrics:
    """
    Detailed metrics for a single inference request.
    
    Attributes:
        request_id: Unique identifier for the request
        ttft_s: Time to first token (seconds)
        itl_s: Inter-token latencies (seconds)
        total_s: Total request duration (seconds)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (input + output)
        start_time: Unix timestamp when request started
        end_time: Unix timestamp when request completed
        success: Whether the request completed successfully
        error_message: Error message if request failed
    """
    
    request_id: str = ""
    ttft_s: float = float("nan")
    itl_s: List[float] = field(default_factory=list)
    total_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = True
    error_message: str = ""
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate output tokens per second."""
        if self.total_s <= 0:
            return 0.0
        return self.output_tokens / self.total_s
    
    @property
    def generation_time_s(self) -> float:
        """Time spent generating (excluding TTFT)."""
        if np.isnan(self.ttft_s):
            return self.total_s
        return max(0.0, self.total_s - self.ttft_s)
    
    @property
    def mean_itl_ms(self) -> float:
        """Mean inter-token latency in milliseconds."""
        if not self.itl_s:
            return float("nan")
        return float(np.mean(self.itl_s) * 1000.0)
    
    @property
    def p50_itl_ms(self) -> float:
        """P50 inter-token latency in milliseconds."""
        if not self.itl_s:
            return float("nan")
        return float(np.percentile(self.itl_s, 50) * 1000.0)
    
    @property
    def p99_itl_ms(self) -> float:
        """P99 inter-token latency in milliseconds."""
        if not self.itl_s:
            return float("nan")
        return float(np.percentile(self.itl_s, 99) * 1000.0)


@dataclass
class BatchMetrics:
    """
    Aggregated metrics for a batch of requests.
    
    Includes throughput, latency distributions, and efficiency metrics.
    """
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing
    wall_time_s: float = 0.0
    
    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Throughput metrics
    throughput_total_tok_s: float = 0.0
    throughput_output_tok_s: float = 0.0
    throughput_requests_s: float = 0.0
    
    # Latency metrics (all in ms)
    ttft_mean_ms: float = float("nan")
    ttft_p50_ms: float = float("nan")
    ttft_p90_ms: float = float("nan")
    ttft_p95_ms: float = float("nan")
    ttft_p99_ms: float = float("nan")
    ttft_min_ms: float = float("nan")
    ttft_max_ms: float = float("nan")
    
    itl_mean_ms: float = float("nan")
    itl_p50_ms: float = float("nan")
    itl_p90_ms: float = float("nan")
    itl_p95_ms: float = float("nan")
    itl_p99_ms: float = float("nan")
    itl_min_ms: float = float("nan")
    itl_max_ms: float = float("nan")
    
    request_latency_mean_ms: float = float("nan")
    request_latency_p50_ms: float = float("nan")
    request_latency_p90_ms: float = float("nan")
    request_latency_p95_ms: float = float("nan")
    request_latency_p99_ms: float = float("nan")
    
    # Average tokens per request
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    
    # Efficiency metrics
    goodput_tok_s: float = 0.0  # Tokens from successful requests only
    efficiency_ratio: float = 0.0  # successful / total requests
    
    # Individual request metrics (for detailed analysis)
    request_metrics: List[RequestMetrics] = field(default_factory=list)

    # KV cache snapshot collected mid-flight (set by runner, not by aggregate)
    kv_cache_snapshot: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding request_metrics for summary)."""
        return {
            k: v for k, v in self.__dict__.items()
            if k != "request_metrics"
        }


def tokenize_chat(tokenizer, messages: List[Dict[str, str]]) -> int:
    """
    Count tokens in a chat message list.
    
    Args:
        tokenizer: HuggingFace tokenizer
        messages: List of chat messages
        
    Returns:
        Token count
    """
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        return len(ids)
    except Exception:
        # Fallback for tokenizers without chat template
        joined = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return len(tokenizer.encode(joined))


async def stream_chat_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    tokenizer,
    max_tokens: int = 128,
    temperature: float = 0.2,
    request_id: str = "",
    timeout_s: int = 600,
    min_tokens: Optional[int] = None,
) -> RequestMetrics:
    """
    Send a streaming chat completion request and collect metrics.

    Args:
        session: aiohttp client session
        base_url: Server base URL (e.g., "http://localhost:8000/v1")
        model: Model name
        messages: Chat messages
        tokenizer: Tokenizer for token counting
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        request_id: Optional request identifier
        timeout_s: Request timeout in seconds
        min_tokens: If set, forces the server to generate at least this many tokens.
                    When min_tokens == max_tokens, output length is fixed exactly.

    Returns:
        RequestMetrics with detailed timing information
    """
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    input_tokens = tokenize_chat(tokenizer, messages)
    
    t0 = time.perf_counter()
    start_time = time.time()
    first_content_time = None
    last_content_time = None
    itl_s: List[float] = []
    out_text_parts: List[str] = []

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s, connect=10, sock_connect=10)
        ) as resp:
            resp.raise_for_status()

            buffer = ""
            done = False
            async for chunk in resp.content.iter_chunked(4096):
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(data)
                    except Exception:
                        continue

                    try:
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content", None)
                    except Exception:
                        content = None

                    if content:
                        now = time.perf_counter()
                        out_text_parts.append(content)
                        if first_content_time is None:
                            first_content_time = now
                            last_content_time = now
                        else:
                            itl_s.append(now - last_content_time)
                            last_content_time = now
                if done:
                    break

        t1 = time.perf_counter()
        end_time = time.time()
        
        out_text = "".join(out_text_parts)
        try:
            output_tokens = len(tokenizer.encode(out_text))
        except Exception:
            output_tokens = max(1, len(out_text) // 4)

        ttft_s = float("nan") if first_content_time is None else (first_content_time - t0)
        total_s = t1 - t0

        return RequestMetrics(
            request_id=request_id,
            ttft_s=ttft_s,
            itl_s=itl_s,
            total_s=total_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            start_time=start_time,
            end_time=end_time,
            success=True,
            error_message="",
        )
        
    except Exception as e:
        t1 = time.perf_counter()
        end_time = time.time()
        
        return RequestMetrics(
            request_id=request_id,
            ttft_s=float("nan"),
            itl_s=[],
            total_s=t1 - t0,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
            start_time=start_time,
            end_time=end_time,
            success=False,
            error_message=str(e),
        )


def aggregate_metrics(
    metrics_list: List[RequestMetrics],
    wall_time_s: float
) -> BatchMetrics:
    """
    Aggregate individual request metrics into batch statistics.
    
    Args:
        metrics_list: List of RequestMetrics from individual requests
        wall_time_s: Total wall clock time for the batch
        
    Returns:
        BatchMetrics with aggregated statistics
    """
    if not metrics_list:
        return BatchMetrics()
    
    successful = [m for m in metrics_list if m.success]
    failed = [m for m in metrics_list if not m.success]
    
    total_input = sum(m.input_tokens for m in metrics_list)
    total_output = sum(m.output_tokens for m in successful)
    total_tokens = total_input + total_output
    
    # Throughput calculations
    throughput_total = total_tokens / max(wall_time_s, 1e-9)
    throughput_output = total_output / max(wall_time_s, 1e-9)
    throughput_requests = len(metrics_list) / max(wall_time_s, 1e-9)
    
    # TTFT statistics
    ttfts = [m.ttft_s for m in successful if not np.isnan(m.ttft_s)]
    ttft_stats = _compute_latency_stats(ttfts)
    
    # ITL statistics (flattened across all requests)
    itls = [x for m in successful for x in m.itl_s]
    itl_stats = _compute_latency_stats(itls)
    
    # Request latency statistics
    request_latencies = [m.total_s for m in successful]
    request_latency_stats = _compute_latency_stats(request_latencies)
    
    # Goodput: tokens from successful requests
    goodput_tokens = sum(m.output_tokens for m in successful)
    goodput = goodput_tokens / max(wall_time_s, 1e-9)
    
    # Efficiency ratio
    efficiency = len(successful) / len(metrics_list) if metrics_list else 0.0
    
    return BatchMetrics(
        total_requests=len(metrics_list),
        successful_requests=len(successful),
        failed_requests=len(failed),
        wall_time_s=round(wall_time_s, 3),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        throughput_total_tok_s=round(throughput_total, 2),
        throughput_output_tok_s=round(throughput_output, 2),
        throughput_requests_s=round(throughput_requests, 3),
        ttft_mean_ms=ttft_stats.get("mean_ms", float("nan")),
        ttft_p50_ms=ttft_stats.get("p50_ms", float("nan")),
        ttft_p90_ms=ttft_stats.get("p90_ms", float("nan")),
        ttft_p95_ms=ttft_stats.get("p95_ms", float("nan")),
        ttft_p99_ms=ttft_stats.get("p99_ms", float("nan")),
        ttft_min_ms=ttft_stats.get("min_ms", float("nan")),
        ttft_max_ms=ttft_stats.get("max_ms", float("nan")),
        itl_mean_ms=itl_stats.get("mean_ms", float("nan")),
        itl_p50_ms=itl_stats.get("p50_ms", float("nan")),
        itl_p90_ms=itl_stats.get("p90_ms", float("nan")),
        itl_p95_ms=itl_stats.get("p95_ms", float("nan")),
        itl_p99_ms=itl_stats.get("p99_ms", float("nan")),
        itl_min_ms=itl_stats.get("min_ms", float("nan")),
        itl_max_ms=itl_stats.get("max_ms", float("nan")),
        request_latency_mean_ms=request_latency_stats.get("mean_ms", float("nan")),
        request_latency_p50_ms=request_latency_stats.get("p50_ms", float("nan")),
        request_latency_p90_ms=request_latency_stats.get("p90_ms", float("nan")),
        request_latency_p95_ms=request_latency_stats.get("p95_ms", float("nan")),
        request_latency_p99_ms=request_latency_stats.get("p99_ms", float("nan")),
        avg_input_tokens=round(total_input / len(metrics_list), 1),
        avg_output_tokens=round(total_output / max(len(successful), 1), 1),
        goodput_tok_s=round(goodput, 2),
        efficiency_ratio=round(efficiency, 4),
        request_metrics=metrics_list,
    )


def _compute_latency_stats(latencies_s: List[float]) -> Dict[str, float]:
    """Compute latency statistics from a list of values in seconds."""
    if not latencies_s:
        return {}
    
    arr = np.array(latencies_s) * 1000.0  # Convert to ms
    
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
    }
