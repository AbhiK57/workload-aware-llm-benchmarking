"""
Workload generation for LLM inference benchmarking.
Provides diverse prompt sets to test different inference characteristics.

Supports saving/loading workloads to/from a JSON file so that both
vLLM and SGLang notebooks can use identical prompts for fair comparison.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def build_prompt_from_dolly(ex: Dict[str, Any]) -> str:
    """Build a prompt from a Dolly dataset example."""
    inst = ex.get("instruction", "") or ""
    ctx = ex.get("context", "") or ""
    if ctx.strip():
        return f"{inst}\n\nContext:\n{ctx}".strip()
    return inst.strip()


@dataclass
class WorkloadConfig:
    """Configuration for a benchmark workload."""

    name: str
    description: str
    max_tokens: int
    temperature: float

    def __post_init__(self):
        # Validate config
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


WORKLOAD_CONFIGS = {
    "random_short": WorkloadConfig(
        name="random_short",
        description="Short prompts (<100 tokens) - tests raw decode speed",
        max_tokens=128,
        temperature=0.2,
    ),
    "random_medium": WorkloadConfig(
        name="random_medium",
        description="Medium prompts (100-500 tokens) - balanced workload",
        max_tokens=128,
        temperature=0.2,
    ),
    "long_context": WorkloadConfig(
        name="long_context",
        description="Long prompts (500-1500 tokens) - stresses prefill throughput",
        max_tokens=128,
        temperature=0.2,
    ),
    "mixed_context": WorkloadConfig(
        name="mixed_context",
        description="50% synthesized ~6000-tok prompts + 50% real <50-tok prompts, interleaved - extreme head-of-line blocking stress test for chunked-prefill comparison",
        max_tokens=512,  # Long outputs widen the TTFT/ITL gap between chunked vs non-chunked
        temperature=0.2,
    ),
    "shared_prefix": WorkloadConfig(
        name="shared_prefix",
        description="Same prefix, different questions - tests KV cache reuse",
        max_tokens=96,
        temperature=0.0,  # Deterministic for cache testing
    ),
    "repetitive": WorkloadConfig(
        name="repetitive",
        description="Repetitive patterns - tests n-gram speculation",
        max_tokens=128,
        temperature=0.1,  # Low temp for more predictable output
    ),
    "code": WorkloadConfig(
        name="code",
        description="Code generation - tests structured output speculation",
        max_tokens=128,
        temperature=0.0,  # Deterministic code
    ),
    "sharegpt": WorkloadConfig(
        name="sharegpt",
        description="Real ChatGPT conversations (ShareGPT) - diverse real-world prompt distribution",
        max_tokens=512,
        temperature=0.2,
    ),
    "sonnet": WorkloadConfig(
        name="sonnet",
        description="Shakespeare sonnets chunked to target length - natural language with optional shared prefix",
        max_tokens=150,
        temperature=0.2,
    ),
}


class WorkloadGenerator:
    """
    Generates diverse workloads for benchmarking LLM inference.

    Each workload tests different aspects:
    - random_short: Raw decode speed with minimal prefill
    - random_medium: Balanced workload for typical usage
    - long_context: Prefill-heavy workload (all-long prompts)
    - mixed_context: 50% short + 50% long in same batch; best shows chunked-prefill benefit
    - shared_prefix: KV cache reuse / prefix caching
    - repetitive: N-gram speculation effectiveness
    - code: Structured output generation

    Workloads can be saved to and loaded from a JSON file so that both
    the vLLM and SGLang notebooks benchmark against identical prompts.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the workload generator.

        Args:
            model_id: Model ID for tokenizer (used for length estimation)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._workloads: Dict[str, List[str]] = {}
        self._token_counts: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_workloads(self, path: str) -> None:
        """
        Save all prepared workloads and token counts to a JSON file.

        The file can be loaded later with :meth:`load_workloads` to
        guarantee identical prompts across different engine benchmarks.

        Args:
            path: File path to write (e.g. ``benchmark_prompts.json``).
        """
        if not self._workloads:
            raise RuntimeError(
                "No workloads to save. Call prepare_workloads() first."
            )

        payload = {
            "version": 1,
            "workloads": self._workloads,
            "token_counts": self._token_counts,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

        n_total = sum(len(v) for v in self._workloads.values())
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved {len(self._workloads)} workloads "
              f"({n_total} prompts, {size_kb:.1f} KB) → {path}")

    def load_workloads(self, path: str, verbose: bool = True) -> Dict[str, List[str]]:
        """
        Load workloads from a previously saved JSON file.

        This replaces any currently prepared workloads.  Token counts are
        loaded if present; otherwise they are recomputed from the prompts.

        Args:
            path:    File path to read.
            verbose: Print summary after loading.

        Returns:
            Dictionary mapping workload names to prompt lists.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._workloads = payload["workloads"]

        # Token counts may or may not be in the file
        if "token_counts" in payload and payload["token_counts"]:
            self._token_counts = payload["token_counts"]
        else:
            # Recompute token counts from prompts
            self._token_counts = {}
            for name, prompts in self._workloads.items():
                self._token_counts[name] = [
                    len(self.tokenizer.encode(p, add_special_tokens=False))
                    for p in prompts
                ]

        if verbose:
            self._print_workload_stats()

        return self._workloads

    # ------------------------------------------------------------------
    # Prepare (generate from Dolly, ShareGPT, and Sonnet corpora)
    # ------------------------------------------------------------------

    def prepare_workloads(
        self,
        n_random: int = 256,
        n_shared: int = 50,
        n_dataset: int = 256,
        verbose: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Prepare all workloads including Dolly-based, ShareGPT, and Sonnet.

        Args:
            n_random:  Number of prompts for random / repetitive / code workloads
            n_shared:  Number of prompts for shared prefix workload
            n_dataset: Number of prompts for sharegpt and sonnet workloads
            verbose:   Print progress information

        Returns:
            Dictionary mapping workload names to prompt lists
        """
        if verbose:
            print("Loading Dolly dataset...")

        ds = load_dataset("databricks/databricks-dolly-15k", split="train")

        # Build prompts with length info
        all_prompts = []
        for ex in ds:
            prompt = build_prompt_from_dolly(ex)
            if prompt:
                tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                all_prompts.append((prompt, tokens))

        # Sort by length
        all_prompts.sort(key=lambda x: x[1])

        # Split into short/medium/long buckets
        short_prompts = [(p, t) for p, t in all_prompts if t < 100][:n_random]
        medium_prompts = [(p, t) for p, t in all_prompts if 100 <= t < 500][:n_random]
        long_prompts = [(p, t) for p, t in all_prompts if 500 <= t <= 1500][:n_random]

        # Pad if needed
        short_prompts = self._pad_prompts(short_prompts, n_random)
        medium_prompts = self._pad_prompts(medium_prompts, n_random)
        long_prompts = self._pad_prompts(long_prompts, n_random)

        # Mixed workload: alternating synthesized-long / real-short prompts.
        #
        # The LONG side is synthesized by repeating a >500-token base text × 12,
        # producing ~6000-token prefills guaranteed regardless of dataset
        # distribution.  With baseline (chunked prefill ON, 512-tok chunks) a
        # single prefill takes ceil(6000/512) = 12 scheduler steps, creating 12
        # windows where short requests can slip in.  With no_chunked_prefill
        # (--max-num-batched-tokens 8192) the entire 6000-token prefill fits in
        # ONE step, forcing short requests to queue behind it.  That 12:1 ratio
        # makes the head-of-line blocking effect unmistakable.
        #
        # Prompts are INTERLEAVED so every batch slice prompts[:n] has both
        # types regardless of batch size.
        half = n_random // 2

        # 1. Short side — real dataset prompts under 50 tokens
        mixed_short = [(p, t) for p, t in all_prompts if t < 50]
        mixed_short = self._pad_prompts(mixed_short[:half], half)

        # 2. Synthesize extreme long prompts (~6000 tokens each)
        base_long = [p for p, t in all_prompts if t > 500]
        if not base_long:
            raise ValueError(
                "No prompts with >500 tokens found in dataset; "
                "cannot build mixed_context workload."
            )
        mixed_long = []
        for i in range(half):
            base_text = base_long[i % len(base_long)]
            synthetic_text = (base_text + "\n\n") * 12
            tokens = len(self.tokenizer.encode(synthetic_text, add_special_tokens=False))
            mixed_long.append((synthetic_text, tokens))

        # 3. Interleave: [long_0, short_0, long_1, short_1, ...]
        mixed = [item for pair in zip(mixed_long, mixed_short) for item in pair]

        # Store token counts for statistics
        self._token_counts["random_short"] = [t for _, t in short_prompts]
        self._token_counts["random_medium"] = [t for _, t in medium_prompts]
        self._token_counts["long_context"] = [t for _, t in long_prompts]
        self._token_counts["mixed_context"] = [t for _, t in mixed]

        # Shared prefix workload (tests prefix caching / RadixAttention)
        shared_prefix_prompts, shared_tokens = self._create_shared_prefix_workload(ds, n_shared)
        self._token_counts["shared_prefix"] = shared_tokens

        # Repetitive prompts (good for n-gram speculation)
        repetitive_prompts = self._create_repetitive_workload(n_random)
        self._token_counts["repetitive"] = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in repetitive_prompts
        ]

        # Code prompts (structured output, good for draft model speculation)
        code_prompts = self._create_code_workload(n_random)
        self._token_counts["code"] = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in code_prompts
        ]

        # ShareGPT workload (real ChatGPT conversations)
        sharegpt_prompts = []
        try:
            from .datasets import load_sharegpt  # noqa: PLC0415

            if verbose:
                print("Loading ShareGPT dataset...")
            samples = load_sharegpt(
                num_prompts=n_dataset,
                tokenizer=self.tokenizer,
                input_len=512,
                output_len=128,
                seed=42,
            )
            sharegpt_prompts = [s.prompt for s in samples]
            self._token_counts["sharegpt"] = [s.input_tokens for s in samples]
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Could not load ShareGPT dataset: {e}")
                print("  The 'sharegpt' workload will be empty.")

        # Sonnet workload (Shakespeare sonnets chunked to target length)
        sonnet_prompts = []
        try:
            from .datasets import load_sonnet  # noqa: PLC0415

            if verbose:
                print("Loading Sonnet dataset...")
            samples = load_sonnet(
                num_prompts=n_dataset,
                tokenizer=self.tokenizer,
                input_len=550,
                output_len=150,
                prefix_len=0,
                seed=42,
            )
            sonnet_prompts = [s.prompt for s in samples]
            self._token_counts["sonnet"] = [s.input_tokens for s in samples]
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Could not load Sonnet dataset: {e}")
                print("  The 'sonnet' workload will be empty.")

        self._workloads = {
            "random_short": [p for p, _ in short_prompts],
            "random_medium": [p for p, _ in medium_prompts],
            "long_context": [p for p, _ in long_prompts],
            "mixed_context": [p for p, _ in mixed],
            "shared_prefix": shared_prefix_prompts,
            "repetitive": repetitive_prompts,
            "code": code_prompts,
            "sharegpt": sharegpt_prompts,
            "sonnet": sonnet_prompts,
        }

        if verbose:
            self._print_workload_stats()

        return self._workloads

    def _pad_prompts(
        self,
        prompts: List[tuple],
        target_count: int,
    ) -> List[tuple]:
        """Pad prompt list to target count by cycling."""
        if len(prompts) == 0:
            return []
        original_len = len(prompts)
        while len(prompts) < target_count:
            prompts.append(prompts[len(prompts) % original_len])
        return prompts

    def _create_shared_prefix_workload(
        self,
        ds,
        n_shared: int,
        max_prefix_tokens: int = 400,  # Limit prefix size for memory safety
    ) -> tuple:
        """Create shared prefix workload for testing KV cache reuse.

        Args:
            ds: Dataset to pull context from
            n_shared: Number of prompts to generate
            max_prefix_tokens: Maximum tokens for the shared prefix (default 400)
                              This prevents OOM at high batch sizes.
        """
        # Find a medium-length context (not too long to avoid OOM at batch=128)
        # Target: 300-500 tokens for a good balance of prefix caching test vs memory
        mid_ctx_examples = ds.filter(
            lambda x: x.get("context") and 800 < len(x["context"]) < 2000
        )

        if len(mid_ctx_examples) > 0:
            prefix = mid_ctx_examples[0]["context"].strip()
        else:
            # Fallback: concatenate a few prompts
            prefix = "\n\n".join([build_prompt_from_dolly(ds[i]) for i in range(5)])

        # Truncate prefix to max_prefix_tokens to prevent OOM at high batch sizes
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        if len(prefix_tokens) > max_prefix_tokens:
            prefix_tokens = prefix_tokens[:max_prefix_tokens]
            prefix = self.tokenizer.decode(prefix_tokens)

        questions = [
            "Summarize the key points in 2-3 sentences.",
            "Extract all names and organizations mentioned.",
            "What is the main argument or claim?",
            "List any dates, numbers, or statistics.",
            "Rewrite this in simpler language for a 10-year-old.",
            "What are 3 follow-up questions someone might ask?",
            "Create a bullet-point outline.",
            "What are potential counterarguments?",
            "Identify assumptions made in the text.",
            "Write a tweet summarizing this.",
        ]

        prompts = [
            f"{prefix}\n\nQuestion: {questions[i % len(questions)]}"
            for i in range(n_shared)
        ]

        token_counts = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in prompts
        ]

        return prompts, token_counts

    def _create_repetitive_workload(self, n_random: int) -> List[str]:
        """Create repetitive prompts for testing n-gram speculation."""
        base_prompts = [
            "Write a list of 20 items: 1. Apple, 2. Banana, 3. Cherry, continue the pattern...",
            "Count from 1 to 50, writing each number on a new line.",
            "Repeat the phrase 'Hello World' 10 times, numbering each repetition.",
            "Write the days of the week, then the months of the year, then the seasons.",
            "Generate a multiplication table for numbers 1-10.",
            "List the first 20 prime numbers with brief explanations.",
            "Write the alphabet, then write it backwards.",
            "Create a pattern: AABB, AABB, AABB... continue for 10 lines.",
        ]

        prompts = base_prompts * (n_random // len(base_prompts) + 1)
        return prompts[:n_random]

    def _create_code_workload(self, n_random: int) -> List[str]:
        """Create code prompts for testing structured output generation."""
        base_prompts = [
            "Write a Python function to calculate fibonacci numbers recursively.",
            "Write a Python class for a binary search tree with insert and search methods.",
            "Write a JavaScript function to debounce another function.",
            "Write a SQL query to find the top 10 customers by total order value.",
            "Write a Python decorator that measures function execution time.",
            "Write a bash script that finds all .py files and counts lines of code.",
            "Write a Python function to merge two sorted lists.",
            "Write a JavaScript async function that fetches data with retry logic.",
        ]

        prompts = base_prompts * (n_random // len(base_prompts) + 1)
        return prompts[:n_random]

    def _print_workload_stats(self):
        """Print statistics about prepared workloads."""
        print("\n📊 Workload Statistics:")
        print("-" * 60)

        for name, prompts in self._workloads.items():
            tokens = self._token_counts.get(name, [])
            if tokens:
                import numpy as np

                print(f"  {name}:")
                print(f"    Prompts: {len(prompts)}")
                print(
                    f"    Tokens - min: {min(tokens)}, max: {max(tokens)}, "
                    f"mean: {np.mean(tokens):.1f}, median: {np.median(tokens):.1f}"
                )
        print("-" * 60)

    def get_workload(self, name: str) -> List[str]:
        """Get a specific workload by name."""
        if not self._workloads:
            self.prepare_workloads()
        prompts = self._workloads.get(name, [])
        if not prompts and name in WORKLOAD_CONFIGS:
            print(
                f"  [WARNING] Workload '{name}' is defined but has no prompts loaded. "
                f"Call prepare_dataset('{name}', ...) first."
            )
        return prompts

    def get_token_counts(self, name: str) -> List[int]:
        """Get token counts for a specific workload."""
        return self._token_counts.get(name, [])

    def get_workload_config(self, name: str) -> WorkloadConfig:
        """Get configuration for a specific workload."""
        return WORKLOAD_CONFIGS.get(name)

    def prepare_dataset(
        self,
        dataset_name: str,
        num_prompts: int,
        input_len: int = 512,
        output_len: int = 128,
        prefix_len: int = 0,
        seed: int = 42,
    ) -> List[str]:
        """
        Load prompts from a named dataset and store them as a workload.

        Uses the dataset loaders in :mod:`llm_benchmark.datasets` to populate
        ``self._workloads[dataset_name]`` so that ``get_workload(dataset_name)``
        works after this call.

        Args:
            dataset_name: One of ``random``, ``sharegpt``, ``sonnet``, ``dolly``.
            num_prompts:  Number of prompts to load.
            input_len:    Target / filter input length (tokens).
            output_len:   Target output length (tokens).
            prefix_len:   Shared prefix length (sonnet only).
            seed:         RNG seed for reproducibility.

        Returns:
            List of prompt strings (also stored in ``self._workloads``).
        """
        from .datasets import load_dataset_prompts  # noqa: PLC0415

        samples = load_dataset_prompts(
            dataset_name=dataset_name,
            num_prompts=num_prompts,
            tokenizer=self.tokenizer,
            input_len=input_len,
            output_len=output_len,
            prefix_len=prefix_len,
            seed=seed,
        )
        prompts = [s.prompt for s in samples]
        token_counts = [s.input_tokens for s in samples]

        self._workloads[dataset_name] = prompts
        self._token_counts[dataset_name] = token_counts
        return prompts

    def list_workloads(self) -> List[str]:
        """List all available workload names."""
        return list(WORKLOAD_CONFIGS.keys())
