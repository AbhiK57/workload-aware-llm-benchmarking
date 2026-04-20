# workload-aware-llm-benchmarking

Benchmarking suite and analysis pipeline for the paper
**"Workload Matters: Benchmarking LLM Inference Optimizations in Modern Serving Systems"**
(Abhinav Khanduja, University of Illinois Urbana-Champaign).

This repository contains everything needed to reproduce the paper's results and to
extend the suite to new engines, configurations, or workloads.

---

## 1. What this project is

Modern LLM serving frameworks (vLLM, SGLang) ship with a long list of advanced
optimizations — speculative decoding, chunked prefill, RadixAttention / automatic
prefix caching, `torch.compile`-based JIT fusion, and so on. Each of these is
marketed as a throughput or latency win, but in practice each one behaves very
differently depending on the *shape* of the traffic hitting the server.

This project evaluates those optimizations under **seven engineered workloads**
and **two external datasets** that each isolate a different inference bottleneck,
rather than using a single benchmark dataset. The goal is to map each optimization
to the prompt distributions where it actually pays off — and to expose the regimes
where it silently hurts.

### What's evaluated

| Engine  | Configurations                                                                 |
|---------|--------------------------------------------------------------------------------|
| vLLM    | `baseline` (chunked prefill, 512-token chunks), `spec_draft_1b` (Llama-3.2-1B draft, 3 spec tokens), `spec_ngram` (prompt lookup, 5 tokens, win=4), `no_chunked_prefill` (`--max-num-batched-tokens 8192`) |
| SGLang  | `baseline` (RadixAttention ON), `no_radix`, `torch_compile`                    |

Target model: **Llama-3.2-3B-Instruct** on a single **NVIDIA A100** at
90% GPU memory utilization (85% when a draft model is co-resident).
Batch sizes: **B ∈ {1, 8, 32, 64}**.

### Workloads

Prompts are engineered from the Databricks Dolly-15k dataset so each distribution
targets a specific system bottleneck:

| Workload         | Mean tokens | Target bottleneck                     |
|------------------|------------:|---------------------------------------|
| `random_short`   |           4 | Raw decode speed                      |
| `random_medium`  |         101 | Balanced prefill / decode             |
| `long_context`   |         517 | Prefill throughput                    |
| `mixed_context`  |        3056 | Scheduler head-of-line blocking       |
| `shared_prefix`  |         321 | KV cache reuse                        |
| `repetitive`     |          15 | N-gram speculation                    |
| `code`           |          12 | Structured-output speculation         |

Plus four external datasets evaluated at fixed concurrency (B=32, 64 prompts each):
**ShareGPT** (multi-turn chat), **Dolly** (instruction-following),
**Sonnet** (fixed-length poetry), and **Random** (uniform synthetic).

### Metrics collected

- **Throughput** — output tok/s, total tok/s, requests/s
- **Latency** — TTFT p50/p95/p99, ITL p50/p95/p99
- **Goodput** — successful tokens/s under an SLA, success rate, SLA compliance rate across strict / moderate / relaxed tiers
- **KV cache** — utilization, prefix cache hit rate (when exposed by the engine)

---

## 2. Repository layout

```
workload-aware-llm-benchmarking/
├── llm_benchmark/                      # Python package — the core suite
│   ├── __init__.py
│   ├── benchmark_core.py               # GPU detection, dtype selection, shared config
│   ├── server.py                       # ServerProcess — spawns vLLM / SGLang servers
│   ├── workloads.py                    # WorkloadGenerator + 9 WORKLOAD_CONFIGS
│   ├── datasets.py                     # ShareGPT / Sonnet / Dolly / Random loaders
│   ├── metrics.py                      # Streaming client, TTFT/ITL percentiles
│   ├── stats.py                        # Goodput + KV cache stats collection
│   └── runner.py                       # BenchmarkRunner, VLLM_ / SGLANG_TEST_CONFIGS
│
├── vLLM_benchmark_refactored.ipynb     # Colab — runs all 4 vLLM configs
├── SGLang_benchmark.ipynb              # Colab — runs all 3 SGLang configs
├── analysis_figures.ipynb              # Generates every figure/table in the paper
│
├── results/                            # CSV outputs (one per config + combined)
└── figures/                            # JPG/PNG figures used in the paper
```

---

## 3. Quick start — reproducing the paper

Everything in the paper is reproducible from two Colab notebooks plus an analysis
notebook. You do not need a local GPU; Colab's A100 runtime is enough.

### Prerequisites

- A Google Colab account with access to an **A100** runtime (Colab Pro / Pro+).
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct` and
  `meta-llama/Llama-3.2-1B-Instruct`, plus an HF token.

### Step 1 — Run the vLLM benchmark notebook

1. Open `vLLM_benchmark_refactored.ipynb` in Colab.
2. Runtime → Change runtime type → **A100 GPU**, High-RAM.
3. Upload the `llm_benchmark/` folder into the Colab file tree (drag and drop, or
   `!git clone` the repo and move the folder). The notebook verifies the folder
   is present before running anything.
4. Paste your Hugging Face token into the auth cell (`os.environ["HF_TOKEN"] = ...`).
5. Run cells top to bottom. The notebook will:
   - install pinned versions of `vllm`, `transformers`, `aiohttp`, etc.,
   - download Llama-3.2-3B and Llama-3.2-1B,
   - prepare the seven engineered workloads from Dolly-15k and cache them to
     `benchmark_prompts.json` (prompt_dataset.json in this repo) (so the SGLang notebook sees *identical* prompts),
   - spawn a fresh vLLM server per configuration, hit it across B ∈ {1, 8, 32, 64},
     tear it down, and move on.

**Outputs (reproducing paper results):**
- `results_baseline.csv`
- `results_spec_draft_1b.csv`
- `results_spec_ngram.csv`
- `results_no_chunked_prefill.csv`
- `vllm_all_results.csv` (combined)
- `benchmark_prompts.json/prompt_dataset.json` (shared prompt cache)

Download these at the end of the notebook (an `Export All Results` cell calls
`google.colab.files.download` for each).

### Step 2 — Run the SGLang benchmark notebook

1. Open `SGLang_benchmark.ipynb` on a **fresh A100 runtime**. SGLang and vLLM
   cannot co-exist in the same kernel — don't try to run them back-to-back
   without restarting.
2. Upload the same `llm_benchmark/` folder **and** the `benchmark_prompts.json`
   file produced by the vLLM notebook. The SGLang notebook detects the file and
   loads identical prompts instead of regenerating, guaranteeing a fair cross-engine
   comparison.
3. Paste your HF token and run top to bottom.

**Outputs:**
- `results_sglang_baseline.csv`
- `results_sglang_no_radix.csv`
- `results_sglang_torch_compile.csv`
- `sglang_all_results.csv` (combined)


### Approximate runtime

| Step                       | Time (A100 High-RAM) |
|----------------------------|----------------------|
| vLLM notebook (4 configs)  | ~2–3 hours           |
| SGLang notebook (3 configs)| ~1.5–2 hours         |
| Analysis notebook          | ~1 minute            |

Batch-size-64 runs on `long_context` / `code` will fail on some configs with
out-of-memory; this is expected and reproduces the Batch-64 Memory Limits finding
from §7 of the paper.

---

## 4. Using the source code for your own benchmarks

The `llm_benchmark/` package is designed to be used independently of the Colab
notebooks. You can run it locally on any machine with a CUDA GPU.

### Install

```bash
git clone https://github.com/AbhiK57/workload-aware-llm-benchmarking.git
cd workload-aware-llm-benchmarking

# Pick ONE — vLLM and SGLang conflict in the same env.
pip install "vllm>=0.6.3"
# OR
pip install "sglang[all]>=0.3.0"

pip install "transformers>=4.44.2" "tokenizers>=0.19.1" "datasets" \
            "aiohttp" "nest_asyncio" "huggingface_hub" "pandas" \
            "seaborn" "matplotlib" "GPUtil" "gpustat" "nvidia-ml-py"
```

### Minimal example — full default sweep

```python
from llm_benchmark import benchmark

# Runs every VLLM_TEST_CONFIG across every WORKLOAD_CONFIG,
# at batch sizes [1, 8, 32, 64], with a fresh server per config.
df = benchmark()   # returns a pandas DataFrame and writes CSVs
```

### Custom sweep

```python
from llm_benchmark import BenchmarkRunner, TestConfig

runner = BenchmarkRunner(
    model_id   = "meta-llama/Llama-3.2-3B-Instruct",
    batch_sizes = [1, 8, 32],
    workloads   = ["random_short", "code", "shared_prefix"],
    configs     = [
        TestConfig(
            name        = "my_custom_vllm",
            description = "vLLM with a beefier batched-token budget",
            engine      = "vllm",
            gpu_mem_util = 0.90,
            extra_args  = ["--max-num-batched-tokens", "16384"],
        ),
    ],
)
df = runner.run()
df.to_csv("my_results.csv", index=False)
```

### Adding a new workload

Open `llm_benchmark/workloads.py` and add an entry to `WORKLOAD_CONFIGS`:

```python
WORKLOAD_CONFIGS["my_workload"] = WorkloadConfig(
    name        = "my_workload",
    description = "Prompts with a distribution I care about",
    max_tokens  = 256,
    temperature = 0.2,
)
```

Then add a generator method on `WorkloadGenerator` (follow the pattern of
`_create_code_workload` for synthetic prompts or the `mixed_context` branch of
`prepare_workloads` for dataset-derived prompts). Prompts are cached to
`benchmark_prompts.json` so both engines see identical inputs.

### Adding a new engine configuration

Append to `VLLM_TEST_CONFIGS` or `SGLANG_TEST_CONFIGS` in
`llm_benchmark/runner.py`:

```python
TestConfig(
    name        = "spec_eagle",
    description = "EAGLE speculative decoding",
    engine      = "vllm",
    spec_config = '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 4}',
    gpu_mem_util = 0.85,
    best_workloads = ["code", "random_medium"],
),
```

`ServerProcess` in `llm_benchmark/server.py` handles spawning the server,
waiting on the health endpoint, and cleanly tearing it down between configs so
every run starts with an empty KV cache.

### Supporting a new engine entirely

The interface a new engine needs is small:

1. Add an `engine="my_engine"` branch in `ServerProcess._build_command()`.
2. Point the OpenAI-compatible chat endpoint at the right port (default 8000).
3. Optionally expose a Prometheus metrics endpoint so `stats.fetch_server_metrics`
   can pull KV cache stats.

Everything else — request streaming, TTFT/ITL computation, goodput — is
engine-agnostic.

---

## 5. Output schema

Each row in the result CSVs is one `(config, workload, batch_size)` combination.
Key columns:

| Column                            | Meaning                                             |
|-----------------------------------|-----------------------------------------------------|
| `config_name`, `workload_name`, `batch_size` | Run identity                             |
| `throughput_output_tok_s`, `throughput_total_tok_s` | Output-only and total token rate  |
| `ttft_p50_ms`, `ttft_p95_ms`, `ttft_p99_ms` | Time-to-first-token percentiles             |
| `itl_p50_ms`, `itl_p95_ms`, `itl_p99_ms` | Per-step inter-token latency percentiles       |
| `goodput_goodput_tok_s`           | Tokens/s from requests that met the TTFT SLO        |
| `goodput_sla_compliance_rate`     | Fraction of requests meeting the SLA                |
| `goodput_effective_batch_utilization` | Realized vs. requested concurrency              |
| `speedup_vs_baseline`             | Throughput ratio against the engine's baseline      |
| `successful_requests`, `failed_requests` | OOM / error tracking                         |

The analysis notebook joins the two combined CSVs on these columns to produce
every cross-engine figure.

---

## 6. Acknowledgments

This suite is built on top of the excellent open-source work of the
[vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang) communities, and uses the
[Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
and ShareGPT datasets for prompt construction.
