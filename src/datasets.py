"""
Dataset loaders for LLM serving benchmarks.

Supports:
  - random   : synthetic prompts with exact token-length control
  - sharegpt : real ChatGPT conversations (first human turn)
  - sonnet   : Shakespeare sonnets chunked to target length
  - dolly    : Databricks Dolly-15k (reuses workloads.py logic)

Each loader returns List[PromptSample] where PromptSample carries the
prompt string, its measured input token count, and the expected output
token count (used for fixed-length output benchmarking).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PromptSample:
    """A single benchmark prompt with associated token metadata."""
    prompt: str
    input_tokens: int
    output_tokens: int  # expected / target; used as max_tokens when running


# ---------------------------------------------------------------------------
# Random synthetic dataset
# ---------------------------------------------------------------------------

def load_random(
    num_prompts: int,
    tokenizer,
    input_len: int = 512,
    output_len: int = 128,
    seed: int = 42,
) -> List[PromptSample]:
    """
    Generate synthetic prompts of exactly *input_len* tokens by sampling
    uniformly from the tokenizer vocabulary.

    Because the decoded text may differ from input_len after round-tripping
    through the tokenizer (special tokens, BPE merges), we over-sample and
    re-encode to hit the target precisely.
    """
    rng = random.Random(seed)
    vocab_ids = list(tokenizer.get_vocab().values())
    # Filter out special tokens that might terminate decoding early
    special = set(tokenizer.all_special_ids)
    vocab_ids = [v for v in vocab_ids if v not in special]

    samples: List[PromptSample] = []
    for _ in range(num_prompts):
        # Sample token ids, decode, then re-encode to measure actual length
        ids = [rng.choice(vocab_ids) for _ in range(input_len)]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        samples.append(PromptSample(
            prompt=text,
            input_tokens=actual,
            output_tokens=output_len,
        ))
    return samples


# ---------------------------------------------------------------------------
# ShareGPT dataset
# ---------------------------------------------------------------------------

def load_sharegpt(
    num_prompts: int,
    tokenizer,
    input_len: Optional[int] = None,
    output_len: Optional[int] = None,
    seed: int = 42,
) -> List[PromptSample]:
    """
    Load prompts from the ShareGPT dataset (real ChatGPT conversations).

    Uses the first human turn as the prompt and, when *output_len* is None,
    estimates the expected output length from the first GPT response.

    If *input_len* is provided, only samples within ±50 % of that length
    are kept; remaining slots are filled by cycling over what was collected.

    Dataset: ``Aeala/ShareGPT_Vicuna_unfiltered`` (HuggingFace Datasets).
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        "Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    raw: List[PromptSample] = []
    for idx in indices:
        sample = ds[idx]
        convs = sample.get("conversations") or []

        human = next((c["value"] for c in convs
                      if c.get("from") in ("human", "user")), None)
        gpt   = next((c["value"] for c in convs
                      if c.get("from") in ("gpt", "assistant")), None)

        if not human:
            continue

        in_toks  = len(tokenizer.encode(human, add_special_tokens=False))
        out_toks = (output_len if output_len is not None
                    else (len(tokenizer.encode(gpt, add_special_tokens=False))
                          if gpt else 128))

        # Length filter ± 50 %
        if input_len is not None:
            lo, hi = input_len * 0.5, input_len * 1.5
            if not (lo <= in_toks <= hi):
                continue

        raw.append(PromptSample(prompt=human,
                                input_tokens=in_toks,
                                output_tokens=out_toks))
        if len(raw) >= num_prompts * 3:   # collect extra, then subsample
            break

    if not raw:
        raise ValueError(
            "ShareGPT: no samples matched the length filter. "
            "Try a wider input_len range or set input_len=None."
        )

    # Pad / cycle to exactly num_prompts
    while len(raw) < num_prompts:
        raw.extend(raw[: num_prompts - len(raw)])
    return raw[:num_prompts]


# ---------------------------------------------------------------------------
# Sonnet dataset
# ---------------------------------------------------------------------------

# A self-contained corpus of Shakespeare's Sonnets 1–20 (public domain).
# Repeating this text lets us hit any target input length without network
# access beyond the initial tokenizer load.
_SONNET_CORPUS = """\
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou, contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament
And only herald to the gaudy spring,
Within thine own bud buriest thy content
And, tender churl, makest waste in niggarding.
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.

Unthrifty loveliness, why dost thou spend
Upon thy self thy beauty's legacy?
Nature's bequest gives nothing, but doth lend,
And being frank she lends to those are free:
Then, beauteous niggard, why dost thou abuse
The bounteous largess given thee to give?
Profitless usurer, why dost thou use
So great a sum of sums, yet canst not live?
For having traffic with thy self alone,
Thou of thy self thy sweet self dost deceive:
Then how when nature calls thee to be gone,
What acceptable audit canst thou leave?
Thy unused beauty must be tombed with thee,
Which, used, lives th' executor to be.

Look in thy glass and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb
Of his self-love, to stop posterity?
Thou art thy mother's glass and she in thee
Calls back the lovely April of her prime;
So thou through windows of thine age shalt see,
Despite of wrinkles, this thy golden time.
But if thou live remember'd not to be,
Die single and thine image dies with thee.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.

When I do count the clock that tells the time,
And see the brave day sunk in hideous night;
When I behold the violet past prime,
And sable curls all silver'd o'er with white;
When lofty trees I see barren of leaves,
Which erst from heat did canopy the herd,
And summer's green all girded up in sheaves
Borne on the bier with white and bristly beard,
Then of thy beauty do I question make,
That thou among the wastes of time must go,
Since sweets and beauties do themselves forsake
And die as fast as they see others grow;
And nothing 'gainst Time's scythe can make defence
Save breed, to brave him when he takes thee hence.
"""


def load_sonnet(
    num_prompts: int,
    tokenizer,
    input_len: int = 550,
    output_len: int = 150,
    prefix_len: int = 0,
    seed: int = 42,
) -> List[PromptSample]:
    """
    Create prompts by chunking the Shakespeare sonnet corpus to *input_len*
    tokens.  When *prefix_len* > 0 every prompt shares an identical prefix of
    that length followed by a unique continuation — this tests prefix-cache
    reuse in the same way as the ``shared_prefix`` workload.

    The corpus is cycled until enough unique chunks exist.
    """
    # Tokenise the full corpus and cycle it to sufficient length
    corpus_ids = tokenizer.encode(_SONNET_CORPUS * 50, add_special_tokens=False)

    rng = random.Random(seed)
    start_positions = list(range(0, len(corpus_ids) - input_len, max(1, input_len // 4)))
    rng.shuffle(start_positions)

    samples: List[PromptSample] = []
    for i in range(num_prompts):
        pos = start_positions[i % len(start_positions)]

        if prefix_len > 0:
            prefix_ids = corpus_ids[:prefix_len]
            suffix_ids = corpus_ids[pos: pos + (input_len - prefix_len)]
            chunk_ids  = prefix_ids + suffix_ids
        else:
            chunk_ids = corpus_ids[pos: pos + input_len]

        text      = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        in_toks   = len(tokenizer.encode(text, add_special_tokens=False))
        samples.append(PromptSample(prompt=text,
                                    input_tokens=in_toks,
                                    output_tokens=output_len))
    return samples


# ---------------------------------------------------------------------------
# Dolly loader (thin wrapper around the existing workload machinery)
# ---------------------------------------------------------------------------

def load_dolly(
    num_prompts: int,
    tokenizer,
    min_len: int = 0,
    max_len: int = 99_999,
    output_len: int = 128,
    seed: int = 42,
) -> List[PromptSample]:
    """Load prompts from Databricks Dolly-15k with optional length bounds."""
    from datasets import load_dataset as _ld           # noqa: PLC0415
    from .workloads import build_prompt_from_dolly     # noqa: PLC0415

    ds = _ld("databricks/databricks-dolly-15k", split="train")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    raw: List[PromptSample] = []
    for idx in indices:
        prompt = build_prompt_from_dolly(ds[idx])
        if not prompt:
            continue
        toks = len(tokenizer.encode(prompt, add_special_tokens=False))
        if min_len <= toks <= max_len:
            raw.append(PromptSample(prompt=prompt,
                                    input_tokens=toks,
                                    output_tokens=output_len))
        if len(raw) >= num_prompts:
            break

    while len(raw) < num_prompts:
        raw.extend(raw[: num_prompts - len(raw)])
    return raw[:num_prompts]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_dataset_prompts(
    dataset_name: str,
    num_prompts: int,
    tokenizer,
    input_len: int = 512,
    output_len: int = 128,
    prefix_len: int = 0,
    seed: int = 42,
) -> List[PromptSample]:
    """
    Unified loader — dispatches to the appropriate dataset function.

    Args:
        dataset_name: One of ``random``, ``sharegpt``, ``sonnet``, ``dolly``.
        num_prompts:  Number of prompts to return.
        tokenizer:    HuggingFace tokenizer for token counting / generation.
        input_len:    Target input length (tokens).  Meaning varies by dataset:
                        random   → exact synthetic length
                        sharegpt → soft filter (±50 %)
                        sonnet   → chunk size in tokens
                        dolly    → used as max_len filter
        output_len:   Target / fixed output length (tokens).
        prefix_len:   Shared prefix length for sonnet workload (0 = disabled).
        seed:         RNG seed for reproducibility.

    Returns:
        List of :class:`PromptSample`.
    """
    name = dataset_name.lower()
    if name == "random":
        return load_random(num_prompts, tokenizer,
                           input_len=input_len, output_len=output_len, seed=seed)
    elif name == "sharegpt":
        return load_sharegpt(num_prompts, tokenizer,
                             input_len=input_len, output_len=output_len, seed=seed)
    elif name == "sonnet":
        return load_sonnet(num_prompts, tokenizer,
                           input_len=input_len, output_len=output_len,
                           prefix_len=prefix_len, seed=seed)
    elif name == "dolly":
        return load_dolly(num_prompts, tokenizer,
                          max_len=input_len, output_len=output_len, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: random, sharegpt, sonnet, dolly"
        )
