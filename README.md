# qwen38-27b-spark-stack

**A complete, working AI stack on one NVIDIA DGX Spark (GB10, 128 GB).** Not one
big model — a 27B vision-language brain, streaming ASR, TTS, and a small model,
all resident at once, with speculative decoding tuned and verified.

| | | |
|---|---|---|
| **Qwen3.8-27B** (NVFP4) — chat, tools, **vision in-process** | `:8001` | 56 GB |
| **ASR** — streaming speech recognition | `:8006` | 10 GB |
| **TTS** — voice synthesis | `:8095` | 5 GB |
| **3B** — a small model for cheap/parallel work | `:8002` | 12 GB |
| | | **83 GB of 121.7** |

Measured on that stack, with everything above co-resident:

| concurrency | tok/s | tokens/chunk (acceptance) |
|---:|---:|---:|
| 1 | **18.99** | 3.20 |
| 4 | **56.28** | 2.97 |
| 8 | **140.42** | 3.07 |

Cross-checked against SGLang's own `gen throughput` counter (139.55 tok/s at
c=8) at the same instant.

**Those numbers are with the whole stack resident.** We measured the brain solo
on the same box and it was not faster — 140.04 vs 140.42 at c=8, and *slower* at
c=1. Decode is memory-bandwidth-bound and idle models do not consume bandwidth,
so running ASR and TTS alongside is close to free.
[`docs/CO-RESIDENCY.md`](docs/CO-RESIDENCY.md) has the full A/B.

⚠️ **One number for "tok/s" on a speculative server is close to meaningless.**
The 18.99 above is a *long-generation* prompt. The same model and hardware with a
drafter tuned for short answers is reported around 3x that, and both figures are
honest — they measure different workloads. Read
[`docs/CHOOSING-A-DRAFTER.md`](docs/CHOOSING-A-DRAFTER.md) before comparing
anyone's throughput to anyone else's, including ours.

**There is no separate vision model.** The 27B is a native VLM and SGLang serves
the vision tower in-process, which is what makes the budget work — we removed a
dedicated 4B vision model and got 14 GB back for free.

---

## The three things this repo is actually for

Getting the stack up is the easy part. These are the three mistakes that cost us
most of a night. All three are silent — nothing errors, nothing warns.

### 1. Your speculative drafter must match the target's quantisation

We ran a drafter trained against the **FP8** target while serving the **NVFP4**
target. Everything worked. Canaries passed. Output was correct. It ran at about
a third of the speed it should have.

| | mismatched | matched |
|---|---:|---:|
| accept length | 1.52–2.15 | **3.03–3.32** |
| accept rate | 0.07–0.16 | **0.29–0.33** |
| single-stream | 6.0 tok/s | **18.99 tok/s** |

Speculative decoding degrades *gracefully* — rejected drafts are simply
re-verified by the target, so correctness is never at risk and nothing
complains. You lose only throughput, and only if you go looking.

```bash
docker logs <container> 2>&1 | grep -o 'accept len: [0-9.]*, accept rate: [0-9.]*' | tail -3
```

Expect ~3–4 for this model class. Under ~2.5 means something is wrong.
Full detail, including how drafters get republished in place under the same
name, in **[`docs/DRAFTER.md`](docs/DRAFTER.md)**.

### 2. Benchmark the workload you actually run

We measured one prompt shape — a 256-token "explain this thoroughly" — and
concluded the model was slow at 18.99 tok/s. A published probe table for this
model class then showed our drafter's *long-essay* figure as **18.3**, against
**66.6** for a different drafter on *short chat*. We had benchmarked our drafter
in the one regime it is worst at.

| probe | EAGLE/MTP | DSpark | DFlash2 |
|---|---:|---:|---:|
| code | 34.5 | **51.5** | 50.9 |
| long essay | 24.1 | **18.3** | 25.4 |
| short chat | 21.0 | 23.2 | **66.6** |

Before choosing a drafter, find out what concurrency and turn shape you actually
serve. [`scripts/track-concurrency.py`](scripts/track-concurrency.py) reads it
out of SGLang's own decode-batch logs:

```bash
python3 scripts/track-concurrency.py --container qwen38-27b --out results/concurrency.json
```

If your traffic is overwhelmingly single-stream, optimising aggregate throughput
is optimising something you never do.

### 3. Count tokens, not stream chunks

With speculative decoding a server emits **several accepted tokens in one SSE
chunk**. A benchmark that counts chunks divides your throughput by the
acceptance length. We shipped that bug: 256 tokens arrived in 83 chunks, and a
server doing 25.4 tok/s was reported as 8.2 — a **3.08× error**.

[`scripts/bench.py`](scripts/bench.py) requests `stream_options.include_usage`
and reports `usage.completion_tokens`. It also prints tokens-per-chunk, which on
a speculative server *is* the acceptance length — a free cross-check on the
point above.

The two bugs partially mask each other: a bad drafter lowers acceptance, and a
chunk-counting benchmark divides by acceptance. Fix only one and the story gets
more confusing, not less.

---

## Quick start

```bash
git clone https://github.com/karti-ai/qwen38-27b-spark-stack
cd qwen38-27b-spark-stack

# Target + matched drafter, downloaded and checksum-verified
DEST=$PWD/weights scripts/download-weights.sh

# Serve. Boot is ~5 min (weights, torch.compile, CUDA graph capture).
MODEL=$PWD/weights/Qwen3.8-27B-NVFP4 \
DRAFT=$PWD/weights/Qwen3.8-27B-DSpark \
  scripts/serve-27b.sh

# Canaries + the acceptance check
scripts/smoke-test.sh

# Vision, tools, throughput
python3 scripts/vision-test.py --base-url http://localhost:8001/v1 --model qwen3.8-27b
python3 scripts/bench.py --base-url http://localhost:8001/v1 --model qwen3.8-27b \
    --concurrency 1,4,8 --out results/mine.json

# What concurrency do you ACTUALLY serve? Run this for a day before tuning.
python3 scripts/track-concurrency.py --container qwen38-27b --out results/concurrency.json
```

**Requirements:** GB10 (sm_121), 128 GB unified memory, aarch64, Docker with the
NVIDIA runtime, ~50 GB free disk.

---

## Traps

| trap | symptom | cause |
|---|---|---|
| **drafter/target mismatch** | everything works, ~3× slow | trained for a different quantisation. Silent. [`docs/DRAFTER.md`](docs/DRAFTER.md) |
| **chunk-counting benchmark** | throughput looks terrible | speculation packs several tokens per SSE chunk |
| **thinking eats the budget** | `content` is `null`, no error | reasoning goes to `reasoning`; `finish_reason` is `length`. Raise `max_tokens` or set `enable_thinking: false` |
| **`--mem-fraction-static` too low** | dies at boot naming the wrong fix | `max_mamba_cache_size=-59` means weights + mamba state pool did not fit. Raise the fraction; do not shave draft tokens |
| **static pool is of TOTAL memory** | second model OOMs | it is not a fraction of what is *free*, and it is reserved at boot. Start the 27B **first** |
| **FP8 KV regression** | `19 × 23` returns 417 | this checkpoint declares `kv_cache_quant_algo: FP8`. The canary catches it |
| **stale drafter** | quietly worse than upstream | drafters get republished in place under the same name. Diff `config.json`, not the `.py` files |
| **benchmarking one prompt shape** | a number that misleads by 3× | speculative gain is workload-dependent. [`docs/CHOOSING-A-DRAFTER.md`](docs/CHOOSING-A-DRAFTER.md) |
| **giving the model the whole box** | no faster, sometimes slower | a KV pool larger than `--max-running-requests` can use is wasted. [`docs/CO-RESIDENCY.md`](docs/CO-RESIDENCY.md) |

---

## Why this stack fits

The 27B is **dense**, so its footprint is predictable — no expert-routing
surprises. At `--mem-fraction-static 0.46` it reserves ~56 GB of the 121.7 GB
pool, and because the vision tower is served in-process there is no separate
vision model to budget for. That leaves ~27 GB for ASR, TTS and a small model,
plus headroom.

Two rules make co-residency work, both learned the hard way:

- **The big model starts first.** `--mem-fraction-static` is a fraction of
  *total* memory reserved as a static pool at boot, not a fraction of what is
  free. Anything that starts before it steals from the pool it is about to
  claim.
- **Declare what a model is *authorised* to take, not what it currently uses.**
  A KV pool grows into its allocation. Sizing a budget from an early
  `nvidia-smi` reading is how you get a small process OOM-killed an hour later.

---

## Credit

- **[RadixArk](https://huggingface.co/RadixArk)** — the SGLang team: the NVFP4
  conversion and the DSpark drafters, published with acceptance-length
  evaluations across 17 workloads and 64,675 prompts. That evaluation is what
  let us identify our mismatch.
- **[SGLang](https://github.com/sgl-project/sglang)** — the serving runtime.
- **[SpecForge](https://github.com/sgl-project/SpecForge)** and
  **[DFlash](https://github.com/z-lab/dflash)** — the speculative-decoding
  training stack DSpark builds on.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab)** — published GB10 serving recipes
  that informed our starting parameters.

**Sibling repo:** [qwen38-flash-next-spark](https://github.com/karti-ai/qwen38-flash-next-spark)
— fitting a *single ~180B model* on one Spark. Different problem, same box.

## Licence

Apache-2.0. Model weights are not part of this repository; Qwen3.8-27B is
Apache-2.0 from Alibaba. See [`NOTICE`](NOTICE).
