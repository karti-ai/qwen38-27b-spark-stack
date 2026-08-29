# DSpark vs DFlash2 on one Spark, and a fix that worked but didn't help

Three configurations, one variable at a time, same box, same harness, same
prompts, token-counted (not chunk-counted). Raw JSON in
[`../results/`](../results/).

| config | short c=1 | short c=4 | long c=1 | long c=4 | accept len |
|---|---:|---:|---:|---:|---:|
| **DSpark v2** (packed-NVFP4 head) | **8.32** | **29.81** | 14.61 | 68.11 | 3.0–4.6 |
| **DFlash2** (same target, head *eager*) | 3.96 | 24.61 | **30.65** | 76.96 | 2.95–4.28 |
| **DFlash2** (BF16 head, folded into graph) | 4.39 | 28.10 | 24.60 | **77.84** | 3.12–3.54 |

`short` = eight terse agent-style turns, 48 tokens. `long` = one sustained
256-token generation. See [CHOOSING-A-DRAFTER.md](CHOOSING-A-DRAFTER.md) for why
one profile is not enough.

## The drafters split cleanly by workload

**DSpark wins short turns by 2.1x. DFlash2 wins long generations by 2.1x.**
Neither dominates.

This *contradicts* the published probe table we had been carrying, which put
DFlash2 at 66.6 tok/s on short chat against DSpark's 23.2 — the exact opposite
ordering. That table was measured against DSpark **v1**; we run v2, which lifted
acceptance substantially. **A drafter comparison does not survive a drafter
upgrade.** Re-run it rather than inherit it.

Why DFlash2 loses on short turns here: its TTFT is 0.381 s against DSpark's
0.128 s. Over a 48-token answer that overhead never amortises. Over 256 tokens
it disappears and DFlash2's faster steady-state decode takes over.

## The fix that worked and didn't help

Serving our target, SGLang logged:

```
DFLASH draft greedy head kept eager (reason=quantized lm_head)
```

That reads like free performance sitting on the table. Confirmed at tensor level
— our target ships `lm_head.weight` as `U8` (packed NVFP4) plus a `weight_scale`,
while RadixArk's `-BF16-LMHead` build ships it as plain `BF16`. Both single-Spark
recipes we could find use a dense-head target.

So we downloaded the BF16-head build and served it. The warning disappeared,
replaced by:

```
DFLASH selector decode (greedy + sampling) folded into the draft cuda graph.
```

**The mechanism was exactly as diagnosed. The throughput went the wrong way:**
long-form single-stream fell from **30.65 to 24.60 tok/s**. Short gained a
little (3.96 → 4.39) but stayed at roughly half DSpark's. Acceptance was flat
across all three configs (3.1–3.5), so this is not a draft-quality effect.

**A log line that looks like a performance warning was not costing performance.**
It was visible, plausible, and mechanically real — and chasing it cost a 23 GB
download and two boots to establish that it is a dead end.

Caveat, stated because it matters: the two targets are different *builds* of the
same quantisation (a 4-shard sm121 build vs RadixArk's 3-shard build), not
solely a head swap. The regression cannot be attributed to the head alone.

## What we actually changed our minds about

- Do not port a drafter comparison across a drafter version.
- Do not treat a warning as a bottleneck without measuring it.
- **Publish the negative result.** Every recipe repo shows the config that won.
  The expensive knowledge is which plausible optimisations do nothing — see also
  the CPU-pinning and page-cache results in our sibling repo, both refuted.

## Choosing: measure your generation length, not your concurrency

We nearly got this wrong. Having measured that **95% of requests arrive to an
idle server**, we reasoned "concurrency is 1, so optimise for c=1" and picked the
drafter that won our short probe. But *both* probes above are c=1 measurements —
concurrency was never the deciding variable. **Output length was.**

The measurement that settled it, from the serving log of live traffic:

| | |
|---|---|
| decode lines per request | ~2.05 (258 lines / 126 requests) |
| implied decode steps per request | ~82 |
| x acceptance 3.1 → mean output | **~250 tokens** |

Every request emitted at least two decode lines, so essentially nothing finished
in under 80 steps: there was no short-turn population at all. Our traffic is the
*long* profile almost exactly — and on that profile DFlash2 is 2.1x faster. We
switched.

**We had assumed "agent traffic = short turns" and never checked.** The
assumption survived several rounds of careful benchmarking because it was never
stated as a claim, only used as one.

To measure your own: SGLang emits a decode line every `--decode-log-interval`
steps (default 40). Count those against your request count, multiply by the
interval and by your acceptance length, and you have mean output tokens. A
request that emits zero decode lines finished in under one interval — if most of
yours do, you genuinely are a short-turn workload and the ordering above
reverses.

One more number worth reconciling: live decode throughput averaged **24.36 tok/s**
while our synthetic c=1 probes read 8.32 and 14.61. Those are different metrics —
the live figure is decode-only, ours are whole-request including prefill and
first-token latency. Do not compare them directly, and be careful reading anyone
else's headline tok/s without knowing which one they quote.
