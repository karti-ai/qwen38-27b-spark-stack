# Your speculative drafter must match the target's quantisation

**This is the most expensive thing we learned, and it fails completely silently.**

## What happened

We served `Qwen3.8-27B-NVFP4` with a DSpark drafter, and it worked. No error, no
warning, canaries passing, sensible output. It was also running at roughly a
third of the speed it should have.

The drafter we had was trained against the **FP8** target. We serve the **NVFP4**
target. Nothing anywhere reports that mismatch.

| | mismatched drafter | matched drafter |
|---|---:|---:|
| accept length | 1.52–2.15 | **3.03–3.32** |
| accept rate | 0.07–0.16 | **0.29–0.33** |
| server `gen throughput` | 13.5–29.8 tok/s | **24.3–139.6 tok/s** |
| single-stream (our harness) | 6.0 tok/s | **18.99 tok/s** |

Speculative decoding degrades *gracefully*. When the draft model proposes tokens
the target rejects, the target simply verifies and moves on — correctness is
never at risk, which is exactly why nothing complains. You lose throughput, and
only throughput, and only if you go looking.

For scale: the vendor documents the newer drafter at **+26%** acceptance over the
older one. Against a *mismatched* older one we measured closer to **+55%**.

## How to check, in one command

```bash
docker logs <container> 2>&1 | grep -o 'accept len: [0-9.]*, accept rate: [0-9.]*' | tail -5
```

SGLang prints this on every decode batch. Read it after any change to the model,
the drafter, or the quantisation of either.

**What good looks like:** accept length in the 3–4 range for this model class.
**What a mismatch looks like:** accept length near 1.5–2.0, accept rate under 0.2.

An accept length of 1.0 means speculation is buying you nothing at all — you are
paying for the draft forward pass and accepting none of it.

## Verify the pairing before you serve

Three things to confirm, in order of how often they are wrong:

1. **Quantisation of the target the drafter was trained against.** Model cards
   state this. Ours said the drafter targeted `-FP8`; we served `-NVFP4`.
2. **`target_layer_ids` in the drafter's `config.json`.** The drafter reads
   auxiliary features from specific layers of the target. Two versions of the
   "same" drafter had `[4, 16, 28, 40, 52]` and `[5, 19, 33, 47, 61]`. Pointing
   at the wrong layers is exactly the kind of thing that degrades quietly.
3. **The serving parameters the drafter was trained for** — block size, draft
   token count, mask token id. Ours differed between versions
   (`mask_token_id` 248077 vs 248070).

SGLang echoes what it actually loaded, and this is worth reading rather than
assuming:

```
Initialized DSpark draft runner. attention_backend=flashinfer,
model=DSparkDraftModel, gamma=7, verify_num_draft_tokens=8,
mask_token_id=248070, markov_head=VanillaMarkov
```

Compare `mask_token_id` and `gamma` against the drafter's card. If they disagree
with the version you think you installed, you have installed a different one.

## Drafters get republished in place

The repository we were pulling from **replaced** its v1 checkpoint with v2 under
the same name. A local copy downloaded weeks earlier stays silently stale — same
path, same filename, different model.

`config.json` is the cheap tell. Ours differed from upstream on
`target_layer_ids`, `intermediate_size` (10240 vs 17408), `num_attention_heads`
(40 vs 32), `eos_token_id`, `mask_token_id`, and gained a whole `rope_scaling`
block — while `dspark.py` and `dflash.py` were byte-identical, so a casual
`diff` of the code would have found nothing.

Check `lastModified` on the Hub against your local mtime, and diff `config.json`
before assuming your copy is current.

## And measure with a harness that counts tokens

Not chunks. With speculative decoding a server emits **several accepted tokens in
one SSE chunk**, so a chunk-counting benchmark divides your throughput by the
acceptance length. We shipped that bug and it cost us a **3.08×** error — 256
tokens arrived in 83 chunks, and a server doing 25.4 tok/s was reported as 8.2.

[`scripts/bench.py`](../scripts/bench.py) requests
`stream_options.include_usage` and reports `usage.completion_tokens`. It also
prints tokens-per-chunk, which on a speculative server **is** the acceptance
length — a free second opinion on everything above.

The irony worth noting: a mismatched drafter lowers acceptance, and a
chunk-counting benchmark divides by acceptance. The two bugs partially mask each
other. Fixing only one of them tells you a confusing story.
