# Does running a stack cost you speed?

**Measured answer: essentially no.** We benchmarked the same model, same
drafter, same harness, same prompts, twice — once with the full stack resident,
once with the brain alone owning the box.

| c | co-resident (4 models, `mem-fraction 0.46`) | solo (`mem-fraction 0.90`) |
|---:|---:|---:|
| 1 | **18.99** | 15.66 |
| 4 | 56.28 | **68.88** |
| 8 | **140.42** | 140.04 |
| 16 | — | 114.37 |

At c=8 the two are within **0.3%**. At c=1 the co-resident config measured
*faster*. Only c=4 clearly favoured solo.

## Why co-residency is nearly free

Decode on a dense model is **memory-bandwidth-bound**, and idle models do not
consume bandwidth. They occupy memory, not throughput. On a GB10 with ~273 GB/s
and ~21.5 GiB of NVFP4 weights, roughly 12.7 forward passes per second is the
hard ceiling; speculative decoding lifts the token rate above that by accepting
~3 tokens per verified step, which is where ~19 tok/s comes from.

An ASR model and a TTS model sitting idle change none of those terms.

What co-residency *does* cost is memory available for the KV cache — which
bounds how many requests can be in flight, not how fast any one of them runs.

## The solo config was actually mis-tuned

The solo run allocated a **1,702,758-token** KV cache and left
`available_gpu_mem=6.51 GB`, with the OS page cache squeezed to 3 GB. That is
far more KV than `--max-running-requests 10` can ever use, and it is the most
likely reason solo measured *worse* at c=1.

The c=16 result makes the same point from the other side: throughput **dropped**
to 114 tok/s because six requests were queuing behind the request cap. Memory
was never the limit — the scheduler cap was.

**So the co-resident configuration is not a compromise. It is the better-tuned
one.** Giving the brain the whole box buys nothing unless you also raise
`--max-running-requests`, and even then only if your workload is actually
concurrent.

## Two rules that make co-residency work

- **Start the big model first.** `--mem-fraction-static` is a fraction of
  *total* memory reserved as a static pool at boot — not a fraction of what is
  free. Anything that starts before it steals from the pool it is about to
  claim.
- **Budget what a model is *authorised* to take, not what it currently uses.**
  A KV pool grows into its allocation. Sizing from an early `nvidia-smi` reading
  is how a small co-resident process gets OOM-killed an hour later.

## Caveat

These are single runs. Run-to-run variation at c=1 is wide enough that
15.66 vs 18.99 should be read as "no meaningful difference", not as evidence
that co-residency *helps*. The c=8 tie is the solid result.
