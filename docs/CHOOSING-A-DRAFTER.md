# Choosing a drafter: measure your workload, not "the model"

A speculative drafter is not uniformly fast. Its benefit depends on how
predictable your output is, and that varies enormously by **workload shape**.
Published throughput numbers for the same model on the same hardware can differ
by **3x** purely because of which regime was measured.

## The trap, concretely

We benchmarked our stack with a single prompt — *"write a clear, self-contained
explanation… be concrete and complete"*, 256 tokens — and measured **18.99
tok/s**. We concluded the model was slow.

Then we found a published probe table for this model class comparing three
drafters:

| probe | EAGLE/MTP | DSpark | DFlash2 |
|---|---:|---:|---:|
| code | 34.5 | **51.5** | 50.9 |
| long essay | 24.1 | **18.3** | 25.4 |
| short chat | 21.0 | 23.2 | **66.6** |

Our prompt is the **long-essay** regime. DSpark's long-essay figure is **18.3**.
We measured **18.99**.

We had not discovered a slow model. We had benchmarked our drafter in the one
workload it is worst at, and then generalised from it.

Meanwhile someone else reporting ~30-40 tok/s on "short answers" with DFlash2 is
also right — that is the 66.6 column. Both numbers are true. They are not
measuring the same thing.

## How to choose

**Work out your real concurrency and turn shape first.** Both change the answer:

- **Short turns, low concurrency** (agents, tool calls, chat) → favours a
  drafter tuned for short chat. This is the regime with the largest spread
  between drafters, so the choice matters most here.
- **Long generations** (essays, narration, summarisation) → the spread narrows
  and some drafters actively regress.
- **High concurrency** → per-request drafting matters less; you are throughput-
  bound and the scheduler cap (`--max-running-requests`) usually binds first.

`scripts/track-concurrency.py` answers the concurrency half from the server's
own logs rather than from a guess:

```bash
python3 scripts/track-concurrency.py --container qwen38-27b --out results/concurrency.json
```

It tails SGLang's decode-batch lines, which carry `#running-req`, and writes a
histogram plus `pct_at_or_below_1`. Let it run through a normal day. If your
traffic is overwhelmingly c=1, **stop optimising aggregate throughput** — it is
measuring something you never do.

## Do not benchmark one regime

Whatever you choose, measure at least two prompt shapes: a short-answer probe
and a long-generation probe. A single number for "tok/s on model X" is close to
meaningless for a speculative server, and it is how both of the mistakes above
happen.

## And re-test after any drafter change

The probe table above was measured with an **older** drafter generation. We have
since upgraded that drafter and gained +26-55% acceptance
(see [`DRAFTER.md`](DRAFTER.md)), which may well have changed the ordering.

Comparisons age. Re-run them rather than inheriting them — that is exactly the
mistake that had us quoting a throughput figure measured on someone else's box.
