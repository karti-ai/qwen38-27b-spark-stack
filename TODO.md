# TODO

## 1. A quiesced single-stream number
The 18.99 tok/s c=1 figure was taken with the rest of the stack resident and
other clients able to reach the port. SGLang's own log shows `#running-req`
during a run; confirm it was 1 throughout, or stop other consumers first. A
"single-stream" number measured against a busy port is a *share*, not a ceiling.

## 2. Sweep the speculative parameters
`--speculative-dspark-block-size 7` and `--speculative-num-draft-tokens 8` come
from the drafter's model card. They are the trained defaults, not necessarily
the best for our request shape (short agent turns). Sweep both and report
acceptance alongside throughput.

## 3. Prose vs code
An earlier drafter generation on this box was directionally faster on code and
tool-calls and *slower* on free-form prose. Whether that trade survives in the
current drafter is untested. It matters: a narration path and an agent path want
different configurations.

## 4. Publish the co-residency numbers properly
Footprints in the README are declared budget figures. Add measured RSS and
`nvidia-smi` readings per model, and note the gap — a KV pool grows into its
allocation, so early readings understate.

## 5. Vision benchmark
`scripts/vision-test.py` proves the tower works. It does not measure quality.
Run something comparable to a published vision benchmark so the in-process
vision claim carries a number.

## 6. Document the ASR/TTS side
This repo currently documents the brain in detail and lists the rest. The
co-residency story is the point, so the other three deserve their own serving
notes and traps.
