#!/usr/bin/env python3
"""Record what concurrency the brain actually serves, from SGLang's own log.

SGLang prints `#running-req: N` on every decode batch. Tailing that gives a real
histogram of served concurrency instead of a guess — which is the only honest
way to decide whether to tune for latency (c=1) or throughput (c=8+).

Writes a JSON histogram periodically so it survives restarts of this script.

  python3 conc-track.py --container brain-qwen38-dspark --out ./results/concurrency.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

# Count DECODE batches only. A "Prefill batch" line reports #running-req as the
# number of OTHER requests already running, so a solo request logs 0 there —
# lumping prefill in inflates the zero bucket and hides the real distribution.
# We hit exactly that: 90% "concurrency 0" that was really solo prefills.
DECODE = re.compile(r"Decode batch.*#running-req:\s*(\d+)")
PREFILL = re.compile(r"Prefill batch.*#running-req:\s*(\d+)")
ACCEPT = re.compile(r"accept len:\s*([0-9.]+)")
GENTP = re.compile(r"gen throughput \(token/s\):\s*([0-9.]+)")


def load(path: str) -> dict:
    """Resume from a previous run.

    NOTE the saved file is a *report*, not the internal state: it stores
    `mean_accept_len`, not the running sum and count that produced it. So we
    rebuild what we can and default the rest, rather than assuming the keys we
    write are the keys we read. Getting that wrong crash-looped this service.
    """
    st = {
        "running_hist": Counter(),
        "decode_batches": 0,
        "prefill_batches": 0,
        "accept_len_sum": 0.0,
        "accept_n": 0,
        "gen_tp_sum": 0.0,
        "gen_tp_n": 0,
        "started": time.time(),
    }
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:  # noqa: BLE001 — missing or corrupt file just starts fresh
        return st

    hist = d.get("running_hist") or {}
    try:
        st["running_hist"] = Counter({int(k): int(v) for k, v in hist.items()})
    except (TypeError, ValueError):
        st["running_hist"] = Counter()
    st["decode_batches"] = int(d.get("decode_batches") or 0)
    st["prefill_batches"] = int(d.get("prefill_batches") or 0)
    st["started"] = d.get("started") or st["started"]

    # Re-seed the running means from the reported average so resuming does not
    # throw away history: one pseudo-sample weighted by the batches behind it.
    mal = d.get("mean_accept_len")
    if mal and st["decode_batches"]:
        st["accept_len_sum"] = float(mal) * st["decode_batches"]
        st["accept_n"] = st["decode_batches"]
    mgt = d.get("mean_gen_tok_s")
    if mgt and st["decode_batches"]:
        st["gen_tp_sum"] = float(mgt) * st["decode_batches"]
        st["gen_tp_n"] = st["decode_batches"]
    return st


def save(path: str, st: dict) -> None:
    hist = st["running_hist"]
    total = sum(hist.values())
    out = {
        "started": st["started"],
        "updated": time.time(),
        "decode_batches": st["decode_batches"],
        "prefill_batches": st.get("prefill_batches", 0),
        "running_hist": {str(k): v for k, v in sorted(hist.items())},
        "running_pct": (
            {str(k): round(100.0 * v / total, 2) for k, v in sorted(hist.items())}
            if total else {}
        ),
        "mean_accept_len": (
            round(st["accept_len_sum"] / st["accept_n"], 3) if st["accept_n"] else None
        ),
        "mean_gen_tok_s": round(st["gen_tp_sum"] / st["gen_tp_n"], 2) if st["gen_tp_n"] else None,
        # The number that decides the tuning question.
        "pct_at_or_below_1": round(
            100.0 * sum(v for k, v in hist.items() if k <= 1) / total, 2) if total else None,
        "max_observed": max(hist) if hist else None,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="qwen38-27b")
    ap.add_argument("--out", default=os.path.expanduser("./results/concurrency.json"))
    ap.add_argument("--flush-seconds", type=float, default=30.0)
    args = ap.parse_args()

    st = load(args.out)
    last = 0.0

    while True:
        # --since 0m: only new lines, so a container restart does not double-count.
        p = subprocess.Popen(
            ["docker", "logs", "-f", "--since", "0m", args.container],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:  # type: ignore[union-attr]
                m = DECODE.search(line)
                if m:
                    st["running_hist"][int(m.group(1))] += 1
                    st["decode_batches"] += 1
                elif PREFILL.search(line):
                    st["prefill_batches"] = st.get("prefill_batches", 0) + 1
                a = ACCEPT.search(line)
                if a:
                    st["accept_len_sum"] += float(a.group(1)); st["accept_n"] += 1
                g = GENTP.search(line)
                if g:
                    st["gen_tp_sum"] += float(g.group(1)); st["gen_tp_n"] += 1
                now = time.time()
                if now - last > args.flush_seconds:
                    save(args.out, st); last = now
        except Exception as exc:  # noqa: BLE001
            print(f"stream ended: {exc}", file=sys.stderr)
        finally:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        save(args.out, st)
        # The container may be restarting (scene swap); wait and re-attach.
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
