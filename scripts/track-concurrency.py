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

# TWO SIGNALS, AND WHICH ONE YOU WANT DEPENDS ON YOUR WORKLOAD.
#
# "Decode batch" lines carry the true count of requests generating concurrently.
# But SGLang only emits one every `--decode-log-interval` steps (default 40), so
# any request that finishes in under 40 decode steps NEVER APPEARS. On a
# short-answer workload that silently discards most of your traffic: we logged
# 35,734 prefill batches and only 3,891 decode batches over the same 19 hours,
# and the decode sample is biased toward long generations.
#
# "Prefill batch" lines report #running-req as the number of OTHER requests
# already in flight when this one arrived. Every request emits at least one, so
# it is the unbiased arrival-concurrency signal — the right one for short turns.
# (A long prompt can span several prefill batches under chunked prefill, so
# treat the count as batches, not requests.)
#
# We record both. Read `prefill_hist` for "how concurrent is my traffic",
# and `decode_hist` for "how concurrent is my generation".
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
        "decode_hist": Counter(),
        "prefill_hist": Counter(),
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

    for key, dest in (("decode_hist", "decode_hist"), ("prefill_hist", "prefill_hist")):
        raw = d.get(key) or {}
        try:
            st[dest] = Counter({int(k): int(v) for k, v in raw.items()})
        except (TypeError, ValueError):
            st[dest] = Counter()
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
    def pct(counter):
        total = sum(counter.values())
        if not total:
            return {}
        return {str(k): round(100.0 * v / total, 2) for k, v in sorted(counter.items())}

    def at_or_below(counter, n):
        total = sum(counter.values())
        if not total:
            return None
        return round(100.0 * sum(v for k, v in counter.items() if k <= n) / total, 2)

    dh, ph = st["decode_hist"], st["prefill_hist"]
    out = {
        "started": st["started"],
        "updated": time.time(),

        # ARRIVAL concurrency: other requests already in flight when one arrived.
        # Unbiased — every request emits a prefill line. Use this for short turns.
        "prefill_batches": st["prefill_batches"],
        "prefill_hist": {str(k): v for k, v in sorted(ph.items())},
        "prefill_pct": pct(ph),
        "prefill_pct_alone": at_or_below(ph, 0),

        # GENERATION concurrency. Biased toward long requests: SGLang emits a
        # decode line only every --decode-log-interval steps (default 40), so
        # short generations never appear here at all.
        "decode_batches": st["decode_batches"],
        "decode_hist": {str(k): v for k, v in sorted(dh.items())},
        "decode_pct": pct(dh),
        "decode_pct_at_or_below_1": at_or_below(dh, 1),
        "decode_sample_bias_note": (
            "decode lines are emitted every --decode-log-interval steps; "
            "requests shorter than that never appear. Prefer prefill_* for "
            "short-turn workloads."
        ),

        "max_observed": max(list(dh) + list(ph)) if (dh or ph) else None,
        "mean_accept_len": (
            round(st["accept_len_sum"] / st["accept_n"], 3) if st["accept_n"] else None
        ),
        "mean_gen_tok_s": (
            round(st["gen_tp_sum"] / st["gen_tp_n"], 2) if st["gen_tp_n"] else None
        ),
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
    empty_streaks = 0

    while True:
        lines_this_attach = 0
        # --since 0m: only new lines, so a container restart does not double-count.
        p = subprocess.Popen(
            ["docker", "logs", "-f", "--since", "0m", args.container],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:  # type: ignore[union-attr]
                lines_this_attach += 1
                # Surface the failure instead of retrying into the void. A
                # systemd --user manager started before the user joined the
                # docker group has no docker group, and `docker logs` then fails
                # with a permission error forever. We shipped that: the service
                # sat "active" for 19 hours recording zero.
                low = line.lower()
                if "permission denied" in low and "docker" in low:
                    print("FATAL: cannot read docker logs — "
                          + line.strip(), file=sys.stderr)
                    print("  the process running this has no access to "
                          "/var/run/docker.sock.", file=sys.stderr)
                    print("  if under systemd --user, its manager predates your "
                          "docker group membership;", file=sys.stderr)
                    print("  run via:  sg docker -c '<command>'   or restart the "
                          "user manager.", file=sys.stderr)
                    return 2
                if "no such container" in low:
                    print(f"FATAL: no such container: {args.container}",
                          file=sys.stderr)
                    return 3
                m = DECODE.search(line)
                if m:
                    st["decode_hist"][int(m.group(1))] += 1
                    st["decode_batches"] += 1
                else:
                    pm = PREFILL.search(line)
                    if pm:
                        st["prefill_hist"][int(pm.group(1))] += 1
                        st["prefill_batches"] += 1
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

        # A stream that attaches and immediately ends, repeatedly, means we are
        # not actually reading anything. Say so rather than looping silently.
        if lines_this_attach == 0:
            empty_streaks += 1
            if empty_streaks in (5, 25, 100):
                print(f"WARNING: {empty_streaks} consecutive empty attaches to "
                      f"'{args.container}' — recording nothing.", file=sys.stderr)
        else:
            empty_streaks = 0

        # The container may be restarting (scene swap); wait and re-attach.
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
