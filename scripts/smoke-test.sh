#!/usr/bin/env bash
# Correctness canaries plus the acceptance check that a mismatched drafter fails.
#
#   PORT=8001  MODEL=qwen3.8-27b  CONTAINER=qwen38-27b
#
# ⚠️ THINKING: this model reasons by default. With --reasoning-parser qwen3 the
#    chain of thought goes to `reasoning` and `content` comes back NULL if the
#    budget runs out first — finish_reason "length", no error. The serve script
#    sets enable_thinking false as a server default; these canaries set it
#    per-request too. Set THINK=1 to exercise the reasoning path (needs ~20x
#    the budget).
set -uo pipefail

PORT="${PORT:-8001}"
MODEL="${MODEL:-qwen3.8-27b}"
BASE="${BASE:-http://localhost:${PORT}/v1}"
CONTAINER="${CONTAINER:-qwen38-27b}"
THINK="${THINK:-0}"
fail=0

if [ "$THINK" = 1 ]; then KW='{}'; MULT=20; else KW='{"enable_thinking": false}'; MULT=1; fi

ask() {
  local n=$(( $2 * MULT ))
  curl -s --max-time 600 "$BASE/chat/completions" -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --arg p "$1" --argjson n "$n" --argjson kw "$KW" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,chat_template_kwargs:$kw}')" \
    | jq -r '.choices[0] | "\(.finish_reason)\t\(.message.content // "NULL_CONTENT")"'
}

check() {
  local label="$1" want="$2" prompt="$3" n="$4" raw finish got
  raw="$(ask "$prompt" "$n")"; finish="${raw%%$'\t'*}"
  got="$(printf '%s' "${raw#*$'\t'}" | tr -d '\n' | sed 's/^ *//;s/ *$//')"
  if [ "$got" = "NULL_CONTENT" ]; then
    printf '  [FAIL] %-16s content NULL (finish=%s)\n' "$label" "$finish"
    [ "$finish" = "length" ] && printf '         reasoning ate the budget — not a broken model.\n'
    fail=$((fail+1)); return
  fi
  if printf '%s' "$got" | grep -qiE "$want"; then
    printf '  [PASS] %-16s %s\n' "$label" "${got:0:60}"
  else
    printf '  [FAIL] %-16s want /%s/ got: %s\n' "$label" "$want" "${got:0:70}"; fail=$((fail+1))
  fi
}

command -v jq >/dev/null || { echo "!! needs jq"; exit 1; }
curl -sf --max-time 10 "$BASE/models" >/dev/null || { echo "!! not answering on $BASE (boot is ~5 min)"; exit 1; }
echo ">> $BASE  (thinking $([ "$THINK" = 1 ] && echo ON || echo OFF))"
echo ">> serving: $(curl -s "$BASE/models" | jq -r '.data[].id' | paste -sd, -)"
echo

# 437 is the canary for FP8 KV integrity on this checkpoint: a 417 means the
# fp8 KV cache regressed. Do not "fix" --kv-cache-dtype without re-running this.
check "arithmetic"     '437'          'What is 19 multiplied by 23? Reply with only the number.' 24
check "reasoning-trap" '\b9\b'        'A farmer has 17 sheep. All but 9 die. How many are left? Reply with only the number.' 32
check "instruction"    'acknowledged' 'Reply with exactly one word: acknowledged' 16
check "code"           'def|return'   'Write an iterative Python fib(n). Code only.' 220

# The check that a silently-mismatched drafter fails. See docs/DRAFTER.md.
echo
echo ">> speculative acceptance (the silent failure mode):"
acc="$(docker logs --tail 200 "$CONTAINER" 2>&1 | tr '\r' '\n' \
      | grep -oE 'accept len: [0-9.]+, accept rate: [0-9.]+' | tail -3)"
if [ -z "$acc" ]; then
  echo "   (no decode batches logged yet, or container name '$CONTAINER' is wrong)"
else
  echo "$acc" | sed 's/^/   /'
  len="$(printf '%s' "$acc" | tail -1 | grep -oE 'accept len: [0-9.]+' | grep -oE '[0-9.]+')"
  if [ -n "$len" ] && awk "BEGIN{exit !($len < 2.5)}"; then
    echo "   ⚠️  accept length ${len} is LOW for this model class (expect ~3-4)."
    echo "      Most likely your drafter was trained for a different target"
    echo "      quantisation. This does not break correctness, only speed."
    echo "      Read docs/DRAFTER.md."
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "all canaries passed."
else
  echo "$fail canary/canaries FAILED."; exit 1
fi
