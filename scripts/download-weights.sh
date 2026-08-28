#!/usr/bin/env bash
# Fetch the 27B NVFP4 target and its matched DSpark drafter, then verify both.
#
#   MODEL_REPO / DRAFT_REPO   HuggingFace repo ids
#   DEST=./weights            where they land
#   SKIP_VERIFY=1             skip checksums (not recommended)
#
# ⚠️ The drafter MUST be the one trained for this target's quantisation.
#    See docs/DRAFTER.md — a mismatch costs ~3x throughput and reports nothing.
set -uo pipefail

MODEL_REPO="${MODEL_REPO:-RadixArk/Qwen3.8-27B-NVFP4}"
DRAFT_REPO="${DRAFT_REPO:-RadixArk/Qwen3.8-27B-DSpark}"
DEST="${DEST:-$(pwd)/weights}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v hf >/dev/null 2>&1 || { echo "!! need the 'hf' CLI:  pip install -U huggingface_hub"; exit 1; }
mkdir -p "$DEST"

for pair in "$MODEL_REPO:target" "$DRAFT_REPO:draft"; do
  repo="${pair%:*}"; kind="${pair##*:}"
  out="$DEST/$(basename "$repo")"
  echo ">> $kind: $repo -> $out"
  ok=0
  for i in 1 2 3 4 5; do
    hf download "$repo" --local-dir "$out" --max-workers 6 && { ok=1; break; }
    echo "-- attempt $i failed; retrying in 30s"; sleep 30
  done
  [ "$ok" = 1 ] || { echo "!! download failed: $repo"; exit 1; }
  if [ "${SKIP_VERIFY:-0}" != 1 ]; then
    python3 "$HERE/verify-weights.py" --repo "$repo" --dir "$out" || exit 1
  fi
done

echo
echo ">> both downloaded and verified."
echo ">> BEFORE SERVING, confirm the drafter matches this target's quantisation:"
echo "     grep -oE '\"(target_layer_ids|mask_token_id)\": *[^,]*' $DEST/$(basename "$DRAFT_REPO")/config.json"
echo "   and read docs/DRAFTER.md. A mismatch is silent and costs ~3x."
