#!/usr/bin/env bash
# Serve Qwen3.8-27B (NVFP4) on ONE GB10 with SGLang + DSpark speculative decoding.
#
#   MODEL=<dir>     REQUIRED. Qwen3.8-27B NVFP4 checkpoint.
#   DRAFT=<dir>     REQUIRED. DSpark draft model. ⚠️ READ docs/DRAFTER.md FIRST —
#                   the drafter must be trained for the quantisation of the target
#                   you actually serve, and a mismatch fails SILENTLY.
#   PORT=8001       host port
#   CTX=262144      context length (native)
#   MEM_FRAC=0.46   --mem-fraction-static. See the warning below.
#   MAX_RUNNING=10  --max-running-requests
#   CPUSET=5-9,15-19  GB10 performance cores (Cortex-X925 at 3.9 GHz; the
#                   efficiency cluster runs at 2.81 GHz). Set empty to disable.
#   IMAGE=...       SGLang image, pinned by digest by default.
#   EXTRA=          extra sglang flags, verbatim.
#
# ⚠️ --mem-fraction-static IS A FRACTION OF *TOTAL* MEMORY, NOT OF WHAT IS FREE,
#    and SGLang reserves it as a STATIC pool at boot. On a 121.7 GiB GB10, 0.46 is
#    ~56 GB. The default here is deliberately modest because the point of this
#    repo is running a whole STACK — ASR, TTS and a small model alongside. If the
#    27B is the only thing on your box, raise it.
#
# ⚠️ A TOO-SMALL POOL FAILS AT BOOT, and the error names the wrong remedy:
#      "Not enough GPU memory for hybrid (mamba/linear-attention) state cache.
#       Computed max_mamba_cache_size=-59"
#    That means weights plus the mamba state pool did not fit. Raise MEM_FRAC;
#    do not start shaving speculative draft tokens.
#
# ⚠️ THE MODEL MUST START FIRST if you run other models on the same box, because
#    the static pool is computed against total memory. Start it, wait for health,
#    then start everything else.
set -euo pipefail

PORT="${PORT:-8001}"
CTX="${CTX:-262144}"
MEM_FRAC="${MEM_FRAC:-0.46}"
MAX_RUNNING="${MAX_RUNNING:-10}"
NAME="${NAME:-qwen38-27b}"
CPUSET="${CPUSET-5-9,15-19}"
EXTRA="${EXTRA:-}"
# Pinned by digest: :latest is not this build and the numbers do not transfer.
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}"

[ -n "${MODEL:-}" ] || { echo "!! set MODEL=/path/to/Qwen3.8-27B-NVFP4"; exit 1; }
[ -n "${DRAFT:-}" ] || { echo "!! set DRAFT=/path/to/Qwen3.8-27B-DSpark  (see docs/DRAFTER.md)"; exit 1; }
MODEL="$(cd "$MODEL" && pwd)"; DRAFT="$(cd "$DRAFT" && pwd)"
[ -f "$MODEL/config.json" ] || { echo "!! no config.json in $MODEL"; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "!! no config.json in $DRAFT"; exit 1; }

PIN=(); [ -n "$CPUSET" ] && PIN=(--cpuset-cpus "$CPUSET")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 32g --network host "${PIN[@]}" \
  -v "$MODEL:/model:ro" -v "$DRAFT:/draft:ro" \
  "$IMAGE" \
  sglang serve --trust-remote-code \
    --model-path /model --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port "$PORT" \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRAC" \
    --disable-prefill-cuda-graph \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 4.21 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 40 \
    --sampling-defaults model \
    --max-running-requests "$MAX_RUNNING" \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /draft \
    --speculative-dspark-block-size 7 \
    --speculative-draft-model-quantization unquant \
    --speculative-num-draft-tokens 8 \
    --enable-torch-compile --torch-compile-max-bs 4 \
    $EXTRA

echo ">> $NAME starting on :$PORT  (ctx $CTX · mem-frac $MEM_FRAC · cpuset ${CPUSET:-all})"
echo ">> boot is ~5 min (weights, torch.compile, CUDA graph capture)."
echo ">> follow:  docker logs -f $NAME"
echo ">> then:    PORT=$PORT scripts/smoke-test.sh"
echo ">> and CHECK ACCEPTANCE — a mismatched drafter is silent:"
echo "     docker logs $NAME 2>&1 | grep -o 'accept len: [0-9.]*, accept rate: [0-9.]*' | tail -3"
