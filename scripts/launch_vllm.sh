#!/usr/bin/env bash
# Launch a local vLLM OpenAI-compatible server for CompactingAgent/FoldingAgent tests.
#
# Default model is Qwen3-4B-Instruct-2507. Override with MODEL=... and PORT=....
# The server speaks /v1/chat/completions, so mini-swe-agent picks it up via
# OPENAI_BASE_URL + OPENAI_API_KEY (any non-empty string).
#
# Typical usage:
#   bash scripts/launch_vllm.sh              # foreground, logs to stdout
#   LOGFILE=/tmp/vllm.log bash scripts/launch_vllm.sh &   # detach
#   curl http://127.0.0.1:8000/v1/models     # smoke check
set -eu

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
# vLLM's served-model-name — what clients pass as `model` in API calls.
# Keep it stable regardless of the HF repo so configs don't churn when we swap models.
SERVED_NAME=${SERVED_NAME:-qwen3-4b-instruct-2507}
TP=${TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
LOGFILE=${LOGFILE:-}

# Pin to a single GPU unless caller overrides — keeps the rest of the node
# free for training.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cmd=(
  python -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
    --disable-log-requests
)

echo "Launching: ${cmd[*]}"
if [[ -n "${LOGFILE}" ]]; then
  exec "${cmd[@]}" > "${LOGFILE}" 2>&1
else
  exec "${cmd[@]}"
fi
