#!/usr/bin/env bash
# Sample decision-certificate judgments for one data split:
# GPT-4o (OpenAI API) + Qwen3-8B (local vLLM server).
#
# Start the Qwen3-8B server first, e.g.:
#   vllm serve Qwen/Qwen3-8B --port 8000
set -euo pipefail

# -------------------- User config --------------------

: "${PYTHON:=python3}"

# Local HelpSteer3 JSONL and its name (example_id = "<SPLIT_NAME>:<line>")
: "${DATA:?Set DATA to a local HelpSteer3 JSONL file; see data/README.md}"
: "${SPLIT_NAME:=train}"

# Output directory and file tag, e.g. rollouts/{gpt4o,qwen3}_train.jsonl
: "${OUT_DIR:=rollouts}"
: "${TAG:=train}"

# GPT-4o rollouts (needs OPENAI_API_KEY)
: "${GPT_MODEL:=gpt-4o}"
: "${GPT_N:=3}"
: "${GPT_TEMPERATURE:=1.0}"

# Qwen3-8B rollouts through an OpenAI-compatible vLLM endpoint
: "${USE_QWEN:=1}"           # set to 0 to skip local Qwen rollouts
: "${QWEN_MODEL:=Qwen/Qwen3-8B}"
: "${QWEN_BASE_URL:=http://localhost:8000/v1}"
: "${QWEN_N:=5}"
: "${QWEN_TEMPERATURE:=1.3}"

: "${WORKERS:=8}"
: "${LIMIT:=}"               # optional: cap the number of examples (smoke test)

# -----------------------------------------------------

LIMIT_ARG=()
[[ -n "${LIMIT}" ]] && LIMIT_ARG=(--limit "${LIMIT}")

echo "[info] ${SPLIT_NAME}: ${DATA}"

echo "[1/2] ${GPT_MODEL} x ${GPT_N} (T=${GPT_TEMPERATURE})"
"${PYTHON}" rollout/rollout.py --input "${DATA}" --split-name "${SPLIT_NAME}" \
  --model "${GPT_MODEL}" --n "${GPT_N}" --temperature "${GPT_TEMPERATURE}" \
  --workers "${WORKERS}" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} \
  --out "${OUT_DIR}/gpt4o_${TAG}.jsonl"

if [[ "${USE_QWEN}" == "1" ]]; then
  echo "[2/2] ${QWEN_MODEL} x ${QWEN_N} (T=${QWEN_TEMPERATURE}, thinking disabled)"
  "${PYTHON}" rollout/rollout.py --input "${DATA}" --split-name "${SPLIT_NAME}" \
    --model "${QWEN_MODEL}" --base-url "${QWEN_BASE_URL}" \
    --api-key-env VLLM_API_KEY --disable-thinking \
    --n "${QWEN_N}" --temperature "${QWEN_TEMPERATURE}" \
    --workers "${WORKERS}" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} \
    --out "${OUT_DIR}/qwen3_${TAG}.jsonl"
else
  echo "[2/2] skipped (USE_QWEN=0)"
fi

echo "[done] Rollouts written to ${OUT_DIR}/"
