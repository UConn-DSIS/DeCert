#!/usr/bin/env bash
# Score the rollouts of one data split with the decision-certificate reward.
# Per example: 1 analyzer call (material issues D*); per parseable judgment:
# 1 grounding + 1 validity + 1 matcher call.
set -euo pipefail

# -------------------- User config --------------------

: "${PYTHON:=python3}"

# Must be the same DATA / SPLIT_NAME pair used for the rollouts
: "${DATA:?Set DATA to the local HelpSteer3 JSONL used for rollouts}"
: "${SPLIT_NAME:=train}"

# Rollout files of this split (space-separated; merged into one group per example)
: "${ROLLOUTS:=rollouts/gpt4o_train.jsonl rollouts/qwen3_train.jsonl}"

# GPT-4o analyzes material issues. Local Qwen3-14B handles the three
# per-judgment checks: grounding, validity, and issue--claim matching.
: "${ANALYZER_MODEL:=gpt-4o}"
: "${VERIFIER_MODEL:=Qwen/Qwen3-14B}"
: "${MATCHER_MODEL:=Qwen/Qwen3-14B}"

# GPT-4o uses the OpenAI endpoint; Qwen3-14B uses one local vLLM server.
: "${ANALYZER_BASE_URL:=}"
: "${VERIFIER_BASE_URL:=http://localhost:8000/v1}"
: "${MATCHER_BASE_URL:=http://localhost:8000/v1}"
: "${ANALYZER_API_KEY_ENV:=OPENAI_API_KEY}"
: "${VERIFIER_API_KEY_ENV:=VLLM_API_KEY}"
: "${MATCHER_API_KEY_ENV:=VLLM_API_KEY}"
: "${ANALYZER_DISABLE_THINKING:=0}"
: "${VERIFIER_DISABLE_THINKING:=1}"
: "${MATCHER_DISABLE_THINKING:=1}"
: "${ANALYZER_REASONING_EFFORT:=}"
: "${VERIFIER_REASONING_EFFORT:=}"
: "${MATCHER_REASONING_EFFORT:=}"

: "${LAMBDA:=0.5}"
: "${TAU_LOC:=1.0}"
: "${WORKERS:=8}"
: "${OUT:=scored/train.jsonl}"

# -----------------------------------------------------

read -r -a ROLLOUT_FILES <<< "${ROLLOUTS}"
EXTRA=()
[[ -n "${ANALYZER_BASE_URL}" ]] && EXTRA+=(--analyzer-base-url "${ANALYZER_BASE_URL}")
[[ -n "${VERIFIER_BASE_URL}" ]] && EXTRA+=(--base-url "${VERIFIER_BASE_URL}")
[[ -n "${MATCHER_BASE_URL}" ]] && EXTRA+=(--matcher-base-url "${MATCHER_BASE_URL}")
[[ "${ANALYZER_DISABLE_THINKING}" == "1" ]] && EXTRA+=(--analyzer-disable-thinking)
[[ "${ANALYZER_DISABLE_THINKING}" == "0" ]] && EXTRA+=(--no-analyzer-disable-thinking)
[[ "${VERIFIER_DISABLE_THINKING}" == "1" ]] && EXTRA+=(--disable-thinking)
[[ "${VERIFIER_DISABLE_THINKING}" == "0" ]] && EXTRA+=(--no-disable-thinking)
[[ "${MATCHER_DISABLE_THINKING}" == "1" ]] && EXTRA+=(--matcher-disable-thinking)
[[ "${MATCHER_DISABLE_THINKING}" == "0" ]] && EXTRA+=(--no-matcher-disable-thinking)
[[ -n "${ANALYZER_REASONING_EFFORT}" ]] && \
  EXTRA+=(--analyzer-reasoning-effort "${ANALYZER_REASONING_EFFORT}")
[[ -n "${VERIFIER_REASONING_EFFORT}" ]] && \
  EXTRA+=(--reasoning-effort "${VERIFIER_REASONING_EFFORT}")
[[ -n "${MATCHER_REASONING_EFFORT}" ]] && \
  EXTRA+=(--matcher-reasoning-effort "${MATCHER_REASONING_EFFORT}")

echo "[info] ${SPLIT_NAME}: analyzer=${ANALYZER_MODEL}, verifier=${VERIFIER_MODEL}, matcher=${MATCHER_MODEL}"
"${PYTHON}" scoring/score_judgments.py \
  --data "${DATA}" --split-name "${SPLIT_NAME}" \
  --rollouts "${ROLLOUT_FILES[@]}" \
  --analyzer-model "${ANALYZER_MODEL}" \
  --verifier-model "${VERIFIER_MODEL}" --matcher-model "${MATCHER_MODEL}" \
  --analyzer-api-key-env "${ANALYZER_API_KEY_ENV}" \
  --verifier-api-key-env "${VERIFIER_API_KEY_ENV}" \
  --matcher-api-key-env "${MATCHER_API_KEY_ENV}" \
  --lam "${LAMBDA}" --local-validity-threshold "${TAU_LOC}" \
  --workers "${WORKERS}" ${EXTRA[@]+"${EXTRA[@]}"} \
  --out "${OUT}"

echo "[done] Scored groups written to ${OUT}"
