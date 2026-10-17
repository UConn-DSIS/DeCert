#!/usr/bin/env bash
# Build SFT and DPO training data from a local HelpSteer3 JSONL file.
set -euo pipefail

# -------------------- User config --------------------

: "${PYTHON:=python3}"
: "${DATA:?Set DATA to the local HelpSteer3 JSONL used for scoring}"
: "${SCORED_DIR:=scored}"
: "${OUT_DIR:=training_data}"

# Minimum Q_proc gap for rationale (same-verdict) preference pairs
: "${DELTA:=0.25}"
: "${DELTA_TAG:=}"

# Process-score aggregation: geometric (paper) or arithmetic (ablation)
: "${AGGREGATION:=geometric}"

# -----------------------------------------------------

if [[ -z "${DELTA_TAG}" ]]; then
  DELTA_TAG="$("${PYTHON}" -c '
import decimal
import sys

scaled = decimal.Decimal(sys.argv[1]) * 100
if scaled != scaled.to_integral_value():
    raise SystemExit("DELTA must use at most two decimal places, or set DELTA_TAG")
print(f"d{int(scaled):03d}")
' "${DELTA}")"
fi

# tag | released data file | split name | scored file
SPLITS=(
  "train|${DATA}|train|${SCORED_DIR}/train.jsonl"
)

mkdir -p "${OUT_DIR}"
SFT_FILES=()
DPO_FILES=()
for entry in "${SPLITS[@]}"; do
  IFS="|" read -r tag data split scored <<< "${entry}"
  echo "[build] ${tag}"
  "${PYTHON}" construct/build_sft_data.py --data "${data}" --split-name "${split}" \
    --scored "${scored}" --out "${OUT_DIR}/${tag}_sft.jsonl"
  # delta = 0 keeps the full pool; the threshold is applied at merge time
  "${PYTHON}" construct/build_dpo_data.py --data "${data}" --split-name "${split}" \
    --scored "${scored}" --same-verdict-delta 0 \
    --process-aggregation "${AGGREGATION}" \
    --out "${OUT_DIR}/${tag}_dpo.jsonl"
  SFT_FILES+=("${OUT_DIR}/${tag}_sft.jsonl")
  DPO_FILES+=("${OUT_DIR}/${tag}_dpo.jsonl")
done

echo "[merge] SFT"
cat "${SFT_FILES[@]}" > "${OUT_DIR}/sft_all.jsonl"
wc -l "${OUT_DIR}/sft_all.jsonl"

echo "[merge] DPO (delta=${DELTA})"
"${PYTHON}" construct/merge_dpo_data.py --inputs "${DPO_FILES[@]}" \
  --same-verdict-delta "${DELTA}" \
  --out-mixed "${OUT_DIR}/dpo_mixed.jsonl" \
  --out-same-verdict "${OUT_DIR}/dpo_same_verdict_${DELTA_TAG}.jsonl" \
  --out-all "${OUT_DIR}/dpo_all_${DELTA_TAG}.jsonl"

echo "[done] Training data written to ${OUT_DIR}/"
