#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; resolve paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Codex CLI backend
export P2C_PROVIDER="${P2C_PROVIDER:-codex}"
export P2C_CODEX_CMD="${P2C_CODEX_CMD:-codex exec}"
PIPELINE_MODE="${P2C_PIPELINE_MODE:-original}"
PROMPT_SET="${P2C_PROMPT_SET:-${PIPELINE_MODE}}"
MAX_REPAIR_ATTEMPTS="${P2C_MAX_REPAIR_ATTEMPTS:-2}"
PYTHON_BIN="${P2C_PYTHON_BIN:-python3}"

PAPER_NAME="molecular_foundation_model"
PAPER_PDF="/Users/mingatlas/Projects/papers/Molecular-driven Foundation Model for Oncologic Pathology.pdf"

# PDF -> JSON converter config (s2orc-doc2json)
# Set this if your clone is elsewhere.
S2ORC_DIR="${S2ORC_DIR:-${ROOT_DIR}/s2orc-doc2json}"
S2ORC_TEMP_DIR="${S2ORC_TEMP_DIR:-${S2ORC_DIR}/temp_dir/paper_coder}"
S2ORC_OUTPUT_DIR="${S2ORC_OUTPUT_DIR:-${S2ORC_DIR}/output_dir/paper_coder}"

# Optional override: if provided and exists, skip conversion and use this file directly.
RAW_JSON="${RAW_JSON:-}"

OUTPUT_DIR="${ROOT_DIR}/outputs/${PAPER_NAME}"
OUTPUT_REPO_DIR="${ROOT_DIR}/outputs/${PAPER_NAME}_repo"
PDF_JSON_CLEANED_PATH="${OUTPUT_DIR}/${PAPER_NAME}_cleaned.json"

if [[ -z "${RAW_JSON}" || ! -f "${RAW_JSON}" ]]; then
  echo "------- Convert PDF -> JSON (s2orc-doc2json) -------"
  if [[ ! -d "${S2ORC_DIR}" ]]; then
    echo "[ERROR] s2orc-doc2json directory not found: ${S2ORC_DIR}"
    echo "Set S2ORC_DIR to your s2orc-doc2json path and re-run."
    exit 1
  fi
  if [[ ! -f "${S2ORC_DIR}/doc2json/grobid2json/process_pdf.py" ]]; then
    echo "[ERROR] process_pdf.py not found under: ${S2ORC_DIR}"
    exit 1
  fi
  if [[ ! -f "${PAPER_PDF}" ]]; then
    echo "[ERROR] PAPER_PDF not found: ${PAPER_PDF}"
    exit 1
  fi

  mkdir -p "${S2ORC_TEMP_DIR}" "${S2ORC_OUTPUT_DIR}"
  (
    cd "${S2ORC_DIR}"
    "${PYTHON_BIN}" -m doc2json.grobid2json.process_pdf \
      -i "${PAPER_PDF}" \
      -t "${S2ORC_TEMP_DIR}" \
      -o "${S2ORC_OUTPUT_DIR}"
  )

  RAW_JSON="${S2ORC_OUTPUT_DIR}/$(basename "${PAPER_PDF%.*}").json"
fi

if [[ ! -f "${RAW_JSON}" ]]; then
  echo "[ERROR] Converted JSON not found: ${RAW_JSON}"
  echo "Check s2orc output directory: ${S2ORC_OUTPUT_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_REPO_DIR}"

echo "------- Preprocess -------"
"${PYTHON_BIN}" "${ROOT_DIR}/codes/0_pdf_process.py" \
  --input_json_path "${RAW_JSON}" \
  --output_json_path "${PDF_JSON_CLEANED_PATH}"

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/codes/structured_extraction.py" \
    --paper_name "${PAPER_NAME}" \
    --gpt_version "codex" \
    --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_set "${PROMPT_SET}"
fi

echo "------- PaperCoder (Codex) -------"
"${PYTHON_BIN}" "${ROOT_DIR}/codes/1_planning.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --pipeline_mode "${PIPELINE_MODE}" \
  --prompt_set "${PROMPT_SET}"

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/codes/planning_verifier.py" \
    --paper_name "${PAPER_NAME}" \
    --gpt_version "codex" \
    --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_set "${PROMPT_SET}"

  "${PYTHON_BIN}" "${ROOT_DIR}/codes/planning_refiner.py" \
    --output_dir "${OUTPUT_DIR}" \
    --gpt_version "codex" \
    --prompt_set "${PROMPT_SET}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/codes/1.1_extract_config.py" \
  --paper_name "${PAPER_NAME}" \
  --output_dir "${OUTPUT_DIR}"

cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"

"${PYTHON_BIN}" "${ROOT_DIR}/codes/2_analyzing.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --pipeline_mode "${PIPELINE_MODE}" \
  --prompt_set "${PROMPT_SET}"

"${PYTHON_BIN}" "${ROOT_DIR}/codes/3_coding.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_repo_dir "${OUTPUT_REPO_DIR}" \
  --pipeline_mode "${PIPELINE_MODE}" \
  --prompt_set "${PROMPT_SET}"

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/codes/execution_test.py" \
    --output_dir "${OUTPUT_DIR}" \
    --output_repo_dir "${OUTPUT_REPO_DIR}"

  for attempt in $(seq 1 "${MAX_REPAIR_ATTEMPTS}"); do
    EXECUTION_STATUS=$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('status', 'failed'))" "${OUTPUT_DIR}/execution_test_report.json")
    if [[ "${EXECUTION_STATUS}" == "passed" ]]; then
      break
    fi

    "${PYTHON_BIN}" "${ROOT_DIR}/codes/repair_agent.py" \
      --output_dir "${OUTPUT_DIR}" \
      --output_repo_dir "${OUTPUT_REPO_DIR}" \
      --gpt_version "codex" \
      --max_attempts "${MAX_REPAIR_ATTEMPTS}" \
      --attempt_index "${attempt}"

    "${PYTHON_BIN}" "${ROOT_DIR}/codes/execution_test.py" \
      --output_dir "${OUTPUT_DIR}" \
      --output_repo_dir "${OUTPUT_REPO_DIR}"
  done
fi

echo "[DONE] Generated repo: ${OUTPUT_REPO_DIR}"
