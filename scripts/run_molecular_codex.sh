#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; resolve paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Codex CLI backend
export P2C_PROVIDER="${P2C_PROVIDER:-codex}"
export P2C_CODEX_CMD="${P2C_CODEX_CMD:-codex exec}"

PAPER_NAME="multimodal_foundation_model"
PAPER_PDF="/Users/mingatlas/Projects/papers/Molecular-driven Foundation Model for Oncologic Pathology.pdf"

# PDF -> JSON converter config (s2orc-doc2json)
# Set this if your clone is elsewhere.
S2ORC_DIR="${S2ORC_DIR:-${ROOT_DIR}/s2orc-doc2json}"
S2ORC_TEMP_DIR="${S2ORC_TEMP_DIR:-${S2ORC_DIR}/temp_dir/paper_coder}"
S2ORC_OUTPUT_DIR="${S2ORC_OUTPUT_DIR:-${S2ORC_DIR}/output_dir/paper_coder}"

# Optional override: if provided and exists, skip conversion and use this file directly.
RAW_JSON="${RAW_JSON:-}"

OUTPUT_DIR="${ROOT_DIR}/repos/${PAPER_NAME}"
OUTPUT_REPO_DIR="${ROOT_DIR}/repos/${PAPER_NAME}_repo"
PDF_JSON_CLEANED_PATH="${OUTPUT_DIR}/${PAPER_NAME}_cleaned.json"
RESUME="${RESUME:-1}"

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

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
    python -m doc2json.grobid2json.process_pdf \
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
if [[ "${RESUME}" == "1" && -f "${PDF_JSON_CLEANED_PATH}" ]]; then
  echo "[RESUME] Skip preprocess: ${PDF_JSON_CLEANED_PATH} exists"
else
  python "${ROOT_DIR}/codes/0_pdf_process.py" \
    --input_json_path "${RAW_JSON}" \
    --output_json_path "${PDF_JSON_CLEANED_PATH}"
fi

echo "------- PaperCoder (Codex) -------"
if [[ "${RESUME}" == "1" && -f "${OUTPUT_DIR}/planning_trajectories.json" ]]; then
  echo "[RESUME] Skip planning: planning_trajectories.json exists"
else
  python "${ROOT_DIR}/codes/1_planning.py" \
    --paper_name "${PAPER_NAME}" \
    --gpt_version "codex" \
    --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
    --output_dir "${OUTPUT_DIR}"
fi

if [[ "${RESUME}" == "1" && -f "${OUTPUT_DIR}/planning_config.yaml" ]]; then
  echo "[RESUME] Skip config extraction: planning_config.yaml exists"
else
  python "${ROOT_DIR}/codes/1.1_extract_config.py" \
    --paper_name "${PAPER_NAME}" \
    --output_dir "${OUTPUT_DIR}"
fi

cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"

python "${ROOT_DIR}/codes/2_analyzing.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}"

python "${ROOT_DIR}/codes/3_coding.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_repo_dir "${OUTPUT_REPO_DIR}" \
  "${RESUME_ARGS[@]}"

echo "------- Environment Files Generation -------"
python "${ROOT_DIR}/codes/3.3_env_files.py" \
  --output_repo_dir "${OUTPUT_REPO_DIR}" \
  --python_version "3.10"

echo "------- README Generation -------"
python "${ROOT_DIR}/codes/3.2_readme.py" \
  --paper_name "${PAPER_NAME}" \
  --gpt_version "codex" \
  --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_repo_dir "${OUTPUT_REPO_DIR}" \
  "${RESUME_ARGS[@]}"

echo "[DONE] Generated repo: ${OUTPUT_REPO_DIR}"
