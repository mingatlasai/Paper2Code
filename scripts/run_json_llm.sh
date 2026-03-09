MODEL_NAME="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
TP_SIZE=2
PIPELINE_MODE="${P2C_PIPELINE_MODE:-original}"
PROMPT_SET="${P2C_PROMPT_SET:-${PIPELINE_MODE}}"
MAX_REPAIR_ATTEMPTS="${P2C_MAX_REPAIR_ATTEMPTS:-2}"
PYTHON_BIN="${P2C_PYTHON_BIN:-python3}"

PAPER_NAME="Transformer"
PDF_PATH="../examples/Transformer.pdf" # .pdf
PDF_JSON_PATH="../examples/Transformer.json" # .json
PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Transformer_dscoder"
OUTPUT_REPO_DIR="../outputs/Transformer_dscoder_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

${PYTHON_BIN} ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
${PYTHON_BIN} ../codes/structured_extraction.py \
    --paper_name $PAPER_NAME \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_set ${PROMPT_SET}
fi

${PYTHON_BIN} ../codes/1_planning_llm.py \
    --paper_name $PAPER_NAME \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --pipeline_mode ${PIPELINE_MODE} \
    --prompt_set ${PROMPT_SET}

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
${PYTHON_BIN} ../codes/planning_verifier.py \
    --paper_name $PAPER_NAME \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_set ${PROMPT_SET}

${PYTHON_BIN} ../codes/planning_refiner.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --prompt_set ${PROMPT_SET}
fi

${PYTHON_BIN} ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

${PYTHON_BIN} ../codes/2_analyzing_llm.py \
    --paper_name $PAPER_NAME \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --pipeline_mode ${PIPELINE_MODE} \
    --prompt_set ${PROMPT_SET}

${PYTHON_BIN} ../codes/3_coding_llm.py  \
    --paper_name $PAPER_NAME \
    --model_name ${MODEL_NAME} \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
    --pipeline_mode ${PIPELINE_MODE} \
    --prompt_set ${PROMPT_SET}

if [[ "${PIPELINE_MODE}" == "enhanced" ]]; then
${PYTHON_BIN} ../codes/execution_test.py \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR}

for attempt in $(seq 1 ${MAX_REPAIR_ATTEMPTS}); do
    EXECUTION_STATUS=$(${PYTHON_BIN} -c "import json,sys; print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('status', 'failed'))" "${OUTPUT_DIR}/execution_test_report.json")
    if [[ "${EXECUTION_STATUS}" == "passed" ]]; then
        break
    fi

    ${PYTHON_BIN} ../codes/repair_agent.py \
        --output_dir ${OUTPUT_DIR} \
        --output_repo_dir ${OUTPUT_REPO_DIR} \
        --model_name ${MODEL_NAME} \
        --tp_size ${TP_SIZE} \
        --max_attempts ${MAX_REPAIR_ATTEMPTS} \
        --attempt_index ${attempt}

    ${PYTHON_BIN} ../codes/execution_test.py \
        --output_dir ${OUTPUT_DIR} \
        --output_repo_dir ${OUTPUT_REPO_DIR}
done
fi
