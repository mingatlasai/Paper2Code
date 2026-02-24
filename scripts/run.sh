# export P2C_PROVIDER="codex"  # or "claude"
# export P2C_CODEX_CMD="codex exec"
# export P2C_CLAUDE_CMD="claude -p"

MODEL_NAME="codex"

PAPER_NAME="Transformer"
PDF_PATH="../examples/Transformer.pdf" # .pdf
PDF_JSON_PATH="../examples/Transformer.json" # .json
PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Transformer"
OUTPUT_REPO_DIR="../outputs/Transformer_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

python ../codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${MODEL_NAME} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}


python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${MODEL_NAME} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${MODEL_NAME} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \

echo "------- Environment Files -------"
python ../codes/3.3_env_files.py \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
    --python_version 3.10
