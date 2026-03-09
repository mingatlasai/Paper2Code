# Original Mode Runbook

This runbook is written for the normal case where you want to process one specific paper and the paper you have is a PDF.

Original mode preserves the baseline Paper2Code pipeline:

`paper -> planning -> analysis -> code_generation`

It does not run structured extraction, planning verification, planning refinement, execution testing, or repair.

## When To Use This

Use this runbook when:

- you want baseline behavior
- you want a clean comparison target against enhanced mode
- you have one paper PDF and want to generate one repository from it

## Result You Should Expect

At the end of the run you will have:

- an artifact directory under `outputs/<paper_name>/`
- a generated repository under `outputs/<paper_name>_repo/`

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

If `python` on your machine is Python 2:

```bash
export P2C_PYTHON_BIN="python3"
```

If you are using Codex or Claude CLI:

```bash
export P2C_PROVIDER="codex"   # or "claude"
export P2C_CODEX_CMD="codex exec"
export P2C_CLAUDE_CMD="claude -p"
```

Set baseline mode explicitly:

```bash
export P2C_PIPELINE_MODE="original"
export P2C_PROMPT_SET="original"
```

## Step 1: Pick A Paper Name

Choose one stable identifier for the paper. Keep it filesystem-safe.

Example:

```bash
PAPER_NAME="my_method_paper"
```

This name will be used for:

- `outputs/${PAPER_NAME}/`
- `outputs/${PAPER_NAME}_repo/`

## Step 2: Convert The PDF To JSON

Paper2Code works best when the PDF has already been converted into structured JSON.

If you do not already have JSON, use `s2orc-doc2json`.

### 2.1 Clone and prepare the converter

From the repository root:

```bash
git clone https://github.com/allenai/s2orc-doc2json.git
```

Start the GROBID service as documented by that project.

### 2.2 Convert your PDF

Example:

```bash
PDF_PATH="/absolute/path/to/your_paper.pdf"
mkdir -p ./s2orc-doc2json/output_dir/paper_coder

python3 ./s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
  -i "${PDF_PATH}" \
  -t ./s2orc-doc2json/temp_dir/ \
  -o ./s2orc-doc2json/output_dir/paper_coder
```

That should produce a raw JSON file named after the PDF.

## Step 3: Decide Which Runner To Use

For a specific PDF paper, the most useful starting point is:

- [run_pdf.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_pdf.sh)

That script already supports:

- direct PDF path
- automatic PDF-to-JSON conversion
- optional override with an existing raw JSON file

If you already have a cleaned or prepared JSON file and do not need PDF conversion inside the runner, you can instead edit:

- [run_json.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_json.sh)

## Step 4: Edit The Runner For Your Paper

### Option A: Use `run_pdf.sh` for a PDF

Open [run_pdf.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_pdf.sh) and update:

- `PAPER_NAME`
- `PAPER_PDF`
- `S2ORC_DIR` if your converter clone is elsewhere

If you already converted the PDF and want to skip conversion, set:

- `RAW_JSON`

You do not need to change `PIPELINE_MODE` in the script if you export it from the shell.

### Option B: Use `run_json.sh` if you already have JSON

Open [run_json.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_json.sh) and update:

- `PAPER_NAME`
- `PDF_JSON_PATH`
- `PDF_JSON_CLEANED_PATH`
- `OUTPUT_DIR`
- `OUTPUT_REPO_DIR`

## Step 5: Run The Pipeline

### PDF-first flow

```bash
export P2C_PIPELINE_MODE="original"
export P2C_PROMPT_SET="original"
export P2C_PYTHON_BIN="python3"
export P2C_PROVIDER="codex"
export P2C_CODEX_CMD="codex exec"

cd scripts
bash run_pdf.sh
```

### JSON-first flow

```bash
export P2C_PIPELINE_MODE="original"
export P2C_PROMPT_SET="original"
export P2C_PYTHON_BIN="python3"
export P2C_PROVIDER="codex"
export P2C_CODEX_CMD="codex exec"

cd scripts
bash run_json.sh
```

## What The Runner Does

In original mode the runner executes:

1. `codes/0_pdf_process.py`
2. `codes/1_planning.py`
3. `codes/1.1_extract_config.py`
4. `codes/2_analyzing.py`
5. `codes/3_coding.py`

## Output Files To Check

After the run, inspect:

- `outputs/<paper_name>/planning_bundle.json`
- `outputs/<paper_name>/planning_config.yaml`
- `outputs/<paper_name>/planning_artifacts/`
- `outputs/<paper_name>/analyzing_artifacts/`
- `outputs/<paper_name>/coding_artifacts/`
- `outputs/<paper_name>_repo/`

The generated repository is the main deliverable:

- `outputs/<paper_name>_repo/`

## If You Want To Process Multiple Papers

Process one paper at a time.

For each paper:

1. change `PAPER_NAME`
2. change the input PDF or JSON path
3. run the script
4. archive or evaluate `outputs/<paper_name>/` and `outputs/<paper_name>_repo/`

Do not reuse the same `PAPER_NAME` unless you intentionally want to overwrite or reuse outputs.

## Troubleshooting

If the run fails before planning:

- check the PDF path
- check that the JSON conversion succeeded
- check that `P2C_PYTHON_BIN` points to Python 3

If the LLM backend fails:

- check `P2C_PROVIDER`
- check `P2C_CODEX_CMD` or `P2C_CLAUDE_CMD`

If downstream stages fail:

- inspect `outputs/<paper_name>/planning_bundle.json`
- inspect `outputs/<paper_name>/task_list.json`
- inspect `outputs/<paper_name>/planning_config.yaml`

## Recommended Baseline Workflow

For a new paper PDF:

1. set a new `PAPER_NAME`
2. point the PDF runner at the paper
3. run in `original` mode
4. inspect the generated repo
5. if you want a stronger version, rerun the same paper in enhanced mode for comparison
