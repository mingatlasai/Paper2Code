# Enhanced Mode Runbook

This runbook is written for the normal case where you want to process one specific paper and the paper you have is a PDF.

Enhanced mode runs the upgraded pipeline:

`paper -> structured_extraction -> planning -> planning_verifier -> planning_refinement -> analysis -> code_generation -> execution_test -> repair_agent`

Use this when you want the stronger planning and validation path for one concrete paper.

## When To Use This

Use this runbook when:

- you have a specific paper PDF
- you want richer planning artifacts
- you want explicit assumption tracking
- you want automated post-generation execution checks
- you want bounded repair attempts after generation

## Result You Should Expect

At the end of the run you should have:

- baseline planning, analysis, and coding artifacts
- enhanced artifacts such as structured extraction and planning review
- an execution report
- optional repair attempt logs
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

Set enhanced mode explicitly:

```bash
export P2C_PIPELINE_MODE="enhanced"
export P2C_PROMPT_SET="enhanced"
export P2C_MAX_REPAIR_ATTEMPTS="2"
```

## Step 1: Pick A Paper Name

Choose one stable identifier for the paper.

Example:

```bash
PAPER_NAME="my_method_paper"
```

This will map to:

- `outputs/${PAPER_NAME}/`
- `outputs/${PAPER_NAME}_repo/`

## Step 2: Convert The PDF To JSON

Enhanced mode still starts from the paper content, and for PDFs the practical workflow is the same: convert the PDF to structured JSON first, or let the PDF-oriented runner do it.

### 2.1 Set up `s2orc-doc2json`

```bash
git clone https://github.com/allenai/s2orc-doc2json.git
```

Start the required GROBID service according to that project.

### 2.2 Convert your PDF manually if needed

```bash
PDF_PATH="/absolute/path/to/your_paper.pdf"
mkdir -p ./s2orc-doc2json/output_dir/paper_coder

python3 ./s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
  -i "${PDF_PATH}" \
  -t ./s2orc-doc2json/temp_dir/ \
  -o ./s2orc-doc2json/output_dir/paper_coder
```

If you prefer, the PDF-oriented runner can do this step for you.

## Step 3: Choose The Right Runner

For a specific paper in PDF form, the best starting point is:

- [run_pdf.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_pdf.sh)

Why:

- it accepts a direct PDF path
- it supports in-run PDF to JSON conversion
- it supports `RAW_JSON` to skip conversion if you already have JSON
- it already includes enhanced stages

If you already have JSON and want a simpler path, edit:

- [run_json.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_json.sh)

## Step 4: Edit The Runner For Your Paper

### Recommended PDF flow

Open [run_pdf.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_pdf.sh) and update:

- `PAPER_NAME`
- `PAPER_PDF`
- `S2ORC_DIR` if needed

Optional:

- `RAW_JSON` if you already converted the PDF and want to skip conversion

The script already honors:

- `P2C_PIPELINE_MODE`
- `P2C_PROMPT_SET`
- `P2C_MAX_REPAIR_ATTEMPTS`
- `P2C_PYTHON_BIN`

### JSON flow

If you already have JSON and use [run_json.sh](/Users/mingatlas/Projects/Paper2Code/scripts/run_json.sh), update:

- `PAPER_NAME`
- `PDF_JSON_PATH`
- `PDF_JSON_CLEANED_PATH`
- `OUTPUT_DIR`
- `OUTPUT_REPO_DIR`

## Step 5: Run Enhanced Mode

### PDF-first flow

```bash
export P2C_PROVIDER="codex"
export P2C_CODEX_CMD="codex exec"
export P2C_PIPELINE_MODE="enhanced"
export P2C_PROMPT_SET="enhanced"
export P2C_MAX_REPAIR_ATTEMPTS="2"
export P2C_PYTHON_BIN="python3"

cd scripts
bash run_pdf.sh
```

### JSON-first flow

```bash
export P2C_PROVIDER="codex"
export P2C_CODEX_CMD="codex exec"
export P2C_PIPELINE_MODE="enhanced"
export P2C_PROMPT_SET="enhanced"
export P2C_MAX_REPAIR_ATTEMPTS="2"
export P2C_PYTHON_BIN="python3"

cd scripts
bash run_json.sh
```

## What Enhanced Mode Actually Runs

For one paper, the pipeline stages are:

1. `codes/0_pdf_process.py`
2. `codes/structured_extraction.py`
3. `codes/1_planning.py`
4. `codes/planning_verifier.py`
5. `codes/planning_refiner.py`
6. `codes/1.1_extract_config.py`
7. `codes/2_analyzing.py`
8. `codes/3_coding.py`
9. `codes/execution_test.py`
10. `codes/repair_agent.py` up to `P2C_MAX_REPAIR_ATTEMPTS`

## Enhanced Artifacts To Inspect

For your specific paper, the most useful files are:

- `outputs/<paper_name>/structured_paper.json`
- `outputs/<paper_name>/planning_bundle.json`
- `outputs/<paper_name>/planning_review.json`
- `outputs/<paper_name>/refined_planning_bundle.json`
- `outputs/<paper_name>/planning_config.yaml`
- `outputs/<paper_name>/execution_test_report.json`
- `outputs/<paper_name>/repair_attempt_1.json`

The final generated repository is still:

- `outputs/<paper_name>_repo/`

## What To Look For In The Artifacts

### Structured extraction

Check whether `structured_paper.json` contains:

- model components
- datasets
- metrics
- training protocol
- referenced algorithms
- assumptions
- unclear items

### Planning review

Check whether `planning_review.json` identified:

- missing modules
- dependency issues
- missing training steps
- evaluation gaps
- API inconsistencies

### Refined planning

Check whether `refined_planning_bundle.json` includes:

- a usable task list
- config with required sections
- explicit assumptions
- clearer planning than the baseline bundle

### Execution test

Check `execution_test_report.json` for:

- Python compile failures
- smoke-run failures
- total failure count

### Repair attempts

Check `repair_attempt_*.json` for:

- whether patches were proposed
- whether any files were modified
- the raw repair response

## Assumption Tracking

Enhanced mode is intended to make assumptions explicit across the full run.

Examples:

- `ASSUMPTION A1: optimizer assumed AdamW`
- `ASSUMPTION A2: batch size inferred as 32`

For your paper, verify that assumptions appear in:

- `structured_paper.json`
- `planning_bundle.json`
- `refined_planning_bundle.json`
- `planning_config.yaml`
- analysis artifacts

If the paper is incomplete, enhanced mode should preserve that uncertainty instead of silently inventing details.

## If You Process Several Papers

Handle one paper at a time.

For each paper:

1. assign a fresh `PAPER_NAME`
2. point the script at the new PDF
3. run enhanced mode
4. inspect the generated repo and reports

Avoid reusing the same output paths unless you intentionally want to overwrite prior results.

## Troubleshooting

If structured extraction is missing:

- confirm `P2C_PIPELINE_MODE="enhanced"`
- confirm `P2C_PROMPT_SET="enhanced"`
- inspect `outputs/<paper_name>/structured_paper.json`

If refinement does not appear to change anything:

- inspect `outputs/<paper_name>/planning_review.json`
- inspect `outputs/<paper_name>/refined_planning_bundle.json`

If execution test fails repeatedly:

- inspect `execution_test_report.json`
- inspect files in `outputs/<paper_name>_repo/`
- inspect `repair_attempt_*.json`
- do not blindly raise `P2C_MAX_REPAIR_ATTEMPTS` unless you have a reason

If PDF conversion fails:

- verify `PAPER_PDF`
- verify `S2ORC_DIR`
- verify the converter service is running

## Recommended Workflow For A New Paper PDF

For one new paper:

1. copy or edit the PDF-oriented runner
2. set `PAPER_NAME`
3. set `PAPER_PDF`
4. export enhanced-mode environment variables
5. run the script
6. inspect `structured_paper.json`, `planning_review.json`, and `execution_test_report.json`
7. inspect the generated repo under `outputs/<paper_name>_repo/`

If you need a strict baseline comparison, rerun the same paper in original mode with the same backend and compare the two output directories.
