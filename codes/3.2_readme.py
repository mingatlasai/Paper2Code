import argparse
import json
import os
import sys
from typing import Dict, List
import textwrap

from llm_backend import chat
from utils import extract_planning, read_all_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, required=True)
    parser.add_argument("--gpt_version", type=str, default="codex")
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str, default="")
    parser.add_argument("--pdf_latex_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_repo_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_paper_content(args: argparse.Namespace) -> str:
    if args.paper_format == "JSON" and args.pdf_json_path:
        with open(args.pdf_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        abstract = ""
        if isinstance(data, dict):
            abstract = str(data.get("abstract", ""))
            if not abstract:
                pdf_parse = data.get("pdf_parse", {})
                if isinstance(pdf_parse, dict):
                    abs_list = pdf_parse.get("abstract", [])
                    if isinstance(abs_list, list) and len(abs_list) > 0 and isinstance(abs_list[0], dict):
                        abstract = str(abs_list[0].get("text", ""))
        payload = {
            "paper_id": data.get("paper_id", "") if isinstance(data, dict) else "",
            "title": data.get("title", "") if isinstance(data, dict) else "",
            "abstract": abstract,
        }
        return json.dumps(payload, ensure_ascii=False)
    if args.paper_format == "LaTeX" and args.pdf_latex_path:
        with open(args.pdf_latex_path, "r", encoding="utf-8") as f:
            return f.read()[:6000]
    return ""


def get_file_tree(files: List[str]) -> str:
    tree: Dict[str, dict] = {}
    for path in sorted(files):
        node = tree
        parts = [p for p in path.split("/") if p]
        for part in parts:
            node = node.setdefault(part, {})

    lines: List[str] = ["."]

    def walk(node: Dict[str, dict], prefix: str) -> None:
        keys = sorted(node.keys())
        for idx, key in enumerate(keys):
            is_last = idx == len(keys) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{key}")
            next_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            walk(node[key], next_prefix)

    walk(tree, "")
    return "\n".join(lines)


def select_code_context(all_files: Dict[str, str]) -> str:
    prioritized = [
        "config.yaml",
        "reproduce.sh",
        "requirements.txt",
        "main.py",
        "app.py",
        "train.py",
        "trainer.py",
        "evaluation.py",
        "README.md",
    ]

    selected_chunks: List[str] = []

    for path in prioritized:
        if path in all_files:
            snippet = all_files[path][:2200]
            selected_chunks.append(f"## File: {path}\n```text\n{snippet}\n```")

    # Add brief snippets from a few additional source files for better usage guidance.
    extra_candidates = sorted(
        [p for p in all_files.keys() if p.endswith((".py", ".sh", ".yaml", ".yml")) and p not in prioritized]
    )[:8]
    for path in extra_candidates:
        content = all_files[path]
        snippet = content[:1800]
        selected_chunks.append(f"## File: {path} (snippet)\n```text\n{snippet}\n```")

    return "\n\n".join(selected_chunks)


def build_command_examples(all_files: Dict[str, str]) -> str:
    has_main = "main.py" in all_files
    has_reproduce = "reproduce.sh" in all_files
    has_requirements = "requirements.txt" in all_files
    has_tests = any(path.startswith("tests/") for path in all_files.keys())

    lines: List[str] = []
    lines.append("# Candidate command examples (adapt to repo paths)")
    if has_requirements:
        lines.append("pip install -r requirements.txt")
    lines.append("python -m venv .venv")
    lines.append("source .venv/bin/activate")

    if has_reproduce:
        lines.append("bash reproduce.sh")

    if has_main:
        lines.append("python main.py --stage preprocess --config configs/default.yaml")
        lines.append("python main.py --stage pretrain --config configs/default.yaml")
        lines.append("python main.py --stage embed --config configs/default.yaml")
        lines.append("python main.py --stage eval --config configs/default.yaml")

    pipeline_paths = [
        "src/pipelines/run_preprocess.py",
        "src/pipelines/run_pretrain.py",
        "src/pipelines/run_embed.py",
        "src/pipelines/run_eval.py",
    ]
    for p in pipeline_paths:
        if p in all_files:
            lines.append(f"python {p}")

    if has_tests:
        lines.append("pytest -q")
    return "\n".join(lines)


def derive_repo_insights(all_files: Dict[str, str]) -> str:
    has_main = "main.py" in all_files
    has_pipelines = any(path.startswith("src/pipelines/") for path in all_files)
    has_train = any(path.startswith("src/train/") for path in all_files)
    has_eval = any(path.startswith("src/eval/") for path in all_files)
    has_data = any(path.startswith("src/data/") for path in all_files)
    has_configs = any(path.startswith("configs/") for path in all_files)

    points: List[str] = []
    if has_main:
        points.append("- Entry point appears to be `main.py`, likely stage-dispatch based.")
    if has_pipelines:
        points.append("- Pipeline scripts exist under `src/pipelines/` for preprocess/pretrain/embed/eval.")
    if has_train:
        points.append("- Training logic is modularized in `src/train/` (callbacks/checkpoint/module split).")
    if has_eval:
        points.append("- Evaluation is separated in `src/eval/` (linear probe/survival/retrieval/prompting).")
    if has_data:
        points.append("- Data responsibilities are encapsulated in `src/data/` (manifest/split/feature stores).")
    if has_configs:
        points.append("- Config composition is organized under `configs/` with stage-specific sub-configs.")
    if not points:
        points.append("- Repository structure is minimal; infer run entrypoints from available scripts.")
    return "\n".join(points)


def is_good_readme(content: str) -> bool:
    text = content.strip()
    if len(text) < 1200:
        return False
    if text.lower().startswith("updated the readme"):
        return False
    required_markers = ["# ", "## ", "```bash"]
    return all(marker in text for marker in required_markers)


def fallback_readme(
    paper_name: str,
    file_tree: str,
    command_examples: str,
    all_files: Dict[str, str],
) -> str:
    has_inference = any(p in all_files for p in ["src/pipelines/run_embed.py", "src/pipelines/run_eval.py"])
    has_tests = any(path.startswith("tests/") for path in all_files)
    req_cmd = "pip install -r requirements.txt" if "requirements.txt" in all_files else "# requirements.txt not found"

    cmd_list = [line for line in command_examples.splitlines() if line and not line.startswith("#")]
    setup_cmds = [c for c in cmd_list if "venv" in c or "activate" in c or "pip install" in c]
    run_cmds = [c for c in cmd_list if c.startswith("python ") or c.startswith("bash ")]

    setup_block = "\n".join(setup_cmds[:4]) if setup_cmds else textwrap.dedent(
        """\
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        """
    ).strip()
    run_block = "\n".join(run_cmds[:8]) if run_cmds else "python main.py --stage preprocess --config configs/default.yaml"

    inference_note = (
        "Inference/embedding export is supported via `embed`/`eval` pipeline paths."
        if has_inference
        else "No dedicated inference entrypoint was detected in this generated repository."
    )
    test_block = "pytest -q" if has_tests else "# tests/ not detected in this generated repository"
    insights = derive_repo_insights(all_files)

    return f"""# {paper_name} Generated Repository

## Project Overview
This repository is an auto-generated implementation scaffold for reproducing the paper workflow and experiments.
It includes pipeline entrypoints, model/training modules, and structured YAML configuration.

## Features and Scope
- Config-driven pipeline execution
- Stage-oriented workflow (`preprocess`, `pretrain`, `embed`, `eval`) when available
- Modular code layout under `src/`
- Limitations: generated code may need manual fixes for environment/data availability and edge cases

## Repository Interpretation
{insights}

## Repository Structure
```text
{file_tree}
```

## Prerequisites
- Python 3.9+ (3.10 recommended)
- Linux/macOS shell environment
- Optional GPU/CUDA for training stages

## Environment Setup
```bash
{setup_block}
```

## Installation
```bash
{req_cmd}
```

## Configuration Guide
Primary config files:
- `config.yaml` (top-level run configuration)
- `configs/default.yaml` (stage defaults and composition)
- `configs/data/*`, `configs/model/*`, `configs/train/*`, `configs/eval/*` (sub-configs)

Suggested workflow:
1. Start from `config.yaml` or `configs/default.yaml`.
2. Update paths, dataset locations, output directories, and runtime stage.
3. Keep stage-specific files aligned with model/data assumptions.

## How To Run
### Data Preparation / Preprocess
```bash
python main.py --stage preprocess --config configs/default.yaml
```

### Training / Pretrain
```bash
python main.py --stage pretrain --config configs/default.yaml
```

### Embedding / Inference
```bash
python main.py --stage embed --config configs/default.yaml
```
{inference_note}

### Evaluation
```bash
python main.py --stage eval --config configs/default.yaml
```

### Additional Commands
```bash
{run_block}
```

### Tests / Debug
```bash
{test_block}
```

## Reproducibility Tips
- Pin dependencies and CUDA/toolchain versions.
- Use fixed seeds in config/runtime.
- Keep input manifests and split definitions versioned.
- Save run artifacts/logs/checkpoints per stage.

## Expected Outputs
- Preprocess artifacts: feature/manifest outputs under configured data/output paths
- Training artifacts: checkpoints and logs
- Evaluation artifacts: metrics reports and summaries

## Troubleshooting
- If a stage fails, run it independently with the same config.
- Validate paths in `config.yaml` and all nested configs.
- Confirm required datasets and metadata files exist.
- Run tests (`pytest -q`) to catch regressions after edits.

## Citation
Please cite the original paper and this generated implementation workflow in your project documentation.
"""


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.output_repo_dir):
        print(f"[ERROR] output_repo_dir not found: {args.output_repo_dir}")
        sys.exit(1)

    repo_dir_abs = os.path.realpath(args.output_repo_dir)
    cwd_abs = os.path.realpath(os.getcwd())
    if repo_dir_abs == cwd_abs:
        print(
            "[ERROR] output_repo_dir resolves to current project root. "
            "Refusing to overwrite root README.md. "
            "Please pass a generated repo path like repos/<paper>_repo."
        )
        sys.exit(1)

    save_path = os.path.join(repo_dir_abs, "README.md")
    if args.resume and os.path.exists(save_path):
        print(f"[RESUME] Skip README generation: {save_path}")
        return

    planning_path = os.path.join(args.output_dir, "planning_trajectories.json")
    planning_overview = ""
    if os.path.exists(planning_path):
        context_lst = extract_planning(planning_path)
        if len(context_lst) > 0:
            planning_overview = context_lst[0][:5000]

    all_files = read_all_files(
        args.output_repo_dir,
        allowed_ext=[".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".json"],
        is_print=False,
    )
    file_tree = get_file_tree(list(all_files.keys()))
    code_context = select_code_context(all_files)
    command_examples = build_command_examples(all_files)
    repo_insights = derive_repo_insights(all_files)
    paper_content = load_paper_content(args)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer writing production-quality README files. "
                "Given a generated repository and paper context, write a detailed README.md that explains how to use the repo end-to-end."
            ),
        },
        {
            "role": "user",
            "content": f"""# Context
## Paper Name
{args.paper_name}

## Planning Overview
{planning_overview}

## Paper ({args.paper_format})
{paper_content}

## Repository File Tree
{file_tree}

## Key Repository Files
{code_context}

## Repository Interpretation
{repo_insights}

## Command Examples
{command_examples}

# Instruction
Write a complete, detailed `README.md` for this generated repository.

Required sections:
1. Project overview
2. Features and scope (what is implemented and limitations)
2.1 Repository interpretation (architecture and module responsibilities)
3. Repository structure (table-like listing of major files/folders)
4. Prerequisites (Python version, system dependencies)
5. Environment setup (venv/conda commands)
6. Installation
7. Configuration guide (`config.yaml` explained key-by-key)
8. How to run:
   - data preparation
   - training
   - evaluation
   - inference (if supported)
9. Reproducibility tips and expected outputs/artifacts
10. Troubleshooting
11. Citation

Rules:
- Be specific to this repository; do not write generic placeholders.
- Provide concrete commands aligned with the actual files and config.
- If a step is unavailable in the generated code, explicitly state that limitation.
- Include runnable command examples in fenced code blocks with language tag `bash`.
- Include at least 6 distinct command snippets (setup + at least 4 run commands + test/debug command).
- Every command snippet must be copy-paste ready.
- For repository structure, use a tree-style code block (not a flat path list).
- Output markdown only. Do not wrap in code fences.
""",
        },
    ]

    readme_content = ""
    try:
        completion = chat(messages, args.gpt_version)
        completion_json = json.loads(completion.model_dump_json())
        readme_content = completion_json["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[WARN] LLM README generation failed: {exc}")

    if not is_good_readme(readme_content):
        print("[WARN] LLM README output failed quality checks. Using local fallback template.")
        readme_content = fallback_readme(
            paper_name=args.paper_name,
            file_tree=file_tree,
            command_examples=command_examples,
            all_files=all_files,
        )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    artifact_output_dir = os.path.join(args.output_dir, "coding_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)
    with open(os.path.join(artifact_output_dir, "README_generation.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    main()
