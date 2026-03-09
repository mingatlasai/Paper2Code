from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

from pipeline_utils import load_json_file, load_planning_bundle, read_text_file, write_json_file
from stage_llm import StageLLM
from utils import read_python_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair a generated repo using execution errors.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_repo_dir", type=str, required=True)
    parser.add_argument("--gpt_version", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--attempt_index", type=int, default=1)
    return parser.parse_args()


def parse_and_apply_changes(response: str, repo_dir: str) -> List[Dict[str, str]]:
    change_log: List[Dict[str, str]] = []
    file_blocks = re.split(r"Filename:\s*([^\n]+)", response)
    if len(file_blocks) < 3:
        return change_log

    for index in range(1, len(file_blocks), 2):
        filename = file_blocks[index].strip()
        file_content_block = file_blocks[index + 1]
        filepath = os.path.join(repo_dir, filename)
        if not os.path.exists(filepath):
            change_log.append({"file": filename, "status": "missing"})
            continue

        search_replace_pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(search_replace_pattern, file_content_block, re.DOTALL)
        if not matches:
            change_log.append({"file": filename, "status": "no_match"})
            continue

        file_content = read_text_file(filepath)
        modified = False
        for search_text, replace_text in matches:
            source = search_text.strip()
            target = replace_text.strip()
            if source not in file_content:
                continue
            file_content = file_content.replace(source, target, 1)
            modified = True

        if not modified:
            change_log.append({"file": filename, "status": "unchanged"})
            continue

        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(file_content)
        change_log.append({"file": filename, "status": "patched"})

    return change_log


def main() -> int:
    args = parse_args()
    report = load_json_file(os.path.join(args.output_dir, "execution_test_report.json"))
    if not report or report.get("status") == "passed":
        write_json_file(
            os.path.join(args.output_dir, f"repair_attempt_{args.attempt_index}.json"),
            {"status": "skipped", "reason": "No failing execution report."},
        )
        return 0

    if args.attempt_index > args.max_attempts:
        write_json_file(
            os.path.join(args.output_dir, f"repair_attempt_{args.attempt_index}.json"),
            {"status": "skipped", "reason": "Maximum repair attempts exceeded."},
        )
        return 0

    planning_bundle = load_planning_bundle(args.output_dir)
    structured_paper = load_json_file(os.path.join(args.output_dir, "structured_paper.json"))
    repo_files = read_python_files(args.output_repo_dir)
    config_yaml = read_text_file(os.path.join(args.output_repo_dir, "config.yaml"))

    codes = ""
    for file_name, code in repo_files.items():
        codes += f"```python\n## File name: {file_name}\n{code}\n```\n\n"
    if config_yaml:
        codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"

    system_prompt = """You are a code repair agent. Fix the generated repository using the runtime failures.
Return only SEARCH/REPLACE patches in the following format:

Filename: path/to/file.py
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE

Use minimal edits and preserve the existing architecture."""
    user_prompt = f"""## Structured Extraction
{json.dumps(structured_paper, indent=2)}

-----

## Planning Bundle
{json.dumps(planning_bundle, indent=2)}

-----

## Repository
{codes}

-----

## Execution Report
{json.dumps(report, indent=2)}
"""

    try:
        runner = StageLLM(
            gpt_version=args.gpt_version or None,
            model_name=args.model_name or None,
            tp_size=args.tp_size,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
        )
        result = runner.run(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        raw_text = result["text"]
    except Exception as exc:  # pragma: no cover - runtime dependent
        write_json_file(
            os.path.join(args.output_dir, f"repair_attempt_{args.attempt_index}.json"),
            {"status": "failed", "error": str(exc)},
        )
        return 0

    changes = parse_and_apply_changes(raw_text, args.output_repo_dir)
    write_json_file(
        os.path.join(args.output_dir, f"repair_attempt_{args.attempt_index}.json"),
        {
            "status": "completed" if any(item["status"] == "patched" for item in changes) else "no_change",
            "changes": changes,
            "raw_response": raw_text,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
