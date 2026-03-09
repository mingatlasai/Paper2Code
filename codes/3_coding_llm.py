from __future__ import annotations

import argparse
import os

from pipeline_utils import (
    assumptions_to_text,
    load_paper_content,
    load_planning_bundle,
    load_prompt,
    load_structured_paper,
    read_text_file,
    render_prompt,
    write_text_file,
)
from stage_llm import StageLLM
from utils import extract_code_from_content, extract_code_from_content2, print_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str)
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str)
    parser.add_argument("--pdf_latex_path", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_repo_dir", type=str, default="")
    parser.add_argument("--pipeline_mode", type=str, default="original", choices=["original", "enhanced"])
    parser.add_argument("--prompt_set", type=str, default="")
    return parser.parse_args()


def get_task_list(planning_bundle: dict) -> list:
    planning_outputs = planning_bundle.get("planning_outputs", {})
    task_list = planning_outputs.get("task_list", [])
    if task_list:
        return task_list

    logic_design = planning_outputs.get("logic_design", {})
    for key in ("Task list", "task_list", "task list"):
        if isinstance(logic_design, dict) and isinstance(logic_design.get(key), list):
            return logic_design[key]
    return []


def get_code_context(done_file_dict: dict, done_file_lst: list) -> str:
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"):
            continue
        code_files += f"```python\n{done_file_dict[done_file]}\n```\n\n"
    return code_files


def main() -> None:
    args = parse_args()
    prompt_set = args.prompt_set or ("enhanced" if args.pipeline_mode == "enhanced" else "original")

    paper_content = load_paper_content(
        args.paper_format,
        args.pdf_json_path,
        args.pdf_latex_path,
    )
    planning_bundle = load_planning_bundle(args.output_dir, prefer_refined=args.pipeline_mode == "enhanced")
    planning_outputs = planning_bundle.get("planning_outputs", {})
    task_list = get_task_list(planning_bundle)
    if not task_list:
        raise RuntimeError("'Task list' does not exist. Please re-generate the planning.")

    config_yaml = read_text_file(
        os.path.join(args.output_dir, "planning_config.yaml"),
        planning_outputs.get("config_yaml", ""),
    )
    assumptions_text = assumptions_to_text(planning_bundle.get("assumptions", []))
    structured_paper = load_structured_paper(args.output_dir) if args.pipeline_mode == "enhanced" else {}

    detailed_logic_analysis_dict = {}
    artifact_output_dir = os.path.join(args.output_dir, "analyzing_artifacts")
    for todo_file_name in task_list:
        if todo_file_name == "config.yaml":
            continue
        save_todo_file_name = todo_file_name.replace("/", "_")
        detailed_logic_analysis_dict[todo_file_name] = read_text_file(
            os.path.join(artifact_output_dir, f"{save_todo_file_name}_simple_analysis.txt"),
            "",
        )

    system_prompt = load_prompt(prompt_set, "coding_system", "")
    user_prompt = load_prompt(prompt_set, "coding_user", "")
    stage_runner = StageLLM(
        model_name=args.model_name,
        tp_size=args.tp_size,
        temperature=args.temperature,
        max_model_len=args.max_model_len,
    )

    coding_artifact_output_dir = os.path.join(args.output_dir, "coding_artifacts")
    os.makedirs(coding_artifact_output_dir, exist_ok=True)

    done_file_lst = ["config.yaml"]
    done_file_dict = {}

    for todo_file_name in task_list:
        print(f"[CODING] {todo_file_name}")
        if todo_file_name == "config.yaml":
            continue

        trajectories = [
            {
                "role": "system",
                "content": render_prompt(
                    system_prompt,
                    {
                        "PAPER_FORMAT": args.paper_format,
                    },
                ),
            },
            {
                "role": "user",
                "content": render_prompt(
                    user_prompt,
                    {
                        "PAPER": paper_content,
                        "OVERVIEW_PLAN": planning_outputs.get("overview_plan", ""),
                        "ARCHITECTURE_DESIGN": planning_outputs.get("architecture_design_text", ""),
                        "LOGIC_DESIGN": planning_outputs.get("logic_design_text", ""),
                        "CONFIG_YAML": config_yaml,
                        "ASSUMPTIONS": assumptions_text,
                        "CODE_FILES": get_code_context(done_file_dict, done_file_lst),
                        "DONE_FILE_LIST": str(done_file_lst),
                        "TODO_FILE_NAME": todo_file_name,
                        "DETAILED_LOGIC_ANALYSIS": detailed_logic_analysis_dict.get(todo_file_name, ""),
                        "STRUCTURED_PAPER": structured_paper,
                        "ACTIVE_PLANNING_BUNDLE": planning_bundle,
                    },
                ),
            },
        ]

        result = stage_runner.run(trajectories)
        print_response(result["raw"], is_llm=result["is_llm"])

        os.makedirs(args.output_repo_dir, exist_ok=True)
        save_todo_file_name = todo_file_name.replace("/", "_")
        write_text_file(
            os.path.join(coding_artifact_output_dir, f"{save_todo_file_name}_coding.txt"),
            result["text"],
        )

        try:
            code = extract_code_from_content(result["text"])
        except Exception:
            code = extract_code_from_content2(result["text"])
        if not code:
            code = result["text"]

        done_file_dict[todo_file_name] = code
        done_file_lst.append(todo_file_name)
        todo_file_dir = os.path.join(args.output_repo_dir, *todo_file_name.split("/")[:-1])
        if todo_file_dir and not os.path.exists(todo_file_dir):
            os.makedirs(todo_file_dir, exist_ok=True)
        write_text_file(os.path.join(args.output_repo_dir, todo_file_name), code)


if __name__ == "__main__":
    main()
