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
    write_json_file,
    write_text_file,
)
from stage_llm import StageLLM
from utils import print_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str)
    parser.add_argument("--gpt_version", type=str, default="o3-mini")
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str)
    parser.add_argument("--pdf_latex_path", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--pipeline_mode", type=str, default="original", choices=["original", "enhanced"])
    parser.add_argument("--prompt_set", type=str, default="")
    return parser.parse_args()


def get_logic_analysis_map(logic_design: dict) -> dict:
    analysis_entries = None
    for key in ("Logic Analysis", "logic_analysis", "logic analysis"):
        if key in logic_design:
            analysis_entries = logic_design[key]
            break

    logic_analysis_dict = {}
    if isinstance(analysis_entries, list):
        for entry in analysis_entries:
            if isinstance(entry, list) and len(entry) >= 2:
                logic_analysis_dict[entry[0]] = entry[1]
    return logic_analysis_dict


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
    logic_design = planning_outputs.get("logic_design", {})
    logic_analysis_dict = get_logic_analysis_map(logic_design)
    task_list = get_task_list(planning_bundle)

    if not task_list:
        raise RuntimeError("'Task list' does not exist. Please re-generate the planning.")

    config_yaml = read_text_file(
        os.path.join(args.output_dir, "planning_config.yaml"),
        planning_outputs.get("config_yaml", ""),
    )
    assumptions_text = assumptions_to_text(planning_bundle.get("assumptions", []))
    structured_paper = load_structured_paper(args.output_dir) if args.pipeline_mode == "enhanced" else {}

    system_prompt = load_prompt(prompt_set, "analysis_system", "")
    user_prompt = load_prompt(prompt_set, "analysis_user", "")
    stage_runner = StageLLM(gpt_version=args.gpt_version)

    artifact_output_dir = os.path.join(args.output_dir, "analyzing_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)

    for todo_file_name in task_list:
        print(f"[ANALYSIS] {todo_file_name}")
        if todo_file_name == "config.yaml":
            continue

        file_directive = f"Write the logic analysis in '{todo_file_name}'."
        todo_desc = logic_analysis_dict.get(todo_file_name, "").strip()
        if todo_desc:
            file_directive = (
                f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_desc}'."
            )

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
                        "FILE_DIRECTIVE": file_directive,
                        "TODO_FILE_NAME": todo_file_name,
                        "STRUCTURED_PAPER": structured_paper,
                        "ACTIVE_PLANNING_BUNDLE": planning_bundle,
                    },
                ),
            },
        ]

        result = stage_runner.run(trajectories)
        print_response(result["raw"], is_llm=result["is_llm"])

        save_todo_file_name = todo_file_name.replace("/", "_")
        write_text_file(
            os.path.join(artifact_output_dir, f"{save_todo_file_name}_simple_analysis.txt"),
            result["text"],
        )
        write_json_file(
            os.path.join(args.output_dir, f"{save_todo_file_name}_simple_analysis_response.json"),
            [result["raw"]],
        )
        trajectories.append({"role": "assistant", "content": result["text"]})
        write_json_file(
            os.path.join(args.output_dir, f"{save_todo_file_name}_simple_analysis_trajectories.json"),
            trajectories,
        )


if __name__ == "__main__":
    main()
