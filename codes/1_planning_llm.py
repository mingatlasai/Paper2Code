from __future__ import annotations

import argparse
import os

from pipeline_utils import (
    build_planning_bundle,
    load_paper_content,
    load_prompt,
    load_structured_paper,
    render_prompt,
    save_planning_bundle,
    write_json_file,
)
from stage_llm import StageLLM
from utils import print_response


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
    parser.add_argument("--pipeline_mode", type=str, default="original", choices=["original", "enhanced"])
    parser.add_argument("--prompt_set", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_set = args.prompt_set or ("enhanced" if args.pipeline_mode == "enhanced" else "original")

    paper_content = load_paper_content(
        args.paper_format,
        args.pdf_json_path,
        args.pdf_latex_path,
    )
    structured_paper = load_structured_paper(args.output_dir) if args.pipeline_mode == "enhanced" else {}

    stage_runner = StageLLM(
        model_name=args.model_name,
        tp_size=args.tp_size,
        temperature=args.temperature,
        max_model_len=args.max_model_len,
    )

    stage_specs = [
        {
            "name": "[Planning] Overall plan",
            "system": "planning_overall_system",
            "user": "planning_overall_user",
            "payload": {
                "PAPER": paper_content,
                "PAPER_FORMAT": args.paper_format,
                "STRUCTURED_PAPER": structured_paper,
            },
        },
        {
            "name": "[Planning] Architecture design",
            "user": "planning_architecture_user",
            "payload": {
                "STRUCTURED_PAPER": structured_paper,
            },
        },
        {
            "name": "[Planning] Logic design",
            "user": "planning_logic_user",
            "payload": {
                "STRUCTURED_PAPER": structured_paper,
            },
        },
        {
            "name": "[Planning] Configuration file generation",
            "user": "planning_config_user",
            "payload": {
                "STRUCTURED_PAPER": structured_paper,
            },
        },
    ]

    responses = []
    trajectories = []

    for stage in stage_specs:
        print(stage["name"])
        if stage.get("system"):
            system_prompt = load_prompt(prompt_set, stage["system"], "")
            trajectories.append(
                {
                    "role": "system",
                    "content": render_prompt(system_prompt, stage["payload"]),
                }
            )

        user_prompt = load_prompt(prompt_set, stage["user"], "")
        trajectories.append(
            {
                "role": "user",
                "content": render_prompt(user_prompt, stage["payload"]),
            }
        )

        result = stage_runner.run(trajectories)
        print_response(result["raw"], is_llm=result["is_llm"])
        responses.append(result["raw"])
        trajectories.append({"role": "assistant", "content": result["text"]})

    os.makedirs(args.output_dir, exist_ok=True)
    write_json_file(os.path.join(args.output_dir, "planning_response.json"), responses)
    write_json_file(os.path.join(args.output_dir, "planning_trajectories.json"), trajectories)

    planning_bundle = build_planning_bundle(
        paper_name=args.paper_name,
        paper_format=args.paper_format,
        output_dir=args.output_dir,
        responses=responses,
        trajectories=trajectories,
        pipeline_mode=args.pipeline_mode,
        prompt_set=prompt_set,
    )
    if structured_paper:
        planning_bundle["structured_paper"] = structured_paper
    save_planning_bundle(args.output_dir, planning_bundle)


if __name__ == "__main__":
    main()
