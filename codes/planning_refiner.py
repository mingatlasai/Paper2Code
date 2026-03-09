from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from pipeline_utils import (
    ensure_config_sections,
    extract_assumptions,
    extract_tagged_block,
    extract_unclear_items,
    load_planning_bundle,
    load_prompt,
    load_json_file,
    render_prompt,
    write_json_file,
    write_text_file,
)
from stage_llm import StageLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine the planning artifact.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_set", type=str, default="enhanced")
    parser.add_argument("--gpt_version", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=128000)
    return parser.parse_args()


def fallback_bundle(planning_bundle: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    refined = dict(planning_bundle)
    refined["refinement_status"] = "fallback"
    refined["refinement_error"] = error_message
    refined["refinement_changes"] = []
    return refined


def main() -> int:
    args = parse_args()
    planning_bundle = load_planning_bundle(args.output_dir, prefer_refined=False)
    planning_review = load_json_file(os.path.join(args.output_dir, "planning_review.json"))

    review_status = planning_review.get("status")
    if not planning_review or review_status == "skipped":
        refined_bundle = fallback_bundle(planning_bundle, "Planning review unavailable; using original planning.")
        write_json_file(os.path.join(args.output_dir, "refined_planning_bundle.json"), refined_bundle)
        return 0

    system_prompt = load_prompt(
        args.prompt_set,
        "planning_refiner_system",
        "You are a planning refinement specialist.",
    )
    user_prompt = load_prompt(
        args.prompt_set,
        "planning_refiner_user",
        "## Original Planning Bundle\n{{PLANNING_BUNDLE}}",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": render_prompt(
                user_prompt,
                {
                    "PLANNING_BUNDLE": planning_bundle,
                    "PLANNING_REVIEW": planning_review,
                },
            ),
        },
    ]

    try:
        runner = StageLLM(
            gpt_version=args.gpt_version or None,
            model_name=args.model_name or None,
            tp_size=args.tp_size,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
        )
        result = runner.run(messages)
        raw_text = result["text"]
        block = extract_tagged_block(raw_text, "REFINED_PLANNING")
        payload = json.loads(block) if block else {}
        if not isinstance(payload, dict):
            payload = {}

        refined_bundle = dict(planning_bundle)
        planning_outputs = refined_bundle.setdefault("planning_outputs", {})

        if payload.get("overview_plan"):
            planning_outputs["overview_plan"] = payload["overview_plan"]
        if isinstance(payload.get("architecture_design"), dict) and payload["architecture_design"]:
            planning_outputs["architecture_design"] = payload["architecture_design"]
            planning_outputs["architecture_design_text"] = json.dumps(
                payload["architecture_design"],
                indent=2,
            )
        if isinstance(payload.get("logic_design"), dict) and payload["logic_design"]:
            planning_outputs["logic_design"] = payload["logic_design"]
            planning_outputs["logic_design_text"] = json.dumps(
                payload["logic_design"],
                indent=2,
            )
            task_list = payload["logic_design"].get("Task list")
            if isinstance(task_list, list):
                planning_outputs["task_list"] = task_list

        config_yaml = payload.get("config_yaml") or planning_outputs.get("config_yaml", "")
        assumptions = extract_assumptions(
            json.dumps(payload, indent=2),
            json.dumps(planning_review, indent=2),
            json.dumps(planning_bundle, indent=2),
        )
        planning_outputs["config_yaml"] = ensure_config_sections(config_yaml, assumptions)
        refined_bundle["assumptions"] = assumptions
        refined_bundle["unclear_items"] = extract_unclear_items(
            json.dumps(payload, indent=2),
            json.dumps(planning_review, indent=2),
        )
        refined_bundle["refinement_status"] = "ok"
        refined_bundle["refinement_changes"] = payload.get("changes_made", [])
        refined_bundle["planning_review"] = planning_review
        write_text_file(
            os.path.join(args.output_dir, "enhanced_artifacts", "refined_planning.txt"),
            raw_text,
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        refined_bundle = fallback_bundle(planning_bundle, str(exc))

    write_json_file(os.path.join(args.output_dir, "refined_planning_bundle.json"), refined_bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
