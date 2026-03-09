from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from pipeline_utils import (
    extract_assumptions,
    extract_tagged_block,
    load_paper_content,
    load_planning_bundle,
    load_prompt,
    load_structured_paper,
    render_prompt,
    save_structured_artifact,
)
from stage_llm import StageLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the planning artifact.")
    parser.add_argument("--paper_name", type=str, required=True)
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str)
    parser.add_argument("--pdf_latex_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_set", type=str, default="enhanced")
    parser.add_argument("--gpt_version", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=128000)
    return parser.parse_args()


def fallback_review(error_message: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "error": error_message,
        "missing_modules": [],
        "dependency_issues": [],
        "missing_training_steps": [],
        "evaluation_gaps": [],
        "api_inconsistencies": [],
        "assumptions": [],
    }


def main() -> int:
    args = parse_args()

    try:
        paper_content = load_paper_content(
            args.paper_format,
            args.pdf_json_path,
            args.pdf_latex_path,
        )
        planning_bundle = load_planning_bundle(args.output_dir, prefer_refined=False)
        structured_paper = load_structured_paper(args.output_dir)
    except Exception as exc:
        save_structured_artifact(args.output_dir, "planning_review", "", fallback_review(str(exc)))
        return 0

    system_prompt = load_prompt(
        args.prompt_set,
        "planning_verifier_system",
        "You are a reproduction-plan auditor.",
    )
    user_prompt = load_prompt(
        args.prompt_set,
        "planning_verifier_user",
        "## Planning Bundle\n{{PLANNING_BUNDLE}}",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": render_prompt(
                user_prompt,
                {
                    "PAPER": paper_content,
                    "STRUCTURED_PAPER": structured_paper,
                    "PLANNING_BUNDLE": planning_bundle,
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
        block = extract_tagged_block(raw_text, "PLANNING_REVIEW")
        payload = json.loads(block) if block else fallback_review("Missing PLANNING_REVIEW block.")
        if not isinstance(payload, dict):
            payload = fallback_review("Planning review payload was not a JSON object.")
        payload.setdefault("status", "ok")
        payload["assumptions"] = extract_assumptions(
            json.dumps(payload, indent=2),
            json.dumps(planning_bundle, indent=2),
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        raw_text = ""
        payload = fallback_review(str(exc))

    save_structured_artifact(args.output_dir, "planning_review", raw_text, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
