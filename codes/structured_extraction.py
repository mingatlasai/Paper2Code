from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from pipeline_utils import (
    extract_tagged_block,
    get_repo_root,
    load_paper_content,
    load_prompt,
    render_prompt,
    save_structured_artifact,
)
from stage_llm import StageLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured paper extraction.")
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


def fallback_payload(error_message: str) -> Dict[str, Any]:
    return {
        "status": "fallback",
        "error": error_message,
        "model_components": [],
        "training_objective": [],
        "datasets": [],
        "metrics": [],
        "training_protocol": [],
        "referenced_algorithms": [],
        "assumptions": [],
        "unclear_items": ["Fallback to original paper text."],
    }


def main() -> int:
    args = parse_args()

    try:
        paper_content = load_paper_content(
            args.paper_format,
            args.pdf_json_path,
            args.pdf_latex_path,
        )
    except Exception as exc:
        payload = fallback_payload(str(exc))
        save_structured_artifact(
            args.output_dir,
            "structured_paper",
            "",
            payload,
        )
        return 0

    system_prompt = load_prompt(
        args.prompt_set,
        "structured_extraction_system",
        "You are a scientific paper extraction specialist.",
    )
    user_prompt = load_prompt(
        args.prompt_set,
        "structured_extraction_user",
        "## Paper\n{{PAPER}}",
    )
    messages = [
        {
            "role": "system",
            "content": render_prompt(
                system_prompt,
                {"PAPER_FORMAT": args.paper_format},
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                user_prompt,
                {"PAPER": paper_content},
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
        block = extract_tagged_block(raw_text, "STRUCTURED_PAPER")
        payload = json.loads(block) if block else fallback_payload("Missing STRUCTURED_PAPER block.")
        if not isinstance(payload, dict):
            payload = fallback_payload("Structured extraction payload was not a JSON object.")
        payload.setdefault("status", "ok")
        payload["paper_name"] = args.paper_name
    except Exception as exc:  # pragma: no cover - depends on provider runtime
        raw_text = ""
        payload = fallback_payload(str(exc))
        payload["paper_name"] = args.paper_name

    save_structured_artifact(
        args.output_dir,
        "structured_paper",
        raw_text,
        payload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
