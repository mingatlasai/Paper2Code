from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from utils import content_to_json, extract_planning


def get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def read_text_file(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def write_json_file(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json_file(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_paper_content(
    paper_format: str,
    pdf_json_path: Optional[str],
    pdf_latex_path: Optional[str],
) -> Any:
    if paper_format == "JSON":
        with open(pdf_json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if paper_format == "LaTeX":
        with open(pdf_latex_path, "r", encoding="utf-8") as handle:
            return handle.read()
    raise ValueError("Invalid paper format. Please select either 'JSON' or 'LaTeX'.")


def prompt_path(prompt_set: str, prompt_name: str) -> str:
    return os.path.join(get_repo_root(), "prompts", prompt_set, f"{prompt_name}.txt")


def load_prompt(prompt_set: str, prompt_name: str, fallback_text: str) -> str:
    path = prompt_path(prompt_set, prompt_name)
    if os.path.exists(path):
        return read_text_file(path, fallback_text)
    return fallback_text


def render_prompt(template: str, replacements: Dict[str, Any]) -> str:
    rendered = template
    for key, value in replacements.items():
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def extract_tagged_block(text: str, tag: str) -> str:
    pattern = rf"\[{re.escape(tag)}\]\s*(.*?)\s*\[/{re.escape(tag)}\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_yaml_code_block(text: str) -> str:
    patterns = [
        r"```yaml\n(.*?)\n```",
        r"```yaml\\n(.*?)\\n```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def normalize_text_lines(raw_text: str) -> List[str]:
    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def extract_assumptions(*texts: str) -> List[str]:
    assumptions: List[str] = []
    seen = set()
    pattern = re.compile(r"(ASSUMPTION\s+[A-Z]?\d+\s*:\s*.+)", re.IGNORECASE)
    for text in texts:
        if not text:
            continue
        for line in normalize_text_lines(text):
            match = pattern.search(line)
            if match:
                assumption = match.group(1).strip()
                normalized = assumption.upper()
                if normalized not in seen:
                    seen.add(normalized)
                    assumptions.append(assumption)
    return assumptions


def extract_unclear_items(*texts: str) -> List[str]:
    unclear_items: List[str] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for line in normalize_text_lines(text):
            lowered = line.lower()
            if "unclear" not in lowered and "missing" not in lowered:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            unclear_items.append(line)
    return unclear_items


def assumptions_to_text(assumptions: List[str]) -> str:
    if not assumptions:
        return "None."
    return "\n".join(f"- {assumption}" for assumption in assumptions)


def default_config_payload(assumptions: List[str]) -> Dict[str, Any]:
    return {
        "dataset": {
            "name": None,
            "path": None,
            "splits": None,
        },
        "model": {
            "name": None,
            "components": [],
        },
        "training": {
            "optimizer": None,
            "learning_rate": None,
            "batch_size": None,
            "epochs": None,
            "objective": None,
            "assumptions": assumptions,
        },
        "evaluation": {
            "metrics": [],
            "protocol": None,
        },
        "logging": {
            "level": "INFO",
            "save_dir": "outputs/logs",
        },
        "checkpoint": {
            "dir": "outputs/checkpoints",
            "save_best_only": True,
        },
        "seeds": {
            "global_seed": 42,
        },
        "assumptions": assumptions,
    }


def ensure_config_sections(config_yaml: str, assumptions: List[str]) -> str:
    default_payload = default_config_payload(assumptions)
    if yaml is None:
        return _ensure_config_sections_text(config_yaml, assumptions)

    try:
        parsed = yaml.safe_load(config_yaml) if config_yaml.strip() else {}
    except Exception:
        return _ensure_config_sections_text(config_yaml, assumptions)

    if not isinstance(parsed, dict):
        parsed = {}

    for key, default_value in default_payload.items():
        if key not in parsed or parsed[key] is None:
            parsed[key] = default_value

    training = parsed.get("training")
    if isinstance(training, dict):
        training.setdefault("assumptions", assumptions)
    parsed.setdefault("assumptions", assumptions)

    return yaml.safe_dump(parsed, sort_keys=False)


def _ensure_config_sections_text(config_yaml: str, assumptions: List[str]) -> str:
    required_sections = {
        "dataset": "dataset:\n  name: null\n  path: null\n  splits: null",
        "model": "model:\n  name: null\n  components: []",
        "training": "training:\n  optimizer: null\n  learning_rate: null\n  batch_size: null\n  epochs: null\n  objective: null",
        "evaluation": "evaluation:\n  metrics: []\n  protocol: null",
        "logging": "logging:\n  level: INFO\n  save_dir: outputs/logs",
        "checkpoint": "checkpoint:\n  dir: outputs/checkpoints\n  save_best_only: true",
        "seeds": "seeds:\n  global_seed: 42",
    }
    rendered = config_yaml.strip()
    for key, block in required_sections.items():
        if re.search(rf"^{key}\s*:", rendered, re.MULTILINE):
            continue
        rendered = f"{rendered}\n\n{block}".strip()

    if assumptions and "assumptions:" not in rendered:
        rendered = f"{rendered}\n\nassumptions:\n" + "\n".join(
            f"  - {item}" for item in assumptions
        )
    return rendered.strip() + "\n"


def build_planning_bundle(
    paper_name: str,
    paper_format: str,
    output_dir: str,
    responses: List[Dict[str, Any]],
    trajectories: List[Dict[str, str]],
    pipeline_mode: str,
    prompt_set: str,
) -> Dict[str, Any]:
    response_texts: List[str] = []
    for response in responses:
        if "choices" in response:
            response_texts.append(response["choices"][0]["message"]["content"])
        else:
            response_texts.append(response.get("text", ""))

    overview_plan = response_texts[0] if len(response_texts) > 0 else ""
    architecture_design_text = response_texts[1] if len(response_texts) > 1 else ""
    logic_design_text = response_texts[2] if len(response_texts) > 2 else ""
    config_text = response_texts[3] if len(response_texts) > 3 else ""

    architecture_design = content_to_json(architecture_design_text) if architecture_design_text else {}
    logic_design = content_to_json(logic_design_text) if logic_design_text else {}
    assumptions = extract_assumptions(
        overview_plan,
        architecture_design_text,
        logic_design_text,
        config_text,
    )
    unclear_items = extract_unclear_items(
        overview_plan,
        architecture_design_text,
        logic_design_text,
        config_text,
    )

    bundle = {
        "paper_name": paper_name,
        "paper_format": paper_format,
        "pipeline_mode": pipeline_mode,
        "prompt_set": prompt_set,
        "planning_outputs": {
            "overview_plan": overview_plan,
            "architecture_design_text": architecture_design_text,
            "architecture_design": architecture_design,
            "logic_design_text": logic_design_text,
            "logic_design": logic_design,
            "config_generation_text": config_text,
            "config_yaml": extract_yaml_code_block(config_text),
            "task_list": _extract_task_list(logic_design),
        },
        "assumptions": assumptions,
        "unclear_items": unclear_items,
        "artifacts": {
            "planning_response_path": os.path.join(output_dir, "planning_response.json"),
            "planning_trajectories_path": os.path.join(output_dir, "planning_trajectories.json"),
        },
        "trajectories": trajectories,
    }
    return bundle


def _extract_task_list(logic_design: Dict[str, Any]) -> List[str]:
    if not isinstance(logic_design, dict):
        return []
    for key in ("Task list", "task_list", "task list"):
        value = logic_design.get(key)
        if isinstance(value, list):
            return value
    return []


def save_planning_bundle(output_dir: str, bundle: Dict[str, Any]) -> None:
    write_json_file(os.path.join(output_dir, "planning_bundle.json"), bundle)
    task_list = bundle.get("planning_outputs", {}).get("logic_design", {})
    if isinstance(task_list, dict) and task_list:
        write_json_file(os.path.join(output_dir, "task_list.json"), task_list)


def derive_bundle_from_trajectories(output_dir: str) -> Dict[str, Any]:
    planning_traj_path = os.path.join(output_dir, "planning_trajectories.json")
    planning_response_path = os.path.join(output_dir, "planning_response.json")
    context_list = extract_planning(planning_traj_path)
    bundle = {
        "paper_name": os.path.basename(output_dir),
        "pipeline_mode": "original",
        "prompt_set": "original",
        "planning_outputs": {
            "overview_plan": context_list[0] if len(context_list) > 0 else "",
            "architecture_design_text": context_list[1] if len(context_list) > 1 else "",
            "architecture_design": content_to_json(context_list[1]) if len(context_list) > 1 else {},
            "logic_design_text": context_list[2] if len(context_list) > 2 else "",
            "logic_design": content_to_json(context_list[2]) if len(context_list) > 2 else {},
            "config_generation_text": "",
            "config_yaml": read_text_file(os.path.join(output_dir, "planning_config.yaml"), ""),
            "task_list": content_to_json(context_list[2]).get("Task list", []) if len(context_list) > 2 else [],
        },
        "assumptions": extract_assumptions(*context_list),
        "unclear_items": extract_unclear_items(*context_list),
        "artifacts": {
            "planning_response_path": planning_response_path,
            "planning_trajectories_path": planning_traj_path,
        },
    }
    return bundle


def load_planning_bundle(output_dir: str, prefer_refined: bool = True) -> Dict[str, Any]:
    bundle_paths = []
    if prefer_refined:
        bundle_paths.append(os.path.join(output_dir, "refined_planning_bundle.json"))
    bundle_paths.append(os.path.join(output_dir, "planning_bundle.json"))

    for path in bundle_paths:
        if os.path.exists(path):
            return load_json_file(path)

    return derive_bundle_from_trajectories(output_dir)


def save_structured_artifact(
    output_dir: str,
    file_stem: str,
    raw_text: str,
    payload: Dict[str, Any],
) -> None:
    artifact_dir = os.path.join(output_dir, "enhanced_artifacts")
    write_text_file(os.path.join(artifact_dir, f"{file_stem}.txt"), raw_text)
    write_json_file(os.path.join(output_dir, f"{file_stem}.json"), payload)


def load_structured_paper(output_dir: str) -> Dict[str, Any]:
    return load_json_file(os.path.join(output_dir, "structured_paper.json"), {})
