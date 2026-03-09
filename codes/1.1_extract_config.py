from __future__ import annotations

import argparse
import os
import shutil

from pipeline_utils import (
    ensure_config_sections,
    load_planning_bundle,
    read_text_file,
    write_text_file,
)
from utils import format_json_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planning_bundle = load_planning_bundle(args.output_dir, prefer_refined=True)
    planning_outputs = planning_bundle.get("planning_outputs", {})
    assumptions = planning_bundle.get("assumptions", [])

    config_yaml = planning_outputs.get("config_yaml", "")
    if not config_yaml:
        config_yaml = read_text_file(os.path.join(args.output_dir, "planning_config.yaml"), "")
    config_yaml = ensure_config_sections(config_yaml, assumptions)

    planning_config_path = os.path.join(args.output_dir, "planning_config.yaml")
    write_text_file(planning_config_path, config_yaml)

    artifact_output_dir = os.path.join(args.output_dir, "planning_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)

    overall_plan = planning_outputs.get("overview_plan", "")
    architecture_design = planning_outputs.get("architecture_design", {})
    logic_design = planning_outputs.get("logic_design", {})

    with open(os.path.join(artifact_output_dir, "1.1_overall_plan.txt"), "w", encoding="utf-8") as handle:
        handle.write(overall_plan)
    with open(os.path.join(artifact_output_dir, "1.2_arch_design.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_json_data(architecture_design) if architecture_design else "")
    with open(os.path.join(artifact_output_dir, "1.3_logic_design.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_json_data(logic_design) if logic_design else "")

    shutil.copy(planning_config_path, os.path.join(artifact_output_dir, "1.4_config.yaml"))


if __name__ == "__main__":
    main()
