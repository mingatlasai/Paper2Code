from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from typing import Dict, List

from pipeline_utils import write_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal execution tests on a generated repo.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_repo_dir", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def compile_python_files(repo_dir: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for root, _, files in os.walk(repo_dir):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            relpath = os.path.relpath(path, repo_dir)
            try:
                py_compile.compile(path, doraise=True)
                results.append({"file": relpath, "status": "passed"})
            except py_compile.PyCompileError as exc:
                results.append({"file": relpath, "status": "failed", "error": str(exc)})
    return results


def run_command(command: List[str], cwd: str, timeout: int) -> Dict[str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(command),
            "returncode": str(completed.returncode),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "status": "passed" if completed.returncode == 0 else "failed",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "command": " ".join(command),
            "returncode": "-1",
            "stdout": "",
            "stderr": str(exc),
            "status": "failed",
        }


def main() -> int:
    args = parse_args()
    repo_dir = os.path.abspath(args.output_repo_dir)

    compile_results = compile_python_files(repo_dir)
    commands: List[Dict[str, str]] = []

    reproduce_path = os.path.join(repo_dir, "reproduce.sh")
    if os.path.exists(reproduce_path):
        commands.append(run_command(["bash", "-n", "reproduce.sh"], repo_dir, args.timeout))

    for entrypoint in ("main.py", "app.py"):
        entrypoint_path = os.path.join(repo_dir, entrypoint)
        if not os.path.exists(entrypoint_path):
            continue
        commands.append(run_command([sys.executable, entrypoint, "--help"], repo_dir, args.timeout))
        break

    failures = [
        result for result in compile_results if result["status"] != "passed"
    ] + [result for result in commands if result["status"] != "passed"]

    report = {
        "status": "passed" if not failures else "failed",
        "compile_results": compile_results,
        "command_results": commands,
        "failure_count": len(failures),
        "runtime_errors": failures,
    }
    write_json_file(os.path.join(args.output_dir, "execution_test_report.json"), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
