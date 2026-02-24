from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class _Message:
    role: str
    content: str


@dataclass
class _Choice:
    message: _Message


class CompletionCompat:
    """OpenAI-compatible shim used by Paper2Code scripts.

    Provides:
    - completion.choices[0].message.content
    - completion.model_dump_json()
    """

    def __init__(self, content: str, model: str):
        self.choices = [_Choice(message=_Message(role="assistant", content=content))]
        self._payload = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "model": model,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self._payload)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _run_cmd(cmd: List[str], input_text: str = "") -> str:
    p = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"LLM command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def chat(messages: List[Dict[str, str]], model: str) -> CompletionCompat:
    provider = os.getenv("P2C_PROVIDER", "codex_local").strip().lower()
    prompt = _messages_to_prompt(messages)

    if provider in {"codex", "codex_local"}:
        cmd = shlex.split(os.getenv("P2C_CODEX_CMD", "codex exec"))
        # Send prompt via stdin to avoid shell/OS argument-length limits.
        text = _run_cmd(cmd, input_text=prompt)
    elif provider in {"claude", "claude_code"}:
        cmd = shlex.split(os.getenv("P2C_CLAUDE_CMD", "claude -p"))
        # Try stdin first (handles long prompts). If CLI requires argv prompt, fallback.
        try:
            text = _run_cmd(cmd, input_text=prompt)
        except RuntimeError:
            text = _run_cmd(cmd + [prompt])
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return CompletionCompat(content=text, model=model)
