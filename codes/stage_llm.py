from __future__ import annotations

import json
from typing import Dict, List, Optional

from llm_backend import chat


class StageLLM:
    """Small wrapper that normalizes CLI-agent and vLLM responses."""

    def __init__(
        self,
        gpt_version: Optional[str] = None,
        model_name: Optional[str] = None,
        tp_size: int = 2,
        temperature: float = 1.0,
        max_model_len: int = 128000,
    ) -> None:
        self.gpt_version = gpt_version
        self.model_name = model_name
        self.tp_size = tp_size
        self.temperature = temperature
        self.max_model_len = max_model_len
        self._tokenizer = None
        self._llm = None
        self._sampling_params = None

        if not self.gpt_version and not self.model_name:
            raise ValueError("Either gpt_version or model_name must be provided.")

        if self.model_name:
            self._init_vllm()

    def _init_vllm(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if "Qwen" in self.model_name:
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tp_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                enforce_eager=True,
                rope_scaling={
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                },
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=131072,
            )
            return

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=128000,
            stop_token_ids=[self._tokenizer.eos_token_id],
        )

    def run(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        if self.gpt_version:
            completion = chat(messages, self.gpt_version)
            completion_json = json.loads(completion.model_dump_json())
            return {
                "text": completion.choices[0].message.content,
                "raw": completion_json,
                "is_llm": False,
            }

        prompt_token_ids = [
            self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        ]
        outputs = self._llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=self._sampling_params,
        )
        text = outputs[0].outputs[0].text
        return {
            "text": text,
            "raw": {"text": text},
            "is_llm": True,
        }
