"""Wrapper target LLM — `google/medgemma-1.5-4b-it` (multimodal).

Theo official EvoPrompt defaults (file TTTH): batch_size 16, max_new_tokens 128,
seed 5 (seed toàn cục set ở run_de.py). Cache generation theo (prompt_text, sample_idx)
=> prompt cũ re-eval không tốn GPU; đổi fitness model sau này chỉ cần re-score.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .data import SYSTEM_PROMPT, Sample

GenCacheKey = tuple[str, int]  # (rendered_prompt, sample.idx)


def _key(prompt_text: str, sample_idx: int) -> str:
    raw = f"{prompt_text}||{sample_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TargetLLM:
    def __init__(self, cfg: dict, cache_dir: str):
        self.model_id: str = cfg["model_id"]
        self.batch_size: int = int(cfg.get("batch_size", 16))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 128))
        self.do_sample: bool = bool(cfg.get("do_sample", False))  # greedy mặc định cho eval
        self.dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        self.cache_path = Path(cache_dir) / "gen_cache.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
        self._flush_count = 0
        self._load()

    def _load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=self.dtype, device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt_text: str, sample: Sample) -> str:
        """Sinh Findings cho 1 sample. Có cache."""
        k = _key(prompt_text, sample.idx)
        if k in self.cache:
            return self.cache[k]
        out = self._generate_batch([(prompt_text, sample)])[0]
        return out

    def generate_batch(self, pairs: list[tuple[str, Sample]]) -> list[str]:
        """Sinh Findings cho nhiều (prompt, sample); bỏ qua cặp đã cache."""
        todo = [(p, s) for p, s in pairs if _key(p, s.idx) not in self.cache]
        results: dict[str, str] = {}
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i : i + self.batch_size]
            outs = self._generate_batch(chunk)
            for (p, s), o in zip(chunk, outs):
                results[_key(p, s.idx)] = o
            self._append_cache(
                [(_key(p, s.idx), o) for (p, s), o in zip(chunk, outs)]
            )
        # gộp kết quả theo thứ tự yêu cầu
        return [self.cache[_key(p, s.idx)] for p, s in pairs]

    @torch.inference_mode()
    def _generate_batch(self, pairs: list[tuple[str, Sample]]) -> list[str]:
        conversations = []
        for prompt_text, sample in pairs:
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "image": sample.image},
                        ],
                    },
                ]
            )
        prompts_text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images = [[s.image] for _, s in pairs]
        inputs = self.processor(
            text=prompts_text, images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs: dict = dict(max_new_tokens=self.max_new_tokens)
        if self.do_sample:
            gen_kwargs.update(do_sample=True, top_k=0, top_p=0.95)
        output = self.model.generate(**inputs, **gen_kwargs)
        trimmed = output[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def _append_cache(self, records: list[tuple[str, str]]) -> None:
        for k, v in records:
            self.cache[k] = v
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for k, v in records:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
