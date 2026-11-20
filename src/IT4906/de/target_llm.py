"""Wrapper target LLM — `google/medgemma-1.5-4b-it` (multimodal).

Theo official EvoPrompt defaults (file TTTH): batch_size 16, max_new_tokens 128,
seed 5 (seed toàn cục set ở run_de.py). Cache generation theo (fingerprint cấu hình
generation, prompt_text, sample_idx) => prompt cũ re-eval không tốn GPU; đổi dtype/
do_sample/max_new_tokens sẽ tự tách cache riêng — KHÔNG trả lại output cũ (đã từng
bị cache poison: output rỗng do fp16 bị dùng lại cả khi đã đổi sang bf16).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .cuda_compat import apply_turing_sdpa_patch
from .data import SYSTEM_PROMPT, Sample

GenCacheKey = tuple[str, str, int]  # (gen_config_fingerprint, rendered_prompt, sample.idx)


def _key(gen_fp: str, prompt_text: str, sample_idx: int) -> str:
    raw = f"{gen_fp}||{prompt_text}||{sample_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TargetLLM:
    def __init__(self, cfg: dict, cache_dir: str):
        self.model_id: str = cfg["model_id"]
        self.batch_size: int = int(cfg.get("batch_size", 16))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 128))
        self.do_sample: bool = bool(cfg.get("do_sample", False))  # greedy mặc định cho eval
        self.dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        # Fingerprint cấu hình generation — đưa vào cache key: đổi dtype/do_sample/
        # max_new_tokens → key khác → không bao giờ dùng lại output của cấu hình cũ.
        self._gen_fp = hashlib.sha1(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "dtype": cfg.get("dtype", "bfloat16"),
                    "do_sample": self.do_sample,
                    "max_new_tokens": self.max_new_tokens,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:10]
        self.cache_path = Path(cache_dir) / f"gen_cache_{self._gen_fp}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            with self.cache_path.open("r+b") as f:
                while line := f.readline():
                    if not line.endswith(b"\n"):
                        # Tiến trình trước có thể bị ngắt khi đang ghi dòng cuối.
                        f.truncate(f.tell() - len(line))
                        break
                    if line.strip():
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
        self._flush_count = 0
        self._load()

    def _load(self) -> None:
        apply_turing_sdpa_patch()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Batched generate cho decoder-only cần LEFT padding, không phải right mặc định
        self.processor.tokenizer.padding_side = "left"
        # Ghim NGUYÊN model lên GPU 0 — KHÔNG dùng "auto": auto chia đều 2 GPU,
        # khiến GPU 0 hết chỗ cho KV cache + activation khi evolution 27B nạp sau
        # và chiếm nốt phần dư (đã OOM thật 2026-11-11).
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=device_map
        )
        self.model.eval()

    def generate(self, prompt_text: str, sample: Sample) -> str:
        """Sinh Findings cho 1 sample. Có cache."""
        k = _key(self._gen_fp, prompt_text, sample.idx)
        if k in self.cache:
            return self.cache[k]
        out = self._generate_batch([(prompt_text, sample)])[0]
        return out

    def generate_batch(self, pairs: list[tuple[str, Sample]]) -> list[str]:
        """Sinh Findings cho nhiều (prompt, sample); bỏ qua cặp đã cache."""
        todo = [(p, s) for p, s in pairs if _key(self._gen_fp, p, s.idx) not in self.cache]
        results: dict[str, str] = {}
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i : i + self.batch_size]
            outs = self._generate_batch(chunk)
            for (p, s), o in zip(chunk, outs):
                results[_key(self._gen_fp, p, s.idx)] = o
            self._append_cache(
                [(_key(self._gen_fp, p, s.idx), o) for (p, s), o in zip(chunk, outs)]
            )
        # gộp kết quả theo thứ tự yêu cầu
        return [self.cache[_key(self._gen_fp, p, s.idx)] for p, s in pairs]

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
