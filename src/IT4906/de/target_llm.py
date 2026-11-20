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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .cuda_compat import apply_turing_sdpa_patch
from .data import SYSTEM_PROMPT, Sample
from .progress import progress_stage


def target_result_config(cfg: dict) -> dict:
    """Keep GPU count in result identity only when it changes sampling RNG assignment."""
    result = cfg.copy()
    if not result.get("do_sample", False) or int(result.get("num_gpus", 1)) == 1:
        result.pop("num_gpus", None)
    return result


def _key(gen_fp: str, prompt_text: str, sample_idx: int) -> str:
    raw = f"{gen_fp}||{prompt_text}||{sample_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TargetLLM:
    def __init__(self, cfg: dict, cache_dir: str):
        self.model_id: str = cfg["model_id"]
        self.batch_size: int = int(cfg.get("batch_size", 16))
        self.num_gpus = int(cfg.get("num_gpus", 1))
        if self.num_gpus not in (1, 2):
            raise ValueError("target_llm.num_gpus phải là 1 hoặc 2.")
        self._use_cuda = torch.cuda.is_available()
        if self.num_gpus == 2 and (not self._use_cuda or torch.cuda.device_count() < 2):
            raise ValueError("--target-gpus 2 cần hai GPU CUDA khả dụng.")
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 128))
        self.do_sample: bool = bool(cfg.get("do_sample", False))  # greedy mặc định cho eval
        self.dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        # Fingerprint cấu hình generation — đưa vào cache key: đổi dtype/do_sample/
        # max_new_tokens → key khác → không bao giờ dùng lại output của cấu hình cũ.
        self._gen_fp = hashlib.sha1(
            json.dumps(
                target_result_config({
                    "model_id": self.model_id,
                    "dtype": cfg.get("dtype", "bfloat16"),
                    "do_sample": self.do_sample,
                    "max_new_tokens": self.max_new_tokens,
                    "num_gpus": self.num_gpus,
                }),
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
        self._load()

    @property
    def parallel_batch_size(self) -> int:
        return self.batch_size * self.num_gpus

    def _load(self) -> None:
        apply_turing_sdpa_patch()
        self._replicas = []
        for gpu in range(self.num_gpus):
            placement = f"cuda:{gpu}" if self._use_cuda else "auto"
            with progress_stage(f"[target] nạp {self.model_id} trên {placement}"):
                processor = AutoProcessor.from_pretrained(self.model_id)
                processor.tokenizer.padding_side = "left"
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, dtype=self.dtype, device_map={"": gpu} if self._use_cuda else "auto"
                )
                model.eval()
                self._replicas.append((processor, model))

    def generate(self, prompt_text: str, sample: Sample) -> str:
        """Sinh Findings cho 1 sample. Có cache."""
        k = _key(self._gen_fp, prompt_text, sample.idx)
        if k in self.cache:
            return self.cache[k]
        out = self._generate_batch([(prompt_text, sample)])[0]
        return out

    def generate_batch(self, pairs: list[tuple[str, Sample]], *, progress: str | None = None) -> list[str]:
        """Sinh Findings cho nhiều (prompt, sample); bỏ qua cặp đã cache."""
        todo = [(p, s) for p, s in pairs if _key(self._gen_fp, p, s.idx) not in self.cache]
        cached = len(pairs) - len(todo)
        if progress and not todo:
            print(f"{progress} Findings dùng cache {cached}/{len(pairs)} mẫu", flush=True)
        stage = (
            progress_stage(f"{progress} sinh Findings cho {len(todo)} mẫu với {self.num_gpus} bản model, "
                           f"cache {cached}/{len(pairs)}")
            if progress and todo else nullcontext()
        )
        with stage:
            generated = 0
            for batch_no, (chunk, outs) in enumerate(self._generate_missing(todo), start=1):
                self._append_cache(
                    [(_key(self._gen_fp, p, s.idx), o) for (p, s), o in zip(chunk, outs, strict=True)]
                )
                generated += len(chunk)
                if progress and (batch_no == 1 or batch_no % 10 == 0 or generated == len(todo)):
                    print(f"{progress} Findings {cached + generated}/{len(pairs)} mẫu", flush=True)
        # gộp kết quả theo thứ tự yêu cầu
        return [self.cache[_key(self._gen_fp, p, s.idx)] for p, s in pairs]

    def _generate_missing(
        self, pairs: list[tuple[str, Sample]]
    ) -> Iterator[tuple[list[tuple[str, Sample]], list[str]]]:
        chunks = [pairs[i : i + self.batch_size] for i in range(0, len(pairs), self.batch_size)]
        if self.num_gpus == 1 or len(chunks) <= 1:
            for chunk in chunks:
                yield chunk, self._generate_batch(chunk)
            return

        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            for i in range(0, len(chunks), self.num_gpus):
                pending = {
                    executor.submit(self._generate_batch, chunk, gpu): chunk
                    for gpu, chunk in enumerate(chunks[i : i + self.num_gpus])
                }
                error = None
                for future in as_completed(pending):
                    try:
                        results = future.result()
                    except Exception as exc:
                        if error is None:
                            error = exc
                    else:
                        yield pending[future], results
                if error is not None:
                    raise error

    @torch.inference_mode()
    def _generate_batch(self, pairs: list[tuple[str, Sample]], replica_index: int = 0) -> list[str]:
        processor, model = self._replicas[replica_index]
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
        prompts_text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images = [[s.image] for _, s in pairs]
        inputs = processor(text=prompts_text, images=images, padding=True, return_tensors="pt")
        gen_kwargs: dict = dict(max_new_tokens=self.max_new_tokens)
        if self.do_sample:
            gen_kwargs.update(do_sample=True, top_k=0, top_p=0.95)
        with torch.cuda.device(replica_index) if self._use_cuda else nullcontext():
            inputs = inputs.to(model.device)
            output = model.generate(**inputs, **gen_kwargs)
            trimmed = output[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def _append_cache(self, records: list[tuple[str, str]]) -> None:
        for k, v in records:
            self.cache[k] = v
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for k, v in records:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
