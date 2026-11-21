"""Sinh Findings từ ảnh và prompt qua endpoint OpenAI-compatible hoặc model local."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .cuda_compat import apply_turing_sdpa_patch
from .data import SYSTEM_PROMPT, Sample
from .progress import progress_stage


DEFAULT_TARGET_MODELS = {"openai": "Qwen3.5-9B", "local": "google/medgemma-1.5-4b-it"}


def resolve_target_config(cfg: dict, evolution_cfg: dict | None = None) -> dict:
    """Điền model mặc định và ghi nhận endpoint mà không lưu thông tin xác thực."""
    result = cfg.copy()
    provider = cfg.get("provider", "local")  # Config và snapshot cũ đều dùng local.
    if provider not in DEFAULT_TARGET_MODELS:
        raise ValueError(f"Provider Target LLM không hợp lệ: {provider}")
    result["provider"] = provider
    result["model_id"] = cfg.get("model_id") or DEFAULT_TARGET_MODELS[provider]
    if provider == "openai":
        evolution_cfg = evolution_cfg or {}
        for field, default in (
            ("base_url_env", "EVOLUTION_LLM_BASE_URL"),
            ("api_key_env", "EVOLUTION_LLM_API_KEY"),
        ):
            result[field] = cfg.get(field) or evolution_cfg.get(field, default)
            if not os.environ.get(result[field]):
                raise RuntimeError(f"Target LLM qua OpenAI-compatible API cần biến {result[field]}.")
        base_url = os.environ[result["base_url_env"]].rstrip("/")
        result["endpoint_fingerprint"] = hashlib.sha256(base_url.encode("utf-8")).hexdigest()
    return result


def target_result_config(cfg: dict) -> dict:
    """Phần cấu hình ảnh hưởng đến kết quả sinh và cache."""
    if cfg.get("provider") == "openai":
        return {
            "provider": "openai",
            "model_id": cfg["model_id"],
            "endpoint_fingerprint": cfg["endpoint_fingerprint"],
            "max_new_tokens": int(cfg.get("max_new_tokens", 128)),
            "do_sample": bool(cfg.get("do_sample", False)),
            "extra_body": cfg.get("extra_body") or {},
        }
    result = cfg.copy()
    for field in ("provider", "base_url_env", "api_key_env", "endpoint_fingerprint", "api_concurrency", "extra_body"):
        result.pop(field, None)
    if not result.get("do_sample", False) or int(result.get("num_gpus", 1)) == 1:
        result.pop("num_gpus", None)
    return result


def _key(gen_fp: str, prompt_text: str, sample_idx: int) -> str:
    raw = f"{gen_fp}||{prompt_text}||{sample_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TargetLLM:
    def __init__(self, cfg: dict, cache_dir: str):
        cfg = resolve_target_config(cfg)
        self.provider: str = cfg["provider"]
        self.model_id: str = cfg["model_id"]
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 128))
        self.do_sample: bool = bool(cfg.get("do_sample", False))  # greedy mặc định cho eval
        if self.provider == "openai":
            self.batch_size = 1  # Mỗi chat completion nhận một cặp prompt và ảnh.
            self._workers = int(cfg.get("api_concurrency", 2))
            if self._workers < 1:
                raise ValueError("target_llm.api_concurrency phải lớn hơn 0.")
            self._extra_body = cfg.get("extra_body") or {}
            generation_cfg = target_result_config(cfg)
        else:
            self.batch_size = int(cfg.get("batch_size", 16))
            self._workers = int(cfg.get("num_gpus", 1))
            if self._workers not in (1, 2):
                raise ValueError("target_llm.num_gpus phải là 1 hoặc 2.")
            self._use_cuda = torch.cuda.is_available()
            if self._workers == 2 and (not self._use_cuda or torch.cuda.device_count() < 2):
                raise ValueError("--target-gpus 2 cần hai GPU CUDA khả dụng.")
            dtype = cfg.get("dtype", "bfloat16")
            self.dtype = getattr(torch, dtype)
            generation_cfg = target_result_config({
                "model_id": self.model_id,
                "dtype": dtype,
                "do_sample": self.do_sample,
                "max_new_tokens": self.max_new_tokens,
                "num_gpus": self._workers,
            })
        self._gen_fp = hashlib.sha1(
            json.dumps(generation_cfg, sort_keys=True).encode("utf-8")
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
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(
                base_url=os.environ[cfg["base_url_env"]], api_key=os.environ[cfg["api_key_env"]], timeout=120.0
            )
            print(f"[target] provider=openai model={self.model_id} concurrency={self._workers}", flush=True)
        else:
            self._load_local()

    @property
    def parallel_batch_size(self) -> int:
        return self.batch_size * self._workers

    def _load_local(self) -> None:
        apply_turing_sdpa_patch()
        self._replicas = []
        for gpu in range(self._workers):
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
        return self.generate_batch([(prompt_text, sample)])[0]

    def generate_batch(self, pairs: list[tuple[str, Sample]], *, progress: str | None = None) -> list[str]:
        """Sinh Findings cho nhiều (prompt, sample); bỏ qua cặp đã cache."""
        todo = [(p, s) for p, s in pairs if _key(self._gen_fp, p, s.idx) not in self.cache]
        cached = len(pairs) - len(todo)
        if progress and not todo:
            print(f"{progress} Findings dùng cache {cached}/{len(pairs)} mẫu", flush=True)
        backend = (
            f"API, {self._workers} request song song" if self.provider == "openai" else f"{self._workers} bản model"
        )
        stage = (
            progress_stage(f"{progress} sinh Findings cho {len(todo)} mẫu với {backend}, "
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
        if self._workers == 1 or len(chunks) <= 1:
            for chunk in chunks:
                yield chunk, self._generate_batch(chunk)
            return

        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            for i in range(0, len(chunks), self._workers):
                pending = {
                    executor.submit(self._generate_batch, chunk, replica_index): chunk
                    for replica_index, chunk in enumerate(chunks[i : i + self._workers])
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

    def _generate_batch(self, pairs: list[tuple[str, Sample]], replica_index: int = 0) -> list[str]:
        if self.provider == "openai":
            return [self._generate_api(prompt, sample) for prompt, sample in pairs]
        return self._generate_local_batch(pairs, replica_index)

    def _generate_api(self, prompt_text: str, sample: Sample) -> str:
        from openai import APIError

        with BytesIO() as image:
            sample.image.convert("RGB").save(image, format="PNG")
            image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        options = {"top_p": 0.95} if self.do_sample else {}
        if self._extra_body:
            options["extra_body"] = self._extra_body
        error_context = f"Target LLM API (model={self.model_id}, sample={sample.idx})"
        try:
            response = self._client.chat.completions.create(
                model=self.model_id, messages=messages, max_tokens=self.max_new_tokens,
                temperature=1.0 if self.do_sample else 0.0, **options,
            )
        except APIError as exc:
            raise RuntimeError(f"{error_context}: {exc}") from exc
        if not response.choices:
            raise RuntimeError(f"{error_context}: phản hồi thiếu choices.")
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            raise RuntimeError(f"{error_context}: hết max_tokens trước khi sinh Findings.")
        return text

    @torch.inference_mode()
    def _generate_local_batch(self, pairs: list[tuple[str, Sample]], replica_index: int) -> list[str]:
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
