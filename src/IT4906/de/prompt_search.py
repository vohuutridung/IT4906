"""Shared helpers for prompt-search baselines that score candidates on validation."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import islice


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def extract_prompt(response: str) -> str:
    response = (response or "").strip()
    if "<prompt>" in response:
        response = response.rsplit("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
    else:
        response = response.removeprefix("```text").removeprefix("```").removesuffix("```").strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1].strip()
    if not response:
        raise ValueError("Optimizer trả về prompt rỗng; kiểm tra model/API rồi chạy lại.")
    if len(response) > 5000:
        raise ValueError(f"Prompt đề xuất quá dài ({len(response)} ký tự); không chấp nhận.")
    return response


def optimizer_client(cfg: dict):
    from openai import OpenAI

    base_env = str(cfg.get("base_url_env", "EVOLUTION_LLM_BASE_URL"))
    key_env = str(cfg.get("api_key_env", "EVOLUTION_LLM_API_KEY"))
    missing = [name for name in (base_env, key_env) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Thiếu Kaggle Secrets/biến môi trường: {', '.join(missing)}")
    return OpenAI(base_url=os.environ[base_env], api_key=os.environ[key_env], timeout=120)


def optimizer_identity(cfg: dict) -> dict:
    """Fingerprint optimizer settings without storing its URL or API key."""
    base_env = str(cfg.get("base_url_env", "EVOLUTION_LLM_BASE_URL"))
    base_url = os.environ.get(base_env, "").rstrip("/")
    endpoint_fp = hashlib.sha256(base_url.encode()).hexdigest()[:16] if base_url else None
    return {
        "model_id": str(cfg.get("model_id", "Qwen3.5-9B")),
        "temperature": float(cfg.get("temperature", 0.5)),
        "max_tokens": int(cfg.get("max_tokens", 1000)),
        "endpoint_fingerprint": endpoint_fp,
    }


def evaluate_prompt(prompt, target, metric, samples, refs, validation_samples, progress):
    """Generate and score one candidate against the unchanged validation sample set."""
    from .data import render_prompt
    from .progress import progress_stage

    predictions = []
    stream = validation_samples()
    with progress_stage(f"{progress} sinh Findings trên {len(samples)} mẫu (có cache)"):
        while batch := list(islice(stream, target.parallel_batch_size)):
            try:
                offset = len(predictions)
                for expected, actual in zip(samples[offset:offset + len(batch)], batch, strict=True):
                    if (expected.idx, expected.meta, expected.findings_gt) != (actual.idx, actual.meta, actual.findings_gt):
                        raise ValueError("Dữ liệu validation thay đổi giữa các lượt đọc.")
                predictions.extend(target.generate_batch([(render_prompt(prompt, s), s) for s in batch]))
            finally:
                for sample in batch:
                    sample.image.close()
            if len(predictions) % 20 == 0 or len(predictions) == len(samples):
                print(f"{progress} Findings {len(predictions)}/{len(samples)} mẫu", flush=True)
    if len(predictions) != len(refs):
        raise ValueError("Số dự đoán không khớp số tham chiếu validation.")
    return metric.score(prompt, predictions, refs, progress=progress)
