"""Evolution LLM — các provider trong `evolution_llm.provider`:

- "openai" (mặc định theo chỉ dẫn của Dũng 2026-11-13): gọi endpoint OpenAI-compatible
  qua `OpenAI(base_url=..., api_key=...)` — GIÁ TRỊ THẬT nằm trong biến môi trường
  (file `.env`, được `run_de.py` nạp; `.env` đã git-ignore, KHÔNG hardcode vào code).
- "local": load `google/medgemma-27b-text-it` bằng transformers (4-bit trên Kaggle 2×T4).
- "openrouter": dùng `OPENROUTER_API_KEY` và bật reasoning trong lượt sinh offspring.

Instruction 4 bước nguyên văn EvoPrompt (Proposal slide 9):
Difference extraction -> Mutation -> Combine (với Prompt 3) -> Crossover (với Basic Prompt).
Decode: temperature 0.5, top_p 0.95, max_tokens 1500 (lệch official 1000 — CẦN BÁO NHÓM;
2026-11-12: 27B trích dẫn nguyên văn donors trong 4 bước → trần 1000 cắt TRƯỚC thẻ
<prompt> → fallback = text phân tích → offspring rác 0.58. Template thêm dòng cấm
trích-dẫn + trần 1500; VRAM an toàn vì sliding_window 1024 chặn KV ~0.65GiB).
Parse output theo `get_final_prompt` code gốc: lấy cặp <prompt>...</prompt> CUỐI CÙNG;
thiếu delimiter -> fallback dùng nguyên response (bỏ cặp "..." bao ngoài nếu có) —
TRỪ khi response là bản phân tích CoT (bỏ thẻ) → trả "" để de_loop giữ prompt cũ
+ bỏ eval, tiết kiệm ~200 target-gen cho 1 offspring rác (thật 2026-11-12: "Okay,
let's break this down…", "**Step 1…" — dấu * chặn startswith nên phải normalize).
"""

from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .cuda_compat import apply_turing_sdpa_patch

DEFAULT_EVOLUTION_MODELS = {
    "openai": "google/medgemma-27b-text-it",
    "openrouter": "openai/gpt-oss-20b",
    "local": "google/medgemma-27b-text-it",
}

# Template v1 — khớp args.py: <prompt0>=target, <prompt1>,<prompt2>=donors, <prompt3>=best
# Dòng "Important" thêm 2026-11-12 (deviation nhẹ cần báo nhóm): template gốc được thiết kế
# cho GPT-3.5 chấp hành format tốt hơn MedGemma 27B — 27B hay trích dẫn nguyên văn donors
# khiến 4 bước không kịp khép thẻ <prompt> trong trần token.
INSTRUCTION_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
   Prompt 1: {prompt1}
   Prompt 2: {prompt2}
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with
   the different parts in Step 2 and generate a new prompt:
   Prompt 3: {prompt3}
4. Cross over the prompt in the Step 3 with the following basic prompt and
   generate a final prompt bracketed with <prompt> and </prompt>:
   Basic Prompt: {prompt0}

Important: do NOT repeat or quote the given prompts in your answer — describe each
step in at most one short line, then output ONLY the final prompt bracketed with
<prompt> and </prompt>. Do not output anything after the closing tag.
"""

# Dấu hiệu response là bản phân tích CoT (không phải prompt) — chỉ áp dụng khi KHÔNG có
# thẻ <prompt>: trả "" thay vì fallback nguyên response (khác code gốc get_final_prompt).
_COT_MARKERS = ("okay", "step 1", "let's", "let us", "sure", "certainly", "got it")


def _looks_like_cot(text: str) -> bool:
    """Nhận dạng CoT bỏ thẻ; lstrip markdown vì '**Step 1…' chặn startswith (thật 2026-11-12)."""
    head = text.lstrip("*#`> \n\t").lower()
    return head.startswith(_COT_MARKERS) or "step 1" in head[:300]


def parse_final_prompt(text: str | list[str]) -> str:
    """Trích prompt offspring — logic khớp `utils.get_final_prompt` của code gốc.

    Khác code gốc 1 điểm (vá 2026-11-12): response không có thẻ VÀ có dạng phân tích
    CoT → trả "" (de_loop giữ prompt cũ + bỏ eval) thay vì dùng nguyên response.
    """
    if isinstance(text, (list, tuple)):  # phòng khi decode trả về batch (transformers 5.x)
        text = text[0] if text else ""
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        return prompt.strip()
    if _looks_like_cot(text):
        return ""
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


class EvolutionLLM:
    def __init__(self, cfg: dict):
        self.provider: str = str(cfg.get("provider", "openai"))
        if self.provider not in DEFAULT_EVOLUTION_MODELS:
            raise ValueError(f"Provider Evolution LLM không hợp lệ: {self.provider}")
        self.model_id: str = cfg.get("model_id") or DEFAULT_EVOLUTION_MODELS[self.provider]
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 1000))
        self.temperature: float = float(cfg.get("temperature", 0.5))
        self.top_p: float = float(cfg.get("top_p", 0.95))
        if self.provider == "openai":
            self._load_openai(cfg)
        elif self.provider == "openrouter":
            self._load_openrouter()
        else:
            self._load(cfg)

    def _load_openai(self, cfg: dict) -> None:
        """Client OpenAI-compatible — base_url + api_key ĐỌC TỪ BIẾN MÔI TRƯỜNG
        (tên biến đặt trong config; giá trị thật trong `.env`, được `run_de.py` nạp).
        Không hardcode credential vào code/repo (chỉ dẫn Dũng 2026-11-13)."""
        from openai import OpenAI  # import muộn: chạy provider "local" không cần gói openai

        base_url_env = str(cfg.get("base_url_env", "EVOLUTION_LLM_BASE_URL"))
        api_key_env = str(cfg.get("api_key_env", "EVOLUTION_LLM_API_KEY"))
        base_url = os.environ.get(base_url_env)
        api_key = os.environ.get(api_key_env)
        missing = [
            name for name, value in ((base_url_env, base_url), (api_key_env, api_key)) if not value
        ]
        if missing:
            raise RuntimeError(
                f"evolution_llm.provider='openai' nhưng thiếu biến môi trường: {', '.join(missing)}. "
                "Copy .env.example thành .env rồi điền base_url + api_key thật "
                "(KHÔNG hardcode vào code — .env đã git-ignore)."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        print(f"[evolution] provider=openai — model={self.model_id} base_url={base_url}")

    def _load_openrouter(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu OPENROUTER_API_KEY cho evolution_llm.provider='openrouter'.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"[evolution] provider=openrouter — model={self.model_id}")

    def _load(self, cfg: dict) -> None:
        apply_turing_sdpa_patch()
        kwargs: dict = {"device_map": "auto"}
        if torch.cuda.device_count() >= 2:
            # 2 GPU (Kaggle T4×2): 27B 4-bit cần ~14.5GiB (embed bf16 2.8GB không
            # quantize được; bnb 4-bit KHÔNG chạy trên CPU → không được để rơi layer).
            # ⚠️ transformers TỰ TRỪ 10% max_memory cho buffer quantize
            # (Bnb4BitHfQuantizer.adjust_max_memory) → cap 4+12=16GiB chỉ còn 14.4
            # → thiếu ~0.1GiB → module rơi CPU → ValueError (đã gặp thật 2026-11-11).
            # Cap 4+14 → budget 16.2GiB ✓; GPU0 giữ ~2.6GiB cho KV cache target.
            kwargs["max_memory"] = {0: "4GiB", 1: "14GiB"}
        if cfg.get("load_in_4bit", False):
            compute_dtype = getattr(torch, cfg.get("bnb_compute_dtype", "float16"))
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,  # bf16 (fp16 gây NaN trên T4 — đã chẩn đoán)
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = getattr(torch, cfg.get("dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

    def evolve(
        self, target_prompt: str, donor1: str, donor2: str, best_prompt: str
    ) -> str:
        """Sinh và bóc tách prompt con từ phản hồi của provider."""
        return parse_final_prompt(self.generate_response(target_prompt, donor1, donor2, best_prompt))

    def generate_response(
        self, target_prompt: str, donor1: str, donor2: str, best_prompt: str
    ) -> str:
        """Trả nội dung phản hồi nguyên văn để có thể lưu trước khi bóc tách."""
        user_text = INSTRUCTION_TEMPLATE.format(
            prompt0=target_prompt, prompt1=donor1, prompt2=donor2, prompt3=best_prompt
        )
        if self.provider == "openai":
            response = self._request_chat(user_text)
        elif self.provider == "openrouter":
            response = self._request_chat(user_text, extra_body={"reasoning": {"enabled": True}})
        else:
            response = self._generate_local(user_text)
        return response

    def _request_chat(self, user_text: str, **options) -> str:
        """Gọi Chat Completions và trả nội dung phản hồi."""
        last_err: Exception | None = None
        for attempt in range(1, 4):  # DE chạy dài, endpoint có thể giật → thử lại 3 lần
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": user_text}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                    **options,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001 — log rồi thử lại, hết lượt mới raise
                last_err = err
                print(f"[evolution] gọi API lỗi (lần {attempt}/3): {err}")
                time.sleep(5 * attempt)
        raise RuntimeError(
            f"Evolution LLM qua API thất bại sau 3 lần thử (model={self.model_id})"
        ) from last_err

    @torch.inference_mode()
    def _generate_local(self, user_text: str) -> str:
        """Sinh bằng model local (transformers) — path cũ khi không có endpoint."""
        # GPU 1 gần đầy (27B ~12GiB + BERTScore): mảnh vỡ allocator 100-200MiB đủ
        # gây OOM lúc prefill (đã gặp thật) → dọn trước mỗi lần sinh
        torch.cuda.empty_cache()
        messages = [{"role": "user", "content": user_text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        trimmed = output[:, inputs["input_ids"].shape[1]:]
        # transformers 5.x: decode với tensor 2D trả về LIST chuỗi → lấy phần tử đầu
        response = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        return response
