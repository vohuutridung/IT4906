"""Evolution LLM — `google/medgemma-27b-text-it` (text-only, 4-bit trên Kaggle 2×T4).

Instruction 4 bước nguyên văn EvoPrompt (Proposal slide 9):
Difference extraction -> Mutation -> Combine (với Prompt 3) -> Crossover (với Basic Prompt).
Decode: temperature 0.5, top_p 0.95, max_tokens 1000.
Parse output theo `get_final_prompt` code gốc: lấy cặp <prompt>...</prompt> CUỐI CÙNG;
thiếu delimiter -> fallback dùng nguyên response (bỏ cặp "..." bao ngoài nếu có).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Template v1 — khớp args.py: <prompt0>=target, <prompt1>,<prompt2>=donors, <prompt3>=best
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
"""


def parse_final_prompt(text: str) -> str:
    """Trích prompt offspring — logic khớp `utils.get_final_prompt` của code gốc."""
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        return prompt.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


class EvolutionLLM:
    def __init__(self, cfg: dict):
        self.model_id: str = cfg["model_id"]
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 1000))
        self.temperature: float = float(cfg.get("temperature", 0.5))
        self.top_p: float = float(cfg.get("top_p", 0.95))
        self._load(cfg)

    def _load(self, cfg: dict) -> None:
        kwargs: dict = {"device_map": "auto"}
        if cfg.get("load_in_4bit", False):
            compute_dtype = getattr(torch, cfg.get("bnb_compute_dtype", "float16"))
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,  # T4: float16; L4/A100: bfloat16
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, cfg.get("dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def evolve(
        self, target_prompt: str, donor1: str, donor2: str, best_prompt: str
    ) -> str:
        """1 request/offspring — trả prompt đã parse (theo code gốc: 1 request, temp 0.5)."""
        user_text = INSTRUCTION_TEMPLATE.format(
            prompt0=target_prompt, prompt1=donor1, prompt2=donor2, prompt3=best_prompt
        )
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
        response = self.tokenizer.decode(trimmed, skip_special_tokens=True)
        return parse_final_prompt(response)
