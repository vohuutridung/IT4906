"""Tương thích VRAM cho GPU đời trước Ampere (T4/Kaggle = sm75) khi chạy bf16.

Vấn đề (chẩn đoán 2026-11-11, OOM thật ở SigLIP vision tower):
- PyTorch SDPA trên GPU sm<80: flash attention chỉ sm80+; kernel memory-efficient
  với fp16 chạy từ sm50 nhưng với **bf16 cũng cần sm80+**.
- → tensor bf16 rớt về backend "math": materialize ma trận attention O(n²) (nâng
  fp32 cho softmax) → MedGemma vision tower (SigLIP, 4096 patch/ảnh) batch 5 cần
  ~5GB transient MỖI LỚP → CUDA OOM dù trọng số vừa đầy ("Tried to allocate
  5.00 GiB"). fp16 trước đó không OOM vì fp16 dùng được mem-efficient trên T4.

Giải pháp: wrap `F.scaled_dot_product_attention` — khi input bf16 trên GPU sm<80,
tính attention trong fp16 (dùng được kernel memory-efficient) rồi cast kết quả
về bf16.
- Trọng số VẪN bf16 → giữ hành vi số học đã xác nhận chạy đúng trên T4
  (fp16 toàn model gây NaN logits — không dùng lại).
- Attention trong fp16 an toàn: Gemma3/MedGemma có QK-norm (chuẩn hoá q,k trước
  attention) nên độ lớn q,k bị chặn; attention fp16 là chuẩn ngành với model fp16.
- Patch idempotent; GPU sm≥80 hoặc dtype khác bf16 → gọi hàm gốc, không đổi gì.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_orig_sdpa = F.scaled_dot_product_attention
_applied = False


def _turing_sdpa(query, key, value, *args, **kwargs):
    if query.dtype == torch.bfloat16 and query.is_cuda:
        major, _ = torch.cuda.get_device_capability(query.device)
        if major < 8:
            # Cast mọi tensor thực (q,k,v + attn_mask nếu là float) sang fp16 để
            # dùng được kernel memory-efficient; scale/is_causal/bool mask giữ nguyên.
            args = tuple(
                a.to(torch.float16)
                if isinstance(a, torch.Tensor) and a.is_floating_point()
                else a
                for a in args
            )
            for k, v in list(kwargs.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    kwargs[k] = v.to(torch.float16)
            out = _orig_sdpa(
                query.to(torch.float16),
                key.to(torch.float16),
                value.to(torch.float16),
                *args,
                **kwargs,
            )
            return out.to(query.dtype)
    return _orig_sdpa(query, key, value, *args, **kwargs)


def apply_turing_sdpa_patch() -> bool:
    """Áp patch nếu đang chạy GPU sm<80. Idempotent. True = patch có hiệu lực."""
    global _applied
    if _applied or not torch.cuda.is_available():
        return _applied
    legacy_gpu = next(
        (i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i)[0] < 8),
        None,
    )
    if legacy_gpu is None:
        return False
    F.scaled_dot_product_attention = _turing_sdpa
    _applied = True
    print(
        f"[cuda_compat] cuda:{legacy_gpu}: bf16 không có kernel SDPA tiết kiệm VRAM "
        "(flash/mem-efficient cần sm80+) → attention tính trong fp16, trọng số vẫn "
        "bf16 — tránh OOM backend math O(n²)"
    )
    return True
