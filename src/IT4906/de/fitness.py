"""Fitness = weight_bertscore × BERTScore(F1) + weight_rouge × ROUGE-L(F1).

Chốt tạm 2026-11-10 (swappable qua config — ưu tiên đổi được model để so sánh sau):
- BERTScore model: xlm-roberta-large (multilingual cho tiếng Việt, không cần word segmentation)
- ROUGE-L: F1 (rouge-score, use_stemmer=False)
Cache fitness theo (số sample + prompt_text + fingerprint cấu hình metric) => đổi
metric/model chỉ re-score, KHÔNG chạy lại target LLM (cache generation độc lập với metric).
"""

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

from .artifacts import write_json
from .progress import progress_stage


def check_metric_dependencies(*, include_bleu: bool = False) -> None:
    """Kiểm tra thư viện tính điểm trước khi nạp Target LLM."""
    modules = [("bert_score", "bert-score"), ("rouge_score.rouge_scorer", "rouge-score")]
    if include_bleu:
        modules.append(("sacrebleu.metrics", "sacrebleu"))

    missing = []
    for module, package in modules:
        try:
            import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module.split(".", 1)[0]:
                raise
            missing.append(package)

    if missing:
        raise ModuleNotFoundError(
            f"Thiếu thư viện tính điểm: {', '.join(missing)}. "
            "Chạy `bash deployment/kaggle/run.sh --mode setup` với cùng PYTHON_BIN rồi thử lại."
        )


def _metric_fingerprint(cfg: dict) -> str:
    payload = json.dumps(
        {
            "bertscore_model": cfg.get("bertscore_model"),
            "bertscore_lang": cfg.get("bertscore_lang"),
            "rouge_type": cfg.get("rouge_type", "rougeL"),
            "use_stemmer": bool(cfg.get("rouge_use_stemmer", False)),
            "w_bertscore": cfg.get("weight_bertscore", 0.7),
            "w_rouge": cfg.get("weight_rouge", 0.3),
            "bertscore_bf16": bool(cfg.get("bertscore_bf16", False)),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


class FitnessEvaluator:
    def __init__(self, cfg: dict, cache_dir: str):
        from rouge_score import rouge_scorer

        self.bertscore_model: str = cfg["bertscore_model"]
        self.bertscore_lang: str | None = cfg.get("bertscore_lang")
        self.rouge_type: str = cfg.get("rouge_type", "rougeL")
        self.use_stemmer: bool = bool(cfg.get("rouge_use_stemmer", False))
        self.w_bertscore: float = float(cfg.get("weight_bertscore", 0.7))
        self.w_rouge: float = float(cfg.get("weight_rouge", 0.3))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.bertscore_bf16: bool = bool(cfg.get("bertscore_bf16", False))
        self.device_cfg: str | None = cfg.get("device")  # VD "cuda:1"; None = tự chọn
        self._bertscore = None  # lazy: nạp khi lần đầu tính
        self._rouge = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=self.use_stemmer)
        self.fp = _metric_fingerprint(cfg)
        self.cache_path = Path(cache_dir) / f"fitness_cache_{self.fp}.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _ensure_bertscore(self, progress: str):
        if self._bertscore is None:
            import torch
            from bert_score import BERTScorer

            kwargs: dict = {"model_type": self.bertscore_model, "batch_size": self.batch_size}
            if self.bertscore_lang:
                kwargs["lang"] = self.bertscore_lang
            device = self.device_cfg or self._auto_device()
            try:
                with progress_stage(f"{progress} nạp BERTScore {self.bertscore_model} trên {device}"):
                    self._bertscore = BERTScorer(**kwargs, device=device)
                    if device != "cpu" and self.bertscore_bf16:
                        self._bertscore._model.to(torch.bfloat16)
            except torch.OutOfMemoryError:
                print(
                    f"[fitness] OOM trên {device} → BERTScore chuyển sang CPU "
                    "(chậm hơn nhưng vẫn chạy)"
                )
                with progress_stage(f"{progress} nạp BERTScore {self.bertscore_model} trên CPU"):
                    self._bertscore = BERTScorer(**kwargs, device="cpu")
        return self._bertscore

    @staticmethod
    def _auto_device() -> str:
        """GPU0 thường đã chứa target LLM + một phần evolution model → ưu tiên GPU1."""
        import torch

        n = torch.cuda.device_count()
        if n >= 2:
            return "cuda:1"
        return "cuda" if n == 1 else "cpu"

    def score(self, prompt_text: str, cands: list[str], refs: list[str], *, progress: str = "[fitness]") -> dict:
        """Trả {"bertscore_f1", "rougeL_f1", "fitness"} đã cache theo prompt."""
        # Key PHẢI gồm len(refs) (= số dev sample): khoá cũ chỉ sha1(prompt) khiến fitness
        # chấm trên thang 100 sample được tái dùng âm thầm cho run dev 200 (poison vector
        # 2026-11-13). Đổi num_samples → key đổi → tự re-score từ gen cache (vài phút),
        # không bao giờ trộn 2 thang đo vào cùng bảng xếp hạng.
        k = hashlib.sha1(f"{len(refs)}||{prompt_text}".encode("utf-8")).hexdigest()
        if k in self.cache:
            print(f"{progress} dùng fitness đã lưu cho {len(refs)} mẫu", flush=True)
            return self.cache[k]
        # Output rỗng (model không sinh chữ nào) phải bị chấm THẤP chứ không làm crash:
        # bert_score với chuỗi rỗng gọi tokenizer.build_inputs_with_special_tokens —
        # API mà transformers 5.x đã gỡ bỏ → AttributeError. Thay placeholder để điểm tự thấp.
        n_empty = sum(1 for c in cands if not (c and c.strip()))
        if n_empty:
            print(
                f"{progress} cảnh báo: {n_empty}/{len(cands)} output rỗng → chấm thấp "
                f"(prompt: {prompt_text[:60]}...)"
            )
        cands = [c if (c and c.strip()) else "(trống)" for c in cands]
        refs = [r if (r and r.strip()) else "(trống)" for r in refs]
        bert = self._ensure_bertscore(progress)
        with progress_stage(f"{progress} tính BERTScore F1 trên {len(cands)} mẫu"):
            _, _, bf1 = bert.score(cands, refs)
            bertscore_f1 = float(bf1.mean().item())
        with progress_stage(f"{progress} tính {self.rouge_type} F1 trên {len(cands)} mẫu"):
            rouge_f1 = sum(
                self._rouge.score(ref, cand)[self.rouge_type].fmeasure
                for cand, ref in zip(cands, refs)
            ) / max(len(cands), 1)
        fitness = self.w_bertscore * bertscore_f1 + self.w_rouge * rouge_f1
        rec = {
            "bertscore_f1": bertscore_f1,
            "rougeL_f1": rouge_f1,
            "fitness": fitness,
        }
        self.cache[k] = rec
        self._save()
        print(f"{progress} BERTScore={bertscore_f1:.4f} {self.rouge_type}={rouge_f1:.4f} "
              f"fitness={fitness:.4f}", flush=True)
        return rec

    def _save(self) -> None:
        write_json(self.cache_path, self.cache)
