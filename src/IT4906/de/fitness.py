"""Fitness = weight_bertscore × BERTScore(F1) + weight_rouge × ROUGE-L(F1).

Chốt tạm 2026-11-10 (swappable qua config — ưu tiên đổi được model để so sánh sau):
- BERTScore model: xlm-roberta-large (multilingual cho tiếng Việt, không cần word segmentation)
- ROUGE-L: F1 (rouge-score, use_stemmer=False)
Cache fitness theo (prompt_text + fingerprint cấu hình metric) => đổi metric/model
chỉ re-score, KHÔNG chạy lại target LLM (cache generation độc lập với metric).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rouge_score import rouge_scorer


def _metric_fingerprint(cfg: dict) -> str:
    payload = json.dumps(
        {
            "bertscore_model": cfg.get("bertscore_model"),
            "bertscore_lang": cfg.get("bertscore_lang"),
            "rouge_type": cfg.get("rouge_type", "rougeL"),
            "use_stemmer": bool(cfg.get("rouge_use_stemmer", False)),
            "w_bertscore": cfg.get("weight_bertscore", 0.7),
            "w_rouge": cfg.get("weight_rouge", 0.3),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


class FitnessEvaluator:
    def __init__(self, cfg: dict, cache_dir: str):
        self.bertscore_model: str = cfg["bertscore_model"]
        self.bertscore_lang: str | None = cfg.get("bertscore_lang")
        self.rouge_type: str = cfg.get("rouge_type", "rougeL")
        self.use_stemmer: bool = bool(cfg.get("rouge_use_stemmer", False))
        self.w_bertscore: float = float(cfg.get("weight_bertscore", 0.7))
        self.w_rouge: float = float(cfg.get("weight_rouge", 0.3))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self._bertscore = None  # lazy: nạp khi lần đầu tính
        self._rouge = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=self.use_stemmer)
        self.fp = _metric_fingerprint(cfg)
        self.cache_path = Path(cache_dir) / f"fitness_cache_{self.fp}.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _ensure_bertscore(self):
        if self._bertscore is None:
            from bert_score import BERTScorer

            kwargs: dict = {"model_type": self.bertscore_model, "batch_size": self.batch_size}
            if self.bertscore_lang:
                kwargs["lang"] = self.bertscore_lang
            self._bertscore = BERTScorer(**kwargs)
        return self._bertscore

    def score(self, prompt_text: str, cands: list[str], refs: list[str]) -> dict:
        """Trả {"bertscore_f1", "rougeL_f1", "fitness"} đã cache theo prompt."""
        k = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()
        if k in self.cache:
            return self.cache[k]
        bert = self._ensure_bertscore()
        _, _, bf1 = bert.score(cands, refs)
        bertscore_f1 = float(bf1.mean().item())
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
        return rec

    def _save(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8"
        )
