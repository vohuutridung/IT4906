"""Vòng lặp DE — khớp semantics code gốc `DEEvoluter.evolute()` (beeevita/EvoPrompt):

- Pool ban đầu = prompts.txt + prompts_auto.txt (manual + ape), eval TOÀN BỘ pool trên dev.
- P₀ chọn từ pool theo `initial_mode` (topk = top-N fitness).
- best ban đầu = best trên TOÀN BỘ pool đã eval (không chỉ P₀); tie -> phần tử đầu (max()).
- Mỗi thế hệ: lần lượt từng prompt làm target; donors a,b = np.random.choice(candidates,
  3, replace=False) (candidates = population trừ target), c = best nếu donor_random=False.
- Offspring chỉ thay target nếu fitness(offspring) > fitness(target) (tie giữ old).
- p_best cập nhật NGAY theo từng offspring, điều kiện strict `>`.
- Checkpoint sau mỗi thế hệ -> resume được khi session Kaggle đứt.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATASET_ID, Sample, load_split, render_prompt
from .evolution_llm import EvolutionLLM
from .fitness import FitnessEvaluator
from .target_llm import TargetLLM


def read_prompt_pool(pool_files: list[str]) -> list[str]:
    """Đọc pool theo thứ tự file (manual trước, auto sau — như code gốc manual_pop + ape_pop)."""
    pool: list[str] = []
    for path in pool_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pool.append(line)
    return pool


def k_init_pop(initial_mode: str, pool_sorted: list[str], k: int) -> list[str]:
    """Khớp `utils.k_init_pop` code gốc. Mode para_* cần paraphrase -> bổ sung khi nhóm chốt."""
    if initial_mode == "topk":
        return list(pool_sorted[:k])
    if initial_mode == "randomk":
        return random.sample(pool_sorted, k)
    raise ValueError(
        f"initial_mode '{initial_mode}' chưa implement. Dùng 'topk' hoặc 'randomk'."
    )


def run_fingerprint(cfg: dict) -> str:
    """Fingerprint phần config ảnh hưởng KẾT QUẢ (bỏ key thuần vận hành).

    Được ghi vào checkpoint: `_try_resume` chỉ chấp nhận checkpoint khi khớp —
    checkpoint sinh ra từ cấu hình khác (đổi metric, dtype, num_samples...)
    sẽ bị LOẠI thay vì resume âm thầm với fitness trộn 2 thang đo (rủi ro thật
    2026-11-11: thêm bertscore_bf16 đổi nhẹ thang đo, checkpoint cũ còn nằm đó).
    - Cấu hình nguồn dữ liệu được giữ trong fingerprint để tách checkpoint giữa các nguồn.
    - Bỏ `de.cache_dir` (thư mục out) và `fitness.device` (bố trí GPU) — không đổi kết quả.
    """
    cfg2 = json.loads(json.dumps(cfg))  # deep copy
    cfg2.get("de", {}).pop("cache_dir", None)
    cfg2.get("fitness", {}).pop("device", None)
    payload = json.dumps(cfg2, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


class DERunner:
    def __init__(self, cfg: dict, out_dir: str):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.N: int = int(cfg["de"]["population_size"])
        self.T: int = int(cfg["de"]["iterations"])
        self.initial_mode: str = cfg["de"].get("initial_mode", "topk")
        self.donor_random: bool = bool(cfg["de"].get("donor_random", False))

        data_cfg = cfg["data"]
        dataset_dir = data_cfg.get("dataset_dir")
        dataset_id = data_cfg.get("dataset_id", DEFAULT_DATASET_ID)
        split = data_cfg.get("split", "val")
        source = {"parquet": str(Path(dataset_dir).resolve())} if dataset_dir else {"hub": dataset_id}
        data_key = json.dumps({"source": source, "split": split}, sort_keys=True)
        data_fingerprint = hashlib.sha1(data_key.encode("utf-8")).hexdigest()[:10]
        cache_dir = str(self.out_dir / "cache" / data_fingerprint)
        print(f"[data] source={dataset_dir or dataset_id} split={split}")
        self.samples: list[Sample] = load_split(
            dataset_dir, split, int(data_cfg.get("num_samples", 200)), dataset_id=dataset_id
        )
        self.target = TargetLLM(cfg["target_llm"], cache_dir)
        self.fitness = FitnessEvaluator(cfg["fitness"], cache_dir)
        self.evolution = EvolutionLLM(cfg["evolution_llm"])

        self.evaluated: dict[str, dict] = {}  # prompt -> {"fitness", "bertscore_f1", "rougeL_f1"}
        self.history: list[dict] = []
        self.checkpoint_path = self.out_dir / "checkpoint.json"
        self._run_fp = run_fingerprint(cfg)

    # ---------- evaluation ----------
    def evaluate_prompt(self, prompt_text: str, samples: list[Sample] | None = None) -> dict:
        """Sinh Findings cho prompt trên toàn bộ dev samples rồi tính fitness (có cache 2 lớp)."""
        samples = samples if samples is not None else self.samples
        pairs = [(render_prompt(prompt_text, s), s) for s in samples]
        cands = self.target.generate_batch(pairs)
        refs = [s.findings_gt for s in samples]
        return self.fitness.score(prompt_text, cands, refs)

    # ---------- checkpoint ----------
    def _save_checkpoint(self, step_done: int, population: list[str], best_prompt: str, best_score: float) -> None:
        ckpt = {
            "run_fp": self._run_fp,
            "step_done": step_done,
            "population": population,
            "best_prompt": best_prompt,
            "best_score": best_score,
            "evaluated_prompts": self.evaluated,
            "history": self.history,
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.checkpoint_path)

    def _try_resume(self) -> tuple[int, list[str], str, float] | None:
        if not self.checkpoint_path.exists():
            return None
        ckpt = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if ckpt.get("run_fp") != self._run_fp:
            # Checkpoint sinh ra từ cấu hình khác → fitness trong đó không so được
            # với thang đo hiện tại → LOẠI (đổi tên giữ lại để tra cứu, không resume).
            stale = self.out_dir / "checkpoint_stale.json"
            self.checkpoint_path.replace(stale)
            print(
                f"[checkpoint] BỎ checkpoint cũ — sinh từ cấu hình khác "
                f"(run_fp {ckpt.get('run_fp')} ≠ {self._run_fp}) → lưu thành {stale.name}. "
                "Pool sẽ được chấm lại, có thể dùng cache generation nếu còn khớp."
            )
            return None
        self.evaluated = ckpt["evaluated_prompts"]
        self.history = ckpt["history"]
        print(
            f"[resume] từ sau thế hệ {ckpt['step_done']} — population {len(ckpt['population'])}, "
            f"best={ckpt['best_score']:.4f}"
        )
        return ckpt["step_done"], ckpt["population"], ckpt["best_prompt"], ckpt["best_score"]

    # ---------- main loop ----------
    def run(self) -> dict:
        t0 = time.time()
        print(f"[init] dev samples: {len(self.samples)}")
        pool = read_prompt_pool(self.cfg["de"]["pool_files"])
        print(f"[init] pool = {len(pool)} prompts (trộn {len(self.cfg['de']['pool_files'])} file)")

        resumed = self._try_resume()
        if resumed is None:
            # 1) Eval toàn bộ pool (khớp --initial all của code gốc)
            for i, p in enumerate(pool):
                if p not in self.evaluated:
                    self.evaluated[p] = self.evaluate_prompt(p)
                    print(
                        f"[pool {i + 1}/{len(pool)}] fitness={self.evaluated[p]['fitness']:.4f} "
                        f"({p[:60]}...)"
                    )
            pool_sorted = sorted(
                self.evaluated.items(), key=lambda kv: kv[1]["fitness"], reverse=True
            )
            # 2) P0 theo initial_mode
            population = k_init_pop(self.initial_mode, [p for p, _ in pool_sorted], self.N)
            # 3) best ban đầu = best trên toàn bộ pool đã eval (tie: phần tử đầu — max() code gốc)
            best_prompt, best_scores = max(
                self.evaluated.items(), key=lambda kv: kv[1]["fitness"]
            )
            best_score = best_scores["fitness"]
            print(f"[P0] = top-{len(population)} theo {self.initial_mode}; best={best_score:.4f}")
            step_done = 0  # lần đầu chạy: chưa có thế hệ nào hoàn thành
            self._save_checkpoint(0, population, best_prompt, best_score)
        else:
            step_done, population, best_prompt, best_score = resumed
            if step_done >= self.T:
                return self._finish(step_done, population, best_prompt, best_score, t0)

        for step in range(step_done + 1, self.T + 1):
            gen_log: dict = {"step": step, "offspring": []}
            new_pop: list[str] = []
            for j in range(self.N):
                old = population[j]
                old_scores = self.evaluated[old]
                # donors: khớp code gốc — choice 3 không thay thế từ population trừ target,
                # rồi c = best (donor_random=False)
                candidates = [population[k] for k in range(self.N) if k != j]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                if not self.donor_random:
                    c = best_prompt
                offspring = self.evolution.evolve(target_prompt=old, donor1=a, donor2=b, best_prompt=c)
                if not offspring.strip():
                    print(
                        f"[gen {step} pop {j + 1}/{self.N}] cảnh báo: offspring rỗng từ Evolution LLM "
                        "→ giữ target, bỏ qua eval (tiết kiệm 1 lần eval GPU)"
                    )
                    gen_log["offspring"].append(
                        {
                            "target": old,
                            "donors": [str(a), str(b), str(c)],
                            "offspring": "",
                            "old_fitness": old_scores["fitness"],
                            "offspring_fitness": None,
                            "kept": False,
                        }
                    )
                    new_pop.append(old)
                    continue
                off_scores = self.evaluated.get(offspring)
                if off_scores is None:
                    off_scores = self.evaluate_prompt(offspring)
                    self.evaluated[offspring] = off_scores
                # greedy pairwise — tie giữ old (khớp max(cur_candidates) với old chèn trước)
                if off_scores["fitness"] > old_scores["fitness"]:
                    selected, selected_scores = offspring, off_scores
                else:
                    selected, selected_scores = old, old_scores
                # p_best update NGAY theo từng offspring, strict >
                if selected_scores["fitness"] > best_score:
                    best_score = selected_scores["fitness"]
                    best_prompt = selected
                new_pop.append(selected)
                gen_log["offspring"].append(
                    {
                        "target": old,
                        "donors": [str(a), str(b), str(c)],
                        "offspring": offspring,
                        "old_fitness": old_scores["fitness"],
                        "offspring_fitness": off_scores["fitness"],
                        "kept": selected == offspring,
                    }
                )
                print(
                    f"[gen {step} pop {j + 1}/{self.N}] old={old_scores['fitness']:.4f} "
                    f"off={off_scores['fitness']:.4f} -> {'giữ offspring' if selected == offspring else 'giữ target'} "
                    f"(best {best_score:.4f})"
                )
            population = new_pop
            avg = sum(self.evaluated[p]["fitness"] for p in population) / len(population)
            gen_log["best_score"] = best_score
            gen_log["avg_score"] = avg
            self.history.append(gen_log)
            self._save_checkpoint(step, population, best_prompt, best_score)
            (self.out_dir / f"step{step}_pop.json").write_text(
                json.dumps(
                    {
                        "population": population,
                        "scores": [self.evaluated[p] for p in population],
                        "best_score": best_score,
                        "avg_score": avg,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[gen {step} done] best={best_score:.4f} avg={avg:.4f} ({time.time() - t0:.0f}s)")
        return self._finish(self.T, population, best_prompt, best_score, t0)

    def _finish(self, step_done: int, population: list[str], best_prompt: str, best_score: float, t0: float) -> dict:
        result = {
            "best_prompt": best_prompt,
            "best_fitness": best_score,
            "steps_done": step_done,
            "final_population": [
                {"prompt": p, **self.evaluated[p]}
                for p in sorted(population, key=lambda p: self.evaluated[p]["fitness"], reverse=True)
            ],
            "elapsed_sec": time.time() - t0,
        }
        (self.out_dir / "best_prompt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[DONE] best fitness={best_score:.4f}\n--- BEST PROMPT ---\n{best_prompt}")
        return result
