"""Vòng lặp DE cho EvoPrompt.

- P₀ mặc định lấy prompt do người viết. Các chế độ khác dùng prompt GPT, chọn ngẫu nhiên
  hoặc xếp hạng theo fitness. Donor tốt nhất ban đầu được chọn trong P₀.
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


INITIAL_MODES = ("human", "gpt", "random", "top", "bottom")
INITIAL_MODE_ALIASES = {"randomk": "random", "topk": "top"}


def read_prompt_pool(pool_files: list[str]) -> list[str]:
    """Đọc prompt phân biệt theo thứ tự xuất hiện trong các file."""
    pool: list[str] = []
    for path in pool_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pool.append(line)
    return list(dict.fromkeys(pool))


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
    de_cfg = cfg2.get("de", {})
    de_cfg.pop("cache_dir", None)
    # Randomk cũ chọn best từ cả pool, nên checkpoint cũ khác ngữ nghĩa với random mới.
    if de_cfg.get("initial_mode") == "randomk":
        de_cfg["initial_mode"] = "random"
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
        initial_mode = cfg["de"].get("initial_mode", "human")
        self.initial_mode = INITIAL_MODE_ALIASES.get(initial_mode, initial_mode)
        if self.initial_mode not in INITIAL_MODES:
            raise ValueError(f"initial_mode '{initial_mode}' không hợp lệ. Chọn {', '.join(INITIAL_MODES)}.")
        if self.N < 1 or (self.T > 0 and self.N < 4):
            raise ValueError("population_size phải dương và cần ít nhất 4 prompt khi chạy thế hệ DE.")
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

    def _initialize_population(self) -> list[str]:
        files = self.cfg["de"]["pool_files"]
        if self.initial_mode == "human":
            files = files[:1]
        elif self.initial_mode == "gpt":
            files = files[1:2]

        pool = read_prompt_pool(files)
        if len(pool) < self.N:
            raise ValueError(
                f"initial_mode '{self.initial_mode}' cần {self.N} prompt phân biệt, "
                f"chỉ tìm thấy {len(pool)} trong {files}."
            )

        ranked = self.initial_mode in ("top", "bottom")
        if self.initial_mode == "random":
            candidates = random.Random(int(self.cfg.get("seed", 5))).sample(pool, self.N)
        else:
            candidates = pool if ranked else pool[:self.N]

        print(f"[init] mode={self.initial_mode}, pool={len(pool)}, eval={len(candidates)}")
        for i, prompt in enumerate(candidates, start=1):
            scores = self.evaluate_prompt(prompt)
            self.evaluated[prompt] = scores
            print(f"[pool {i}/{len(candidates)}] fitness={scores['fitness']:.4f} ({prompt[:60]}...)")

        if ranked:
            candidates.sort(key=lambda p: self.evaluated[p]["fitness"], reverse=self.initial_mode == "top")
        return candidates[:self.N]

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
        resumed = self._try_resume()
        if resumed is None:
            population = self._initialize_population()
            best_prompt = max(population, key=lambda p: self.evaluated[p]["fitness"])
            best_score = self.evaluated[best_prompt]["fitness"]
            print(f"[P0] mode={self.initial_mode}, size={len(population)}, best={best_score:.4f}")
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
