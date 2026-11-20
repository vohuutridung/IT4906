"""Vòng lặp DE cho EvoPrompt.

- P₀ mặc định lấy prompt do người viết. Các chế độ khác dùng prompt GPT, chọn ngẫu nhiên
  hoặc xếp hạng theo fitness. Donor tốt nhất ban đầu được chọn trong P₀.
- Mỗi thế hệ: lần lượt từng prompt làm target; donors a,b = np.random.choice(candidates,
  3, replace=False) (candidates = population trừ target), c = best nếu donor_random=False.
- Offspring chỉ thay target nếu fitness(offspring) > fitness(target) (tie giữ old).
- p_best cập nhật NGAY theo từng offspring, điều kiện strict `>`.
- Checkpoint theo thế hệ, lưu riêng từng phản hồi để resume giữa thế hệ.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from .artifacts import write_json
from .data import DEFAULT_DATASET_ID, Sample, load_split, render_prompt
from .evolution_llm import INSTRUCTION_TEMPLATE, EvolutionLLM, parse_final_prompt
from .fitness import FitnessEvaluator, check_metric_dependencies
from .progress import progress_stage
from .target_llm import TargetLLM, target_result_config


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
    if "target_llm" in cfg2:
        cfg2["target_llm"] = target_result_config(cfg2["target_llm"])
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
        check_metric_dependencies()
        self.donor_random: bool = bool(cfg["de"].get("donor_random", False))

        data_cfg = cfg["data"]
        dataset_dir = data_cfg.get("dataset_dir")
        dataset_id = data_cfg.get("dataset_id", DEFAULT_DATASET_ID)
        split = data_cfg.get("split", "val")
        target_cfg = cfg["target_llm"]
        source = {"parquet": str(Path(dataset_dir).resolve())} if dataset_dir else {"hub": dataset_id}
        cache_identity = {"source": source, "split": split}
        sampling_gpus = target_result_config(target_cfg).get("num_gpus")
        if sampling_gpus is not None:
            cache_identity["sampling_gpus"] = sampling_gpus
        data_key = json.dumps(cache_identity, sort_keys=True)
        data_fingerprint = hashlib.sha1(data_key.encode("utf-8")).hexdigest()[:10]
        cache_dir = str(self.out_dir / "cache" / data_fingerprint)
        num_samples = int(data_cfg.get("num_samples", 200))
        with progress_stage(f"[data] đọc tối đa {num_samples} mẫu, source={dataset_dir or dataset_id} split={split}"):
            self.samples: list[Sample] = load_split(dataset_dir, split, num_samples, dataset_id=dataset_id)
        self.target = TargetLLM(target_cfg, cache_dir)
        self.fitness = FitnessEvaluator(cfg["fitness"], cache_dir)
        evolution_cfg = cfg["evolution_llm"]
        with progress_stage(f"[evolution] khởi tạo provider={evolution_cfg.get('provider', 'openai')}"):
            self.evolution = EvolutionLLM(evolution_cfg)

        self.evaluated: dict[str, dict] = {}  # prompt -> {"fitness", "bertscore_f1", "rougeL_f1"}
        self.history: list[dict] = []
        self.checkpoint_path = self.out_dir / "checkpoint.json"
        self.effective_config = {
            **cfg,
            "evolution_llm": {**evolution_cfg, "model_id": self.evolution.model_id},
        }
        self._run_fp = run_fingerprint(self.effective_config)
        self.run_id = uuid4().hex

    @property
    def audit_dir(self) -> Path:
        return self.out_dir / "evolution" / self.run_id

    # ---------- evaluation ----------
    def evaluate_prompt(
        self, prompt_text: str, samples: list[Sample] | None = None, *, progress: str = "[fitness]"
    ) -> dict:
        """Sinh Findings cho prompt trên toàn bộ dev samples rồi tính fitness (có cache 2 lớp)."""
        samples = samples if samples is not None else self.samples
        pairs = [(render_prompt(prompt_text, s), s) for s in samples]
        cands = self.target.generate_batch(pairs, progress=progress)
        refs = [s.findings_gt for s in samples]
        return self.fitness.score(prompt_text, cands, refs, progress=progress)

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
            progress = f"[pool {i}/{len(candidates)}]"
            scores = self.evaluate_prompt(prompt, progress=progress)
            self.evaluated[prompt] = scores
            print(f"{progress} fitness={scores['fitness']:.4f} ({prompt[:60]}...)", flush=True)

        if ranked:
            candidates.sort(key=lambda p: self.evaluated[p]["fitness"], reverse=self.initial_mode == "top")
        return candidates[:self.N]

    # ---------- checkpoint ----------
    def _save_checkpoint(self, step_done: int, population: list[str], best_prompt: str, best_score: float) -> None:
        rng = list(np.random.get_state())
        rng[1] = rng[1].tolist()
        ckpt = {
            "run_id": self.run_id,
            "run_fp": self._run_fp,
            "step_done": step_done,
            "population": population,
            "best_prompt": best_prompt,
            "best_score": best_score,
            "evaluated_prompts": self.evaluated,
            "history": self.history,
            "numpy_random_state": rng,
        }
        write_json(self.checkpoint_path, ckpt)

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
        rng = ckpt.get("numpy_random_state")
        if rng is not None:
            np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
        if "run_id" in ckpt:
            self.run_id = ckpt["run_id"]
        else:
            # Cho phép tiếp tục checkpoint cũ và gắn kho lưu mới trước lần gọi API tiếp theo.
            self._save_checkpoint(ckpt["step_done"], ckpt["population"], ckpt["best_prompt"], ckpt["best_score"])
        print(
            f"[resume] từ sau thế hệ {ckpt['step_done']} — population {len(ckpt['population'])}, "
            f"best={ckpt['best_score']:.4f}"
        )
        return ckpt["step_done"], ckpt["population"], ckpt["best_prompt"], ckpt["best_score"]

    def _save_run_config(self) -> None:
        snapshot = {"run_id": self.run_id, "run_fp": self._run_fp, "config": self.effective_config}
        write_json(self.out_dir / "run_config.json", snapshot)
        archived_config = self.audit_dir / "run_config.json"
        if not archived_config.exists():
            write_json(archived_config, {**snapshot, "evolution_instruction": INSTRUCTION_TEMPLATE})

    def _proposal(self, path: Path, target: str, donors: list[str], progress: str) -> dict:
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["target"] != target or record["donors"] != donors:
                raise RuntimeError(f"Donor hoặc target không khớp với phản hồi đã lưu trong {path}.")
            print(f"{progress} dùng lại phản hồi đã lưu", flush=True)
            return record
        with progress_stage(f"{progress} sinh prompt con bằng Evolution LLM"):
            response = self.evolution.generate_response(target, *donors)
            record = {"target": target, "donors": donors, "raw_response": response}
            write_json(path, record)
        return record

    # ---------- main loop ----------
    def run(self) -> dict:
        t0 = time.time()
        print(f"[init] dev samples: {len(self.samples)}")
        resumed = self._try_resume()
        self._save_run_config()
        if resumed is None:
            population = self._initialize_population()
            best_prompt = max(population, key=lambda p: self.evaluated[p]["fitness"])
            best_score = self.evaluated[best_prompt]["fitness"]
            print(f"[P0] mode={self.initial_mode}, size={len(population)}, best={best_score:.4f}")
            step_done = 0  # lần đầu chạy: chưa có thế hệ nào hoàn thành
            write_json(self.audit_dir / "initial_population.json", {
                "population": population, "evaluated_prompts": self.evaluated,
            })
            self._save_checkpoint(0, population, best_prompt, best_score)
        else:
            step_done, population, best_prompt, best_score = resumed
            if step_done >= self.T:
                return self._finish(step_done, population, best_prompt, best_score, t0)

        for step in range(step_done + 1, self.T + 1):
            generation_start = time.monotonic()
            print(f"[gen {step}/{self.T}] bắt đầu, {self.N} ứng viên", flush=True)
            gen_log: dict = {"step": step, "offspring": []}
            new_pop: list[str] = []
            for j in range(self.N):
                progress = f"[gen {step}/{self.T} pop {j + 1}/{self.N}]"
                old = population[j]
                old_scores = self.evaluated[old]
                # donors: khớp code gốc — choice 3 không thay thế từ population trừ target,
                # rồi c = best (donor_random=False)
                candidates = [population[k] for k in range(self.N) if k != j]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                if not self.donor_random:
                    c = best_prompt
                path = self.audit_dir / f"gen_{step:03d}" / f"slot_{j + 1:03d}.json"
                record = self._proposal(path, old, [str(a), str(b), str(c)], progress)
                if "offspring_scores" not in record:
                    offspring = parse_final_prompt(record["raw_response"])
                    off_scores = None
                    if offspring.strip():
                        off_scores = self.evaluated.get(offspring)
                        if off_scores is None:
                            off_scores = self.evaluate_prompt(offspring, progress=progress)
                        else:
                            print(f"{progress} dùng lại fitness của prompt trùng", flush=True)
                    kept = off_scores is not None and off_scores["fitness"] > old_scores["fitness"]
                    record.update(
                        offspring=offspring, old_scores=old_scores, offspring_scores=off_scores,
                        kept=kept, selected=offspring if kept else old,
                    )
                    write_json(path, record)
                else:
                    print(f"{progress} dùng lại quyết định và điểm đã lưu", flush=True)

                offspring, off_scores = record["offspring"], record["offspring_scores"]
                if off_scores is not None:
                    self.evaluated[offspring] = off_scores
                selected = record["selected"]
                selected_scores = self.evaluated[selected]
                # p_best update NGAY theo từng offspring, strict >
                if selected_scores["fitness"] > best_score:
                    best_score = selected_scores["fitness"]
                    best_prompt = selected
                new_pop.append(selected)
                gen_log["offspring"].append(
                    {
                        "target": old,
                        "donors": record["donors"],
                        "offspring": offspring,
                        "old_fitness": old_scores["fitness"],
                        "offspring_fitness": off_scores["fitness"] if off_scores is not None else None,
                        "kept": record["kept"],
                        "record_path": str(path.relative_to(self.out_dir)),
                    }
                )
                off_display = f"{off_scores['fitness']:.4f}" if off_scores is not None else "rỗng"
                decision = "giữ offspring" if record["kept"] else "giữ target"
                print(f"{progress} old={old_scores['fitness']:.4f} "
                      f"off={off_display} -> {decision} (best {best_score:.4f})")
            population = new_pop
            avg = sum(self.evaluated[p]["fitness"] for p in population) / len(population)
            gen_log["best_score"] = best_score
            gen_log["avg_score"] = avg
            self.history.append(gen_log)
            summary = {
                "population": population,
                "scores": [self.evaluated[p] for p in population],
                "best_score": best_score,
                "avg_score": avg,
            }
            with progress_stage(f"[gen {step}/{self.T}] lưu quần thể và checkpoint"):
                write_json(self.out_dir / f"step{step}_pop.json", summary)
                write_json(self.audit_dir / f"gen_{step:03d}" / "population.json", summary)
                self._save_checkpoint(step, population, best_prompt, best_score)
            print(f"[gen {step}/{self.T} done] best={best_score:.4f} avg={avg:.4f} "
                  f"({time.monotonic() - generation_start:.0f}s)", flush=True)
        return self._finish(self.T, population, best_prompt, best_score, t0)

    def _finish(self, step_done: int, population: list[str], best_prompt: str, best_score: float, t0: float) -> dict:
        result = {
            "run_id": self.run_id,
            "run_fp": self._run_fp,
            "best_prompt": best_prompt,
            "best_fitness": best_score,
            "steps_done": step_done,
            "final_population": [
                {"prompt": p, **self.evaluated[p]}
                for p in sorted(population, key=lambda p: self.evaluated[p]["fitness"], reverse=True)
            ],
            "elapsed_sec": time.time() - t0,
        }
        write_json(self.audit_dir / "best_prompt.json", result)
        write_json(self.out_dir / "best_prompt.json", result)
        print(f"[DONE] best fitness={best_score:.4f}\n--- BEST PROMPT ---\n{best_prompt}")
        return result
