"""APE baseline: paraphrase an elite prompt population and retain the fittest."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from uuid import uuid4


def _paraphrase(client, cfg: dict, prompt: str, generation: int, parent_id: str) -> str:
    from ..de.prompt_search import extract_prompt

    instruction = f"""You are an Automatic Prompt Engineer. Paraphrase the current prompt into a new
candidate prompt for generating concise, accurate Vietnamese chest X-ray findings.
Preserve its task and useful constraints while expressing the instructions differently.
Do not answer the prompt, add patient-specific findings, or introduce unsupported clinical claims.
Return only the rewritten prompt inside <prompt> and </prompt>.

Generation: {generation}; parent: {parent_id}
Current prompt:
{prompt}"""
    response = client.chat.completions.create(
        model=str(cfg.get("model_id", "Qwen3.5-9B")),
        messages=[{"role": "user", "content": instruction}],
        temperature=float(cfg.get("temperature", 0.7)),
        max_tokens=int(cfg.get("max_tokens", 1000)),
    )
    return extract_prompt(response.choices[0].message.content or "")


def optimize(cfg: dict, prompt_file: Path, out_dir: Path) -> dict:
    from ..de.artifacts import write_json
    from ..de.data import DEFAULT_DATASET_ID, PROMPT_RENDER_VERSION, Sample, iter_split
    from ..de.fitness import FitnessEvaluator, check_metric_dependencies
    from ..de.prompt_search import evaluate_prompt, fingerprint, optimizer_client, optimizer_identity
    from ..de.run_de import set_seed
    from ..de.target_llm import TargetLLM, resolve_target_config, target_result_config

    data = cfg["data"]
    if data.get("split", "val") != "val":
        raise ValueError("APE chỉ được chọn prompt trên split val.")
    ape_cfg = cfg["ape"]
    population_size = int(ape_cfg.get("population_size", 10))
    generations = int(ape_cfg.get("generations", 2))
    if (isinstance(ape_cfg.get("population_size", 10), bool)
            or isinstance(ape_cfg.get("generations", 2), bool)
            or population_size < 1 or generations < 0):
        raise ValueError("population_size phải dương và generations không được âm.")
    prompts = list(dict.fromkeys(x.strip() for x in prompt_file.read_text(encoding="utf-8").splitlines() if x.strip()))
    if len(prompts) != population_size:
        raise ValueError(f"APE cần đúng {population_size} prompt phân biệt; nhận được {len(prompts)}.")
    check_metric_dependencies()
    set_seed(int(cfg.get("seed", 5)))
    target_cfg = resolve_target_config(cfg["target_llm"], cfg.get("evolution_llm"))
    client = optimizer_client(ape_cfg) if generations else None
    num_samples = data.get("num_samples", 50)
    if num_samples is not None and (
        isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1
    ):
        raise ValueError("data.num_samples phải dương hoặc null để dùng toàn bộ val.")

    def validation_samples():
        return iter_split(data.get("dataset_dir"), "val", num_samples,
                          dataset_id=data.get("dataset_id", DEFAULT_DATASET_ID))

    samples = []
    for sample in validation_samples():
        samples.append(Sample(idx=sample.idx, meta=sample.meta, image=None, findings_gt=sample.findings_gt))
        sample.image.close()
    if not samples:
        raise ValueError("Không có mẫu validation.")
    identity = {
        "data": data, "target": target_result_config(target_cfg), "fitness": cfg["fitness"],
        "render": PROMPT_RENDER_VERSION,
        "prompts": prompts, "population_size": population_size, "generations": generations,
        "optimizer": optimizer_identity(ape_cfg),
        "samples": [(s.idx, s.meta, s.findings_gt) for s in samples],
    }
    run_fp = fingerprint(identity)
    state_path = out_dir / "ape_state.json"
    snapshot_path = out_dir / "run_config.json"
    run_id = uuid4().hex
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_fp") != run_fp:
            raise ValueError("APE checkpoint có cấu hình khác; dùng output folder mới.")
        run_id = state["run_id"]
        if not snapshot_path.is_file():
            raise ValueError("APE checkpoint thiếu run_config.json; không thể xác nhận cấu hình của run.")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if (snapshot.get("run_fp"), snapshot.get("run_id")) != (run_fp, run_id):
            raise ValueError("APE checkpoint và run_config.json không thuộc cùng một run.")
    else:
        if any((out_dir / name).exists() for name in
               ("run_config.json", "best_prompt.json", "candidate_scores.json")):
            raise ValueError("APE output đã có artifact nhưng thiếu checkpoint; dùng output folder mới.")
        state = {"run_id": run_id, "run_fp": run_fp, "completed_generations": 0,
                 "population": [], "evaluated": [], "pending_offspring": [], "history": [],
                 "selection_split": "val", "num_samples": len(samples),
                 "population_size": population_size, "generations_budget": generations}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", {"run_id": run_id, "run_fp": run_fp,
        "config": {**cfg, "target_llm": target_cfg}})
    write_json(state_path, state)
    cache_dir = out_dir / "cache" / fingerprint({"target": target_result_config(target_cfg), "data": data})
    target = TargetLLM(target_cfg, str(cache_dir))
    metric = FitnessEvaluator(cfg["fitness"], str(cache_dir))
    refs = [s.findings_gt for s in samples]

    def write_state() -> None:
        write_json(state_path, state)
        write_json(out_dir / "candidate_scores.json", {
            "run_id": run_id, "run_fp": run_fp, "selection_split": "val",
            "num_samples": len(samples), "population_size": population_size,
            "generations": generations, "total_candidates_evaluated": len(state["evaluated"]),
            "candidates": state["evaluated"], "generation_history": state["history"],
        })
        if state["population"]:
            elite = state["population"][0]
            write_json(out_dir / "best_prompt.json", {
                "run_id": run_id, "run_fp": run_fp, "method": "ape_automatic_prompt_engineer",
                "best_prompt": elite["prompt"], "best_fitness": elite["metrics"]["fitness"],
                "selection_split": "val", "num_samples": len(samples),
                "num_candidates_evaluated": len(state["evaluated"]),
                "population_size": population_size, "generations": state["completed_generations"],
            })
            write_json(out_dir / "population.json", {
                "population": state["population"], "selection_split": "val",
                "completed_generations": state["completed_generations"],
            })

    # Score all P0 prompts. Save after each candidate so an interrupted run resumes safely.
    initial_by_id = {row["candidate_id"]: row for row in state["evaluated"]}
    for i, prompt in enumerate(prompts, 1):
        candidate_id = f"P0-{i:02d}"
        if candidate_id in initial_by_id:
            continue
        metrics = evaluate_prompt(prompt, target, metric, samples, refs, validation_samples,
                                  f"[APE P0 {i}/{population_size}]")
        row = {"candidate_id": candidate_id, "generation": 0, "parent_id": None,
               "prompt": prompt, "metrics": metrics, "origin": "initial"}
        state["evaluated"].append(row)
        state["population"].append(row)
        write_state()
    state["population"] = sorted(state["population"], key=lambda row: -row["metrics"]["fitness"])
    write_state()

    while state["completed_generations"] < generations:
        generation = state["completed_generations"] + 1
        parents = sorted(state["population"], key=lambda row: -row["metrics"]["fitness"])[:population_size]
        done_parent_ids = {row["parent_id"] for row in state["pending_offspring"]}
        if client is None:
            raise RuntimeError("APE cần Evolution API khi generations > 0.")
        for j, parent in enumerate(parents, 1):
            if parent["candidate_id"] in done_parent_ids:
                continue
            prompt = _paraphrase(client, ape_cfg, parent["prompt"], generation, parent["candidate_id"])
            metrics = evaluate_prompt(prompt, target, metric, samples, refs, validation_samples,
                                      f"[APE G{generation} {j}/{population_size}]")
            child = {"candidate_id": f"G{generation}-O{j:02d}", "generation": generation,
                     "parent_id": parent["candidate_id"], "prompt": prompt,
                     "metrics": metrics, "origin": "paraphrase"}
            state["pending_offspring"].append(child)
            state["evaluated"].append(child)
            write_state()
            print(f"[APE G{generation} {j}/{population_size}] fitness={metrics['fitness']:.4f}", flush=True)

        # Stable descending sort: existing elites precede offspring in ties.
        pool = state["population"] + state["pending_offspring"]
        state["population"] = sorted(pool, key=lambda row: -row["metrics"]["fitness"])[:population_size]
        state["completed_generations"] = generation
        elite_scores = [row["metrics"]["fitness"] for row in state["population"]]
        state["history"].append({"generation": generation, "parent_ids": [r["candidate_id"] for r in parents],
            "offspring_ids": [r["candidate_id"] for r in state["pending_offspring"]],
            "best_fitness": elite_scores[0], "population_fitness": elite_scores})
        state["pending_offspring"] = []
        write_state()
        print(f"[APE] generation {generation}/{generations}; best={elite_scores[0]:.4f}; "
              f"elite range={elite_scores[-1]:.4f}–{elite_scores[0]:.4f}", flush=True)

    print(f"[APE] hoàn tất: {population_size} elite, {generations} thế hệ; "
          f"p* fitness={state['population'][0]['metrics']['fitness']:.4f}")
    return json.loads((out_dir / "best_prompt.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=False)
    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.data_root:
        cfg["data"]["dataset_dir"] = args.data_root
    if args.stage == "validation":
        optimize(cfg, args.prompt_file, args.output_dir)
    else:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from ..de.evaluate_de import evaluate
        evaluate(args.output_dir, args.config, args.data_root, None)


if __name__ == "__main__":
    main()
