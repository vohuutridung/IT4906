"""Hill Climbing: refine one prompt at a time using validation fitness."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from uuid import uuid4


def _propose(client, cfg: dict, prompt: str, metrics: dict, iteration: int) -> str:
    from ..de.prompt_search import extract_prompt

    instruction = f"""You improve prompts for generating concise, accurate Vietnamese chest X-ray findings.
Revise the current prompt into one improved candidate. Keep the task and language appropriate,
make instructions clear and clinically grounded, avoid asking for unsupported details, and do not
include an actual finding or answer. Return only the new prompt between <prompt> and </prompt>.

Current prompt (iteration {iteration}):
{prompt}

Its validation fitness is {metrics['fitness']:.6f} (BERTScore F1 {metrics['bertscore_f1']:.6f};
ROUGE-L F1 {metrics['rougeL_f1']:.6f}). Suggest a focused revision that could improve these
metrics. Do not use test data or invent patient-specific findings."""
    response = client.chat.completions.create(
        model=str(cfg.get("model_id", "Qwen3.5-9B")),
        messages=[{"role": "user", "content": instruction}],
        temperature=float(cfg.get("temperature", 0.5)),
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
        raise ValueError("Hill Climbing chỉ được chọn prompt trên split val.")
    hc_cfg = cfg["hill_climbing"]
    budget = int(hc_cfg.get("max_iterations", 9))
    if isinstance(hc_cfg.get("max_iterations", 9), bool) or budget < 0:
        raise ValueError("hill_climbing.max_iterations không được âm.")
    prompts = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Pool prompt rỗng: {prompt_file}")
    check_metric_dependencies()
    seed = int(cfg.get("seed", 5))
    set_seed(seed)
    initial_index = hc_cfg.get("initial_prompt_index")
    if initial_index is None:
        initial_index = random.Random(seed).randrange(len(prompts))
    initial = prompts[int(initial_index) % len(prompts)]
    target_cfg = resolve_target_config(cfg["target_llm"], cfg.get("evolution_llm"))
    client = optimizer_client(hc_cfg) if budget else None
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
        "data": data, "target": target_result_config(target_cfg),
        "render": PROMPT_RENDER_VERSION,
        "fitness": cfg["fitness"], "pool": prompts, "initial": initial, "budget": budget,
        "optimizer": optimizer_identity(hc_cfg),
        "samples": [(s.idx, s.meta, s.findings_gt) for s in samples],
    }
    run_fp = fingerprint(identity)
    state_path = out_dir / "hill_climbing_state.json"
    snapshot_path = out_dir / "run_config.json"
    run_id = uuid4().hex
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_fp") != run_fp:
            raise ValueError("Checkpoint có cấu hình khác; dùng output folder mới.")
        run_id = state["run_id"]
        if not snapshot_path.is_file():
            raise ValueError("Checkpoint thiếu run_config.json; không thể xác nhận cấu hình của run.")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if (snapshot.get("run_fp"), snapshot.get("run_id")) != (run_fp, run_id):
            raise ValueError("Checkpoint và run_config.json không thuộc cùng một run.")
    else:
        if any((out_dir / name).exists() for name in
               ("run_config.json", "best_prompt.json", "candidate_scores.json")):
            raise ValueError("Output đã có artifact nhưng thiếu checkpoint; chọn --output-dir mới.")
        state = {"run_id": run_id, "run_fp": run_fp, "iteration": 0,
                 "current_prompt": initial, "current_metrics": None,
                 "history": [], "selection_split": "val", "num_samples": len(samples),
                 "max_iterations": budget}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", {"run_id": run_id, "run_fp": run_fp,
        "config": {**cfg, "target_llm": target_cfg}})
    write_json(state_path, state)
    cache_dir = out_dir / "cache" / fingerprint({"target": target_result_config(target_cfg), "data": data})
    target = TargetLLM(target_cfg, str(cache_dir))
    metric = FitnessEvaluator(cfg["fitness"], str(cache_dir))
    refs = [s.findings_gt for s in samples]

    def save_state():
        write_json(state_path, state)
        write_json(out_dir / "candidate_scores.json", {
            "run_id": run_id, "run_fp": run_fp, "selection_split": "val",
            "num_samples": len(samples), "max_iterations": budget, "history": state["history"],
        })
        write_json(out_dir / "best_prompt.json", {
            "run_id": run_id, "run_fp": run_fp, "method": "hill_climbing",
            "best_prompt": state["current_prompt"],
            "best_fitness": state["current_metrics"]["fitness"],
            "selection_split": "val", "num_samples": len(samples),
            "num_candidates_evaluated": len(state["history"]),
        })

    if state["current_metrics"] is None:
        initial_metrics = evaluate_prompt(initial, target, metric, samples, refs, validation_samples, "[HC 0/]")
        state["current_prompt"] = initial
        state["current_metrics"] = initial_metrics
        state["history"].append({"iteration": 0, "prompt": initial, "metrics": initial_metrics,
                                 "accepted": True, "kind": "initial"})
        save_state()

    while state["iteration"] < budget:
        if client is None:
            raise RuntimeError("Hill Climbing cần Evolution API khi max_iterations > 0.")
        step = state["iteration"] + 1
        proposal = _propose(client, hc_cfg, state["current_prompt"], state["current_metrics"], step)
        proposed_metrics = evaluate_prompt(
            proposal, target, metric, samples, refs, validation_samples, f"[HC {step}/{budget}]"
        )
        accepted = proposed_metrics["fitness"] > state["current_metrics"]["fitness"]
        previous = state["current_metrics"]["fitness"]
        if accepted:
            state["current_prompt"] = proposal
            state["current_metrics"] = proposed_metrics
        state["iteration"] = step
        state["history"].append({"iteration": step, "prompt": proposal,
            "metrics": proposed_metrics, "accepted": accepted,
            "fitness_before": previous, "fitness_after": state["current_metrics"]["fitness"]})
        save_state()
        print(f"[HC {step}/{budget}] {'accept' if accepted else 'keep'}; "
              f"proposal={proposed_metrics['fitness']:.4f}; current={state['current_metrics']['fitness']:.4f}", flush=True)

    print(f"[HC] hoàn tất: {budget} lượt cải tiến; p* fitness={state['current_metrics']['fitness']:.4f}")
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
