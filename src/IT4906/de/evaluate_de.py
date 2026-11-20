"""Evaluate a frozen EvoPrompt-DE result on the held-out test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import yaml
from sacrebleu.metrics import BLEU

from .artifacts import write_json, write_text_atomic
from .data import DEFAULT_DATASET_ID, iter_split, render_prompt
from .fitness import FitnessEvaluator
from .run_de import load_env, set_seed
from .target_llm import TargetLLM


def evaluate(run_dir: Path, config_path: Path, data_root: str | None, limit: int | None) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("--num-samples phải lớn hơn 0.")

    best = json.loads((run_dir / "best_prompt.json").read_text(encoding="utf-8"))
    prompt = best["best_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("best_prompt.json không chứa prompt hợp lệ.")
    if "run_id" in best:
        snapshot = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if any(snapshot[key] != best[key] for key in ("run_id", "run_fp")):
            raise ValueError("best_prompt.json và run_config.json thuộc hai run khác nhau.")
        config = snapshot["config"]
    else:
        print(f"[test] Run cũ chưa có snapshot, dùng cấu hình {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data = config["data"]
    selection_split = data.get("split", "val")
    if selection_split == "test":
        raise ValueError("Prompt đã được chọn trên test, không thể báo cáo test độc lập.")
    dataset_dir = data_root or data.get("dataset_dir")
    dataset_id = data.get("dataset_id", DEFAULT_DATASET_ID)
    eval_dir = run_dir / ("test_eval" if limit is None else f"test_eval_first_{limit}")
    set_seed(int(config.get("seed", 5)))

    # Tách cache test khỏi các mẫu validation và khỏi các cấu hình target khác.
    cache_key = json.dumps([dataset_id, dataset_dir, config["target_llm"]], sort_keys=True)
    cache_path = eval_dir / "cache" / hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    target = TargetLLM(config["target_llm"], str(cache_path))
    metric = FitnessEvaluator(config["fitness"], str(cache_path))
    samples = iter_split(dataset_dir, "test", limit, dataset_id=dataset_id)
    rows = []
    print(f"[test] source={dataset_dir or dataset_id} split=test target={target.model_id}")
    while batch := list(islice(samples, target.batch_size)):
        pairs = [(render_prompt(prompt, sample), sample) for sample in batch]
        predictions = target.generate_batch(pairs)
        rows.extend(
            {"sample_idx": sample.idx, "prediction": prediction, "reference": sample.findings_gt}
            for sample, prediction in zip(batch, predictions, strict=True)
        )
        print(f"[test] {len(rows)} mẫu")
    if not rows:
        raise ValueError("Không tìm thấy mẫu trong split test.")

    write_text_atomic(eval_dir / "predictions.jsonl", "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    cands = [row["prediction"] for row in rows]
    refs = [row["reference"] for row in rows]
    scores = metric.score(prompt, cands, refs)
    bleu = BLEU()
    bleu_score = bleu.corpus_score(cands, [refs]).score
    report = {
        "run_id": best.get("run_id"),
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
        "split": "test",
        "num_samples": len(rows),
        "sample_limit": limit,
        "best_prompt": prompt,
        "selection_split": selection_split,
        "selection_fitness": best.get("best_fitness"),
        "target_llm": config["target_llm"],
        "fitness_config": config["fitness"],
        "metrics": {**scores, "bleu": bleu_score},
        "bleu_signature": str(bleu.get_signature()),
    }
    write_json(eval_dir / "metrics.json", report)
    print(f"[test] metrics={report['metrics']} → {eval_dir / 'metrics.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/de.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()
    load_env()
    evaluate(args.run_dir, args.config, args.data_root, args.num_samples)


if __name__ == "__main__":
    main()
