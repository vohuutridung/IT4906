"""Entrypoint: uv run python -m IT4906.de.run_de --config configs/de.yaml [--smoke]

--smoke: tối đa 5 sample, quần thể 4 prompt, 1 thế hệ.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml

from .de_loop import DERunner
from .evolution_llm import DEFAULT_EVOLUTION_MODELS


def load_env() -> None:
    """Nạp biến môi trường từ file `.env` ở thư mục chạy (nếu có python-dotenv).

    base_url + api_key của Evolution LLM KHÔNG hardcode vào code/repo (chỉ dẫn
    Dũng 2026-11-13) — điền vào `.env` (đã git-ignore); thiếu dotenv thì biến
    môi trường vẫn đọc được nếu đã export thủ công."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[env] chưa cài python-dotenv — bỏ qua .env, dùng biến môi trường có sẵn")
        return
    loaded = load_dotenv()
    if loaded:
        print("[env] đã nạp .env")


def set_seed(seed: int) -> None:
    """Khớp `utils.set_seed` code gốc: seed toàn cục random + numpy + torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoPrompt DE — Medical Findings Generation")
    parser.add_argument("--config", default="configs/de.yaml")
    parser.add_argument("--smoke", action="store_true", help="5 sample × 1 thế hệ")
    parser.add_argument(
        "--evolution-provider", choices=tuple(DEFAULT_EVOLUTION_MODELS),
        help="chọn provider của Evolution LLM",
    )
    parser.add_argument(
        "--evolution-model-id", metavar="ID", help="ghi đè model_id của Evolution LLM",
    )
    parser.add_argument(
        "--set", nargs=2, action="append", metavar=("KEY", "VALUE"),
        help="ghi đè config, VD: --set data.num_samples 50",
    )
    args = parser.parse_args()
    if args.evolution_model_id is not None and not args.evolution_model_id.strip():
        parser.error("--evolution-model-id cần một ID model.")

    load_env()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in args.set or []:
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(value)  # cast kiểu theo YAML

    if args.evolution_provider is not None:
        cfg["evolution_llm"]["provider"] = args.evolution_provider
    if args.evolution_model_id is not None:
        cfg["evolution_llm"]["model_id"] = args.evolution_model_id

    if args.smoke:
        cfg["data"]["num_samples"] = min(5, int(cfg["data"].get("num_samples", 200)))
        cfg["de"]["iterations"] = 1
        # N phải >= 4: donors cần 3 prompt khác target (code gốc choice 3, replace=False)
        cfg["de"]["population_size"] = 4

    set_seed(int(cfg.get("seed", 5)))
    out_dir = cfg["de"].get("cache_dir", "runs/de")
    if args.smoke:
        out_dir = out_dir.rstrip("/") + "_smoke"

    runner = DERunner(cfg, out_dir)
    runner.run()


if __name__ == "__main__":
    main()
