"""Chấm toàn bộ prompt_auto trên val, chọn p*, rồi đánh giá trên test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from itertools import islice
from uuid import uuid4


def select_prompt(cfg: dict, prompt_file: Path, out_dir: Path) -> dict:
    from ..de.artifacts import write_json
    from ..de.data import DEFAULT_DATASET_ID, PROMPT_RENDER_VERSION, Sample, iter_split, render_prompt
    from ..de.fitness import FitnessEvaluator, check_metric_dependencies
    from ..de.progress import progress_stage
    from ..de.target_llm import TargetLLM, resolve_target_config, target_result_config

    if cfg['data'].get('split', 'val') != 'val':
        raise ValueError('Chỉ được chọn prompt trên split val.')
    if cfg['target_llm'].get('provider', 'local') not in ('openai', 'local'):
        raise ValueError('target_llm.provider phải là openai hoặc local.')
    prompts = list(dict.fromkeys(
        line.strip() for line in prompt_file.read_text(encoding='utf-8').splitlines() if line.strip()
    ))
    if not prompts:
        raise ValueError(f'Pool prompt rỗng: {prompt_file}')
    check_metric_dependencies()
    data = cfg['data']
    num_samples = data.get('num_samples', 200)
    if num_samples is not None and (
        isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1
    ):
        raise ValueError('data.num_samples phải dương hoặc null để dùng toàn bộ val.')
    target_cfg = resolve_target_config(cfg['target_llm'], cfg.get('evolution_llm'))
    def validation_samples():
        return iter_split(data.get('dataset_dir'), 'val', num_samples,
                          dataset_id=data.get('dataset_id', DEFAULT_DATASET_ID))

    # Chỉ giữ metadata/reference; ảnh được đọc theo batch khi sinh Findings.
    samples = []
    for sample in validation_samples():
        samples.append(Sample(idx=sample.idx, meta=sample.meta, image=None,
                              findings_gt=sample.findings_gt))
        sample.image.close()
    if not samples:
        raise ValueError('Không có mẫu validation.')
    identity = {
        'data': data, 'target': target_result_config(target_cfg),
        'render': PROMPT_RENDER_VERSION,
        'samples': [(s.idx, s.meta, s.findings_gt) for s in samples],
    }
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    # Cache tách theo dữ liệu/target; đổi metric có thể dùng lại Findings đã sinh.
    cache_dir = out_dir / 'cache' / fingerprint(identity)
    run_fp = fingerprint({'identity': identity, 'fitness': cfg['fitness'], 'prompts': prompts,
                          'seed': cfg.get('seed', 5)})
    snapshot_path = out_dir / 'run_config.json'
    run_id = uuid4().hex
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        if snapshot.get('run_fp') != run_fp:
            raise ValueError('Output đã có run khác cấu hình/pool/dữ liệu; chọn --output-dir mới.')
        run_id = snapshot['run_id']
    elif any((out_dir / name).exists() for name in ('best_prompt.json', 'candidate_scores.json')):
        raise ValueError('Output cũ thiếu run_config.json; chọn --output-dir mới.')
    write_json(snapshot_path, {
        'run_id': run_id, 'run_fp': run_fp,
        'config': {**cfg, 'target_llm': target_cfg},
    })
    target = TargetLLM(target_cfg, str(cache_dir))
    metric = FitnessEvaluator(cfg['fitness'], str(cache_dir))
    refs = [s.findings_gt for s in samples]
    scores = []
    for index, prompt in enumerate(prompts, 1):
        progress = f'[RS {index}/{len(prompts)}]'
        predictions = []
        stream = validation_samples()
        with progress_stage(f'{progress} sinh Findings trên {len(samples)} mẫu (batch nhỏ, có cache)'):
            while batch := list(islice(stream, target.parallel_batch_size)):
                try:
                    offset = len(predictions)
                    for expected, actual in zip(samples[offset:offset + len(batch)], batch, strict=True):
                        if (expected.idx, expected.meta, expected.findings_gt) != (actual.idx, actual.meta, actual.findings_gt):
                            raise ValueError('Dữ liệu validation thay đổi giữa các lượt đọc.')
                    pairs = [(render_prompt(prompt, s), s) for s in batch]
                    predictions.extend(target.generate_batch(pairs))
                finally:
                    for sample in batch:
                        sample.image.close()
                if len(predictions) % 20 == 0 or len(predictions) == len(samples):
                    print(f'{progress} Findings {len(predictions)}/{len(samples)} mẫu', flush=True)
        if len(predictions) != len(refs):
            raise ValueError('Số dự đoán không khớp số tham chiếu validation.')
        scores.append({'candidate_id': index, 'prompt': prompt,
                       **metric.score(prompt, predictions, refs, progress=progress)})
        write_json(out_dir / 'candidate_scores.json', {
            'run_id': run_id, 'run_fp': run_fp, 'selection_split': 'val',
            'num_samples': len(samples), 'num_candidates': len(prompts), 'candidates': scores,
        })
    # Hòa điểm: giữ prompt xuất hiện trước trong file.
    best = max(scores, key=lambda row: row['fitness'])
    result = {
        'run_id': run_id, 'run_fp': run_fp, 'method': 'random_search_best_of_n',
        'best_prompt': best['prompt'], 'best_fitness': best['fitness'],
        'selection_split': 'val', 'num_samples': len(samples), 'num_candidates': len(prompts),
    }
    write_json(out_dir / 'best_prompt.json', result)
    print(f"[RS] Đã chọn p*: fitness={best['fitness']:.4f}; {len(prompts)} prompt × {len(samples)} mẫu")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/de.yaml'))
    parser.add_argument('--prompt-file', type=Path, default=Path('data/med_findings_gen/prompts_auto.txt'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/random_search'))
    parser.add_argument('--data-root', help='Thư mục Parquet local; mặc định tự dùng dữ liệu đã tải trong repo.')
    parser.add_argument('--val-num-samples', type=int, help='Ghi đè số mẫu validation trong config.')
    parser.add_argument('--stage', choices=('all', 'validation', 'test'), default='all')
    args = parser.parse_args()
    if args.val_num_samples is not None and args.val_num_samples < 1:
        parser.error('--val-num-samples phải dương.')

    from dotenv import load_dotenv
    load_dotenv(override=False)
    if args.stage != 'test':
        import yaml
        from ..de.run_de import set_seed
        cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if args.data_root:
            cfg['data']['dataset_dir'] = args.data_root
        elif not cfg['data'].get('dataset_dir'):
            local_data = Path('data/med_findings_gen/VNMedical_bv175')
            if any(local_data.glob('data/val-*.parquet')):
                cfg['data']['dataset_dir'] = str(local_data)
        if args.val_num_samples is not None:
            cfg['data']['num_samples'] = args.val_num_samples
        set_seed(int(cfg.get('seed', 5)))
        select_prompt(cfg, args.prompt_file, args.output_dir)
    if args.stage != 'validation':
        # select_prompt đã kết thúc; nhả bộ nhớ validation trước khi nạp evaluator test.
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from ..de.evaluate_de import evaluate
        evaluate(args.output_dir, args.config, args.data_root, None)


if __name__ == '__main__':
    main()
