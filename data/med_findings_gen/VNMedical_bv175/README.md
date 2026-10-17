---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: findings
    dtype: string
  - name: impressions
    dtype: string
  - name: Ages
    dtype: string
  - name: view
    dtype: string
  - name: Ghi chú
    dtype: string
  - name: Kỹ thuật
    dtype: string
  - name: sex
    dtype: string
  splits:
  - name: train
    num_bytes: 13892105733.76
    num_examples: 4320
  - name: val
    num_bytes: 1710561765
    num_examples: 540
  - name: test
    num_bytes: 1732489868
    num_examples: 540
  download_size: 17342416928
  dataset_size: 17335157366.760002
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- vi
---

# ViX-Ray: A Vietnamese Chest X-Ray Dataset for Vision-Language Models

ViX-Ray is the **first publicly available Vietnamese chest X-ray dataset** designed for vision-language model (VLM) research in medical AI. It pairs radiographic images from Vietnamese patients with expert-written clinical annotations in Vietnamese, directly addressing the lack of Vietnamese medical data in existing VLMs.

- **License**: CC BY-NC-SA 4.0
- **Paper**: [arXiv:2603.15513](https://arxiv.org/abs/2603.15513)

---

## Dataset Overview

Each sample contains a PA-view chest X-ray image paired with Vietnamese-language radiology reports written by certified radiologists. All Protected Health Information (PHI) has been removed; clinically relevant metadata (age, gender) is retained. The dataset was collected under Institutional Review Board (IRB) approval.

### Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 4,320   | 80%        |
| Val   | 540     | 10%        |
| Test  | 540     | 10%        |
| **Total** | **5,400** | 100% |

### Fields

| Field | Description |
|-------|-------------|
| `image` | PA-view chest X-ray image |
| `findings` | Detailed descriptive narrative of anatomical/pathological observations (Vietnamese) |
| `impressions` | Clinical diagnostic conclusions (Vietnamese) |
| `Ages` | Patient age in years |
| `sex` | Patient gender |
| `view` | Imaging perspective (PA — Posterior Anterior) |
| `Ghi chú` | Procedure description (Vietnamese) |
| `Kỹ thuật` | Technical details of the radiograph |

### Report Text Length (tokens)

| Split | Findings (avg) | Impressions (avg) |
|-------|---------------|-------------------|
| Train | 46            | 12                |
| Val   | 45            | 14                |
| Test  | 46            | 13                |

Average patient age: ~69–70 years across all splits.

---

## Common Findings

**Most frequently mentioned anatomical structures**: Heart (tim), Lungs (phổi), Ribs (xương sườn), Diaphragm (vòm hoành), Pulmonary hilum (rốn phổi)

**Most common diagnoses**: Interstitial lung disease, Cardiomegaly, Pulmonary infiltrates/consolidations, Rib fractures, Aortic calcifications

---

## Intended Use

- Training and fine-tuning vision-language models for Vietnamese medical image analysis
- Benchmarking multilingual and Vietnamese-specific VLMs
- Research on hallucination reduction in clinical AI
- Multi-turn clinical diagnostic conversation modeling

---

## Benchmark Results

Five open-source VLMs were fine-tuned and evaluated on three stages: findings generation → impression generation → multi-turn clinical dialogue. Best performing model: **Qwen2.5-VL-7B** (ROUGE-1: 84.30% on findings, 95.20% on multi-turn impressions).

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{nguyen2026vix,
  title={ViX-Ray: A Vietnamese Chest X-Ray Dataset for Vision-Language Models},
  author={Nguyen, Duy Vu Minh and Truong, Chinh Thanh and Tran, Phuc Hoang and Le, Hung Tuan and Dat, Nguyen Van-Thanh and Pham, Trung Hieu and Van Nguyen, Kiet},
  journal={arXiv preprint arXiv:2603.15513},
  year={2026}
}
```