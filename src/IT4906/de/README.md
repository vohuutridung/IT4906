# Module DE — EvoPrompt (Differential Evolution với LLM) cho Medical Findings Generation

Đặc tả: `context/docs/de_spec.md` (đã đối chiếu code gốc `beeevita/EvoPrompt` ngày 2026-11-10).
Kế hoạch chạy: `context/docs/de_execution_plan.md` (Kaggle/Colab, checkpoint + cache).
👉 **Cài đặt + chạy + bàn giao: `HANDOVER.md` (cùng thư mục).**

## Cấu trúc
| File | Vai trò |
|---|---|
| `data.py` | Load split `val` (parquet), lấy N sample đầu, render prompt (substitute placeholder metadata) |
| `target_llm.py` | Wrapper `medgemma-1.5-4b-it` (batch, max_new_tokens 128, cache generation theo (prompt, sample)) |
| `fitness.py` | `0.7 × BERTScore(xlm-roberta-large, F1) + 0.3 × ROUGE-L(F1)` — model/variant qua config, có cache |
| `evolution_llm.py` | Wrapper `medgemma-27b-text-it` (4-bit), instruction 4 bước EvoPrompt, parse `<prompt>...</prompt>` |
| `de_loop.py` | Khởi tạo P₀ theo `initial_mode`, chạy T thế hệ với greedy pairwise (`>`), checkpoint/resume |
| `run_de.py` | Entrypoint: `uv run python -m IT4906.de.run_de --config configs/de.yaml` |

## Dependencies (chưa thêm vào pyproject — chờ xác nhận vì file dùng chung với nhóm GA)
`torch`, `transformers`, `accelerate`, `bitsandbytes`, `datasets`, `pillow`,
`bert_score`, `rouge-score`, `pyyaml`.
Đã thêm vào pyproject: `openai`, `python-dotenv` (gọn, không đụng GPU).

## Evolution LLM — 2 provider (`configs/de.yaml` → `evolution_llm.provider`)
- `openai` (mặc định, chỉ dẫn Dũng 2026-11-13): gọi endpoint OpenAI-compatible:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="...", api_key="...")
  ```
  base_url + api_key ĐỌC TỪ BIẾN MÔI TRƯỜNG — copy `.env.example` thành `.env`,
  điền giá trị thật (`.env` git-ignore, không commit). Được `run_de.py` nạp tự động.
- `local`: load `medgemma-27b-text-it` bằng transformers 4-bit trên Kaggle 2×T4 (path cũ).

## Chạy trên Kaggle

Dùng `deployment/kaggle/run.sh` với các chế độ `setup`, `smoke` và `evolve`. Các cell mẫu, đường dẫn dataset và cách nạp Secrets có trong mục [Chạy trên Kaggle](HANDOVER.md#chạy-trên-kaggle).
