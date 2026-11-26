# Random Search (Best-of-N)

## Thiết lập theo notebook it4906-pipeline

- Pool: toàn bộ **10 prompt** trong `data/med_findings_gen/prompts_auto.txt`.
- Target sinh Findings: **google/medgemma-1.5-4b-it local trên GPU**, như notebook gốc.
- Model chấm BERTScore: **xlm-roberta-large**, dùng F1.
- ROUGE-L: F1, không stemmer.
- Fitness chọn prompt: **0.7 × BERTScore F1 + 0.3 × ROUGE-L F1**.
- Validation: **50 mẫu đầu**; test: **toàn bộ 540 mẫu**.
- Corpus BLEU: SacreBLEU, thang 0–100; các điểm F1/Fitness dùng thang 0–1.

## Quy trình

1. MedGemma sinh Findings cho từng prompt trên cùng 50 mẫu validation.
2. Tính fitness của đủ 10 prompt, chọn p* có fitness cao nhất.
3. Nếu hòa điểm, giữ prompt xuất hiện trước trong pool.
4. Khóa p*, chỉ đánh giá prompt này trên toàn bộ test bằng cùng Target LLM.
5. Điền bảng bằng **metric test**: BERTScore, ROUGE-L, Corpus BLEU, Fitness.
   Fitness validation được ghi riêng để giải thích lựa chọn p*.

Không dùng kết quả test để chọn lại prompt.

## Chạy trên Kaggle

Dùng [notebook Kaggle Random Search](https://www.kaggle.com/code/drakan02/it4906-random-search-50-val-and-full-test).
Chọn T4 ×2, bật Internet, và tạo Kaggle Secret `HF_TOKEN` có quyền tải
`google/medgemma-1.5-4b-it` từ Hugging Face sau khi chấp nhận điều khoản.
Notebook dùng `from_pretrained` như pipeline gốc; không cần Add Input model hay
key endpoint Qwen/Evolution LLM.

Bản hai GPU dùng batch 16/GPU, bf16, greedy và `max_new_tokens=128` theo notebook gốc.
Ảnh và metric được xử lý trên Kaggle; source dùng lại loader/Target LLM/fitness/evaluator DE.

## Runner dùng chung

Runner chỉ gồm `run_random_search.py` và `__init__.py`, tái sử dụng code DE.
Nó hỗ trợ `target_llm.provider: local` và `openai` theo file config.
Runner đọc `configs/de.yaml` nếu không truyền `--config`; file đó hiện mặc định
Qwen qua API. Để chạy MedGemma local, truyền một file config có
`target_llm.provider: local` và model MedGemma phù hợp.

```bash
# Đổi đường dẫn thành file config MedGemma local của bạn.
uv run --no-sync python -m IT4906.rs.run_random_search \
  --config /path/to/medgemma.yaml
```

Có thể chạy riêng `--stage validation` hoặc `--stage test`.
Test đọc prompt và cấu hình đã lưu trong run validation.

## Artifact

- `candidate_scores.json`: điểm validation của đủ 10 prompt.
- `best_prompt.json`: p* và fitness validation.
- `run_config.json`: snapshot model/data/metric của run.
- `cache/`: Findings và fitness để tiếp tục khi chạy lại.
- `test_eval/metrics.json`: điểm test, số mẫu và metric config.
- `test_eval/predictions.jsonl`: dự đoán và tham chiếu test.
- Notebook xuất thêm `summary.csv` và ZIP kết quả.

Dùng output riêng cho MedGemma; không trộn cache/score của Qwen API với run này.
