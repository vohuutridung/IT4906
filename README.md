# IT4906 — Phân tích dữ liệu X-quang ngực tiếng Việt

HF repo: [vohuutridung/IT4906-TTTH](https://huggingface.co/datasets/vohuutridung/IT4906-TTTH)

Project chứa prompt tạo báo cáo `Findings`, script tải dataset VNMedical_bv175, notebook phân tích khám phá dữ liệu và các baseline tối ưu prompt.

## Kaggle notebooks

| Phương pháp | Notebook Kaggle | Mô tả |
|---|---|---|
| DE — EvoPrompt | [it4906-pipeline](https://www.kaggle.com/code/vohuutridung/it4906-pipeline) | Pipeline DE |
| Random Search (Best-of-10) | [IT4906 Random Search](https://www.kaggle.com/code/drakan02/it4906-random-search-50-val-and-full-test) | Chấm 10 prompt trên 50 mẫu validation, chọn p*, đánh giá trên toàn bộ test |
| Hill Climbing | [IT4906 Hill Climbing](https://www.kaggle.com/code/drakan02/it4906-hill-climbing-50-val-and-full-test) | 1 prompt khởi tạo + 9 lượt cải tiến, cùng ngân sách 10 lần chấm fitness |
| APE | [IT4906 APE](https://www.kaggle.com/code/drakan02/it4906-ape-50-val-and-full-test) | 10 prompt ban đầu, 2 thế hệ paraphrase; 30 lần chấm fitness |

Các notebook baseline dùng MedGemma 1.5 4B làm Target LLM, cùng 50 mẫu validation, fitness `0.7 × BERTScore F1 + 0.3 × ROUGE-L F1`, và đánh giá prompt đã chọn trên toàn bộ test. Mỗi notebook Kaggle có runtime dataset private riêng.

README cho từng thuật toán và hướng dẫn chạy nằm cạnh runner: [RS](src/IT4906/rs/README.md), [Hill Climbing](src/IT4906/hc/README.md), [APE](src/IT4906/ape/README.md).

## Cài đặt

```bash
uv sync
uv add huggingface_hub
```

## Tải dataset

DE đọc trực tiếp `MilitaryHospital175/VNMedical_bv175` từ Hugging Face theo mặc định. Nếu muốn có bản Parquet local để chạy offline hoặc mở notebook EDA, tải riêng bằng lệnh sau.

```bash
uv run python data/med_findings_gen/download_data.py
```

Các shard Parquet sẽ được lưu tại `data/med_findings_gen/VNMedical_bv175/data/`. Dữ liệu lớn này được loại khỏi Git; tải lại bằng script sau khi clone repo.

## Phân tích (EDA)

Mở `data/med_findings_gen/EDA/01_vnmedical_data_analysis.ipynb` trong Jupyter hoặc VS Code và chạy các cell từ trên xuống. Notebook đọc các cột metadata/văn bản để tránh nạp ảnh gốc vào bộ nhớ; phần xem ảnh sử dụng mẫu trong `data/med_findings_gen/sample/`.

## Cấu hình

- `configs/de.yaml`: cấu hình pipeline de.
- `configs/ga.yaml`: cấu hình pipeline ga.
- **`src/IT4906/de/HANDOVER.md`**: hướng dẫn cài đặt + chạy + bàn giao pipeline DE (đọc cái này trước).

## Chạy DE trên Kaggle

Notebook DE gọi `bash deployment/kaggle/run.sh --mode MODE`, với `MODE` là `setup`, `smoke`, `evolve` hoặc `evaluate`. Các cell mẫu nằm trong mục [Chạy trên Kaggle](src/IT4906/de/HANDOVER.md#chạy-trên-kaggle). Hướng dẫn cho các baseline RS/HC/APE nằm trong README tương ứng ở `src/IT4906/`.

## Prompt

- `data/med_findings_gen/prompts.txt`: 10 prompt do người viết, dùng mặc định cho DE.
- `data/med_findings_gen/prompts_auto.txt`: 10 prompt tự động, dùng cho chế độ `gpt`.
