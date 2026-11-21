# IT4906 — Phân tích dữ liệu X-quang ngực tiếng Việt

Kaggle notebook chạy full pipeline (giải thuật DE): https://www.kaggle.com/code/vohuutridung/it4906-pipeline
HF repo: [vohuutridung/IT4906-TTTH](https://huggingface.co/datasets/vohuutridung/IT4906-TTTH)

Project chứa prompt tạo báo cáo `Findings`, script tải dataset VNMedical_bv175, notebook phân tích khám phá dữ liệu và config cho các pipeline de/ga.

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

Notebook gọi `bash deployment/kaggle/run.sh --mode MODE`, với `MODE` là `setup`, `smoke`, `evolve` hoặc `evaluate`. Các cell mẫu nằm trong mục [Chạy trên Kaggle](src/IT4906/de/HANDOVER.md#chạy-trên-kaggle).

## Prompt

- `data/med_findings_gen/prompts.txt`: 10 prompt do người viết, dùng mặc định cho DE.
- `data/med_findings_gen/prompts_auto.txt`: 10 prompt tự động, dùng cho chế độ `gpt`.
