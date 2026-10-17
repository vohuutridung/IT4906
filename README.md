# IT4906 — Phân tích dữ liệu X-quang ngực tiếng Việt

Project chứa prompt tạo báo cáo `Findings`, script tải dataset VNMedical_bv175, notebook phân tích khám phá dữ liệu và config cho các pipeline de/ga.

## Cài đặt

```bash
uv sync
uv add huggingface_hub
```

## Tải dataset

```bash
uv run python data/med_findings_gen/download_data.py
```

Các shard Parquet sẽ được lưu tại `data/med_findings_gen/VNMedical_bv175/data/`. Dữ liệu lớn này được loại khỏi Git; tải lại bằng script sau khi clone repo.

## Phân tích (EDA)

Mở `data/med_findings_gen/EDA/01_vnmedical_data_analysis.ipynb` trong Jupyter hoặc VS Code và chạy các cell từ trên xuống. Notebook đọc các cột metadata/văn bản để tránh nạp ảnh gốc vào bộ nhớ; phần xem ảnh sử dụng mẫu trong `data/med_findings_gen/sample/`.

## Cấu hình

- `configs/de.yaml`: cấu hình pipeline de.
- `configs/ga.yaml`: cấu hình pipeline ga.

## Prompt

- `data/med_findings_gen/prompts.txt`: các phiên bản prompt đang được lưu để so sánh.
- `data/med_findings_gen/prompts_auto.txt`: file prompt tự động, hiện đang trống.
