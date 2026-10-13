# IT4906 — Phân tích dữ liệu X-quang ngực tiếng Việt

Project chứa prompt tạo báo cáo `Findings`, script tải dataset VNMedical_bv175 và notebook phân tích khám phá dữ liệu.

## Cài đặt

```bash
uv sync
uv add huggingface_hub
```

## Tải dataset

```bash
uv run python data/download_data.py
```

Các shard Parquet sẽ được lưu tại `data/VNMedical_bv175/data/`. Dữ liệu lớn này được loại khỏi Git; tải lại bằng script sau khi clone repo.

## Phân tích

Mở `notebooks/01_vnmedical_data_analysis.ipynb` trong Jupyter hoặc VS Code và chạy các cell từ trên xuống. Notebook đọc các cột metadata/văn bản để tránh nạp ảnh gốc vào bộ nhớ; phần xem ảnh sử dụng mẫu trong `data/sample/`.

## Prompt

- `data/prompts.txt`: các phiên bản prompt đang được lưu để so sánh.
- `data/prompts_auto.txt`: file prompt tự động, hiện đang trống.
