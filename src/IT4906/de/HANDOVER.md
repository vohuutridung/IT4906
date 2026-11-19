# HANDOVER — Pipeline DE (EvoPrompt) Medical Findings Generation

> Tài liệu bàn giao cho người tiếp nhận chạy full run. Cập nhật: **2026-11-14**.
> Đặc tả chi tiết module: `README.md` (cùng thư mục).
> ⚠️ Mọi lệnh bên dưới chạy **từ root repo** (thư mục chứa `configs/`, `pyproject.toml`).

## 1. Dự án là gì

Tối ưu prompt tạo phần **Findings** trong báo cáo X-quang ngực tiếng Việt bằng
**Differential Evolution + LLM** (thuật toán EvoPrompt):

- **Target LLM** — `google/medgemma-1.5-4b-it`: sinh Findings theo prompt đang xét.
- **Fitness** — `0.7 × BERTScore(xlm-roberta-large) + 0.3 × ROUGE-L(F1)` so với findings thật, trên 200 sample đầu của split `val` (VNMedical_bv175).
- **Evolution LLM** — model lớn lai ghép + đột biến prompt theo instruction 4 bước EvoPrompt. Chọn 1 trong 2 provider (xem §4).

## 2. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Code pipeline DE end-to-end | ✅ Xong, đã push nhánh `feat/de-med-findings` |
| Chạy thật kiểm chứng (đợt ngắn, Kaggle 2×T4, 11–12/11) | ✅ Đã xác nhận các fix (fp16 NaN, OOM, offspring rác, cache poison) |
| Endpoint thật cho Evolution LLM (base_url / api_key / model name) | ❌ **Chưa có** — đang placeholder trong `.env.example` |
| **Full run** (pool 20 → top-10 → 10 thế hệ × population 10 × 200 sample) | ❌ **Chưa chạy** — hết quota Kaggle 30h/tuần của người viết |

## 3. Cài đặt

```bash
git clone git@github.com:vohuutridung/IT4906.git && cd IT4906
git checkout feat/de-med-findings
uv sync                     # deps trong pyproject + tự cài package (editable)

# Deps nặng CHƯA đưa vào pyproject (file dùng chung với nhóm GA — chờ xác nhận).
# Cài riêng trên MÁY CHẠY (local/dev100/Kaggle):
uv pip install torch transformers accelerate bitsandbytes datasets pillow bert_score rouge-score pyyaml

# Credential Evolution LLM — KHÔNG hardcode vào code:
cp .env.example .env        # rồi điền giá trị thật vào 2 biến (xem §4)

# Tải dataset (parquet lớn, ngoài git):
uv run python data/med_findings_gen/download_data.py
```

`.env` đã git-ignore — không bao giờ commit. Biến môi trường export thủ công cũng được
nhận (file `.env` chỉ được nạp nếu có, không ghi đè biến có sẵn).

## 4. Chọn provider cho Evolution LLM (`configs/de.yaml` → `evolution_llm.provider`)

| | `openai` (mặc định) | `local` |
|---|---|---|
| Cách gọi | `OpenAI(base_url, api_key)` qua API OpenAI-compatible | transformers load `medgemma-27b-text-it` 4-bit |
| Credential | `.env`: `EVOLUTION_LLM_BASE_URL`, `EVOLUTION_LLM_API_KEY` | Không cần (nhưng cần `HF_TOKEN` để tải model) |
| Yêu cầu GPU | Nhẹ — chỉ target 4B + BERTScore (~1 GPU 16GB thoải mái) | Nặng — 27B 4-bit cần ~14.5GiB: Kaggle **2×T4** (tự chia 4+14GiB) hoặc 1 GPU ≥24GB |
| Khi nào dùng | Khi đã có endpoint (chỉ dẫn của Dũng 13/11) | Khi không có endpoint |

⚠️ Với `openai`: `evolution_llm.model_id` phải là **tên model của endpoint** (chờ người
cung cấp endpoint chốt). Retry 3 lần tự động khi endpoint giật.

## 5. Chạy

```bash
# 1) SMOKE TRƯỚC — 5 sample × 4 prompt × 1 thế hệ, vài phút.
#    Kiểm tra trọn bộ: dataset, .env/endpoint, GPU, cache. Output vào runs/de_smoke/ (tách biệt).
uv run python -m IT4906.de.run_de --config configs/de.yaml --smoke

# 2) FULL RUN — output vào runs/de/
uv run python -m IT4906.de.run_de --config configs/de.yaml

# Ghi đè config tạm thời (không sửa file), VD chạy ít sample hơn để đo thời gian:
uv run python -m IT4906.de.run_de --config configs/de.yaml --set data.num_samples 50
```

### Chạy lâu trên server (dev100) — khuyến nghị dùng tmux

```bash
mkdir -p runs && tmux new -s de
uv run python -m IT4906.de.run_de --config configs/de.yaml 2>&1 | tee runs/de_log_$(date +%m%d_%H%M).txt
# Ctrl+B rồi D để thoát (chạy tiếp nền); quay lại: tmux attach -t de
```

### Chạy trên Kaggle

Bật GPU và Internet, rồi attach dataset VNMedical_bv175. `--data-root` trỏ đến thư mục chứa `val-*.parquet`, `data/val-*.parquet` hoặc `val/*.parquet`.

**Cell 1. Clone và cài dependencies**

```python
!git clone --branch main --depth 1 https://github.com/vohuutridung/IT4906.git /kaggle/working/IT4906
%cd /kaggle/working/IT4906
!bash deployment/kaggle/run.sh --mode setup
```

**Cell 2. Nạp Kaggle Secrets**

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["EVOLUTION_LLM_BASE_URL"] = secrets.get_secret("EVOLUTION_LLM_BASE_URL")
os.environ["EVOLUTION_LLM_API_KEY"] = secrets.get_secret("EVOLUTION_LLM_API_KEY")
```

`HF_TOKEN` cần có quyền tải target model. Nếu dùng Evolution LLM local, bỏ hai dòng `EVOLUTION_LLM_*`, dùng `--set evolution_llm.provider local` và đặt model ID theo repo Hugging Face đã được cấp quyền.

**Cell 3. Chạy thử**

Thay `--data-root` bằng đường dẫn dataset đã attach và `MODEL_ID_CUA_ENDPOINT` bằng tên model mà endpoint cung cấp.

```python
!bash deployment/kaggle/run.sh \
  --mode smoke \
  --data-root /kaggle/input/vnmedical-bv175 \
  --work-root /kaggle/working/it4906 \
  --set evolution_llm.model_id MODEL_ID_CUA_ENDPOINT
```

**Cell 4. Chạy đầy đủ**

```python
!bash deployment/kaggle/run.sh \
  --mode evolve \
  --data-root /kaggle/input/vnmedical-bv175 \
  --work-root /kaggle/working/it4906 \
  --set evolution_llm.model_id MODEL_ID_CUA_ENDPOINT
```

`smoke` ghi vào `/kaggle/working/it4906/runs/de_smoke/`, còn `evolve` ghi vào `/kaggle/working/it4906/runs/de/`. Kết quả cuối là `best_prompt.json`. Để tiếp tục ở session mới, chép thư mục kết quả cũ vào đúng thư mục output và gọi lại `evolve` với cấu hình cũ. Runner tự tìm `checkpoint.json`.

Thêm `--set data.num_samples 50` hoặc `--set de.iterations 3` để thử cấu hình nhỏ hơn. Dùng một `--work-root` riêng cho từng cấu hình thí nghiệm.

## 6. Checkpoint & cache — đọc trước khi dọn dẹp

Tất cả nằm trong `runs/de/` (hoặc thư mục `de.cache_dir`):

| File | Vai trò |
|---|---|
| `checkpoint.json` | Lưu sau MỖI thế hệ. Chạy lại **cùng config** → tự resume, không mất công cũ |
| `checkpoint_stale.json` | Checkpoint sinh từ config KHÁC bị đổi tên thế này (chống trộn fitness 2 thang đo) — có thể xóa |
| `gen_cache_<fp>.jsonl` | Cache generation target 4B. Đổi dtype/do_sample/max_new_tokens → `<fp>` mới → cache cũ không tái dùng (chống poison) |
| `fitness_cache_<fp>.json` | Cache điểm fitness. Đổi metric/model → file mới |
| `stepN_pop.json` | Population + điểm sau mỗi thế hệ |
| `best_prompt.json` | **Kết quả cuối**: best_prompt, best_fitness, final_population có điểm từng prompt |

Muốn chạy lại từ đầu: **đổi `de.cache_dir`** (VD `runs/de_v2`) — đừng xóa cache cũ (tra cứu so sánh).
Smoke luôn ghi vào `runs/de_smoke/`, không đụng full run.

## 7. Config chính (`configs/de.yaml`)

| Nhóm | Tham số | Giá trị | Ghi chú |
|---|---|---|---|
| seed | | 5 | official default |
| data | num_samples | 200 | 200 sample ĐẦU của val (thứ tự parquet gốc — cần nhóm xác nhận) |
| target_llm | batch_size | **4** (official 16) | ⚠️ lệch official vì VRAM T4; GPU ≥24GB có thể trả lại 16 |
| target_llm | dtype | bfloat16 | ⚠️ fp16 gây NaN trên Gemma3/T4 — KHÔNG đổi lại |
| fitness | weight | 0.7 BERTScore + 0.3 ROUGE-L | chốt tạm 10/11, swappable qua config |
| evolution_llm | max_new_tokens | **1500** (official 1000) | ⚠️ lệch official: 27B trích-dẫn donors làm cắt cụt thẻ `<prompt>` (thật 12/11) |
| evolution_llm | temperature / top_p | 0.5 / 0.95 | official default |
| de | population_size / iterations | 10 / 10 | theo file TTTH |
| de | initial_mode | topk | P₀ = top-10 của pool 20 (2 file prompts.txt + prompts_auto.txt) |

## 8. Sự cố đã gặp thật (đã fix — gặp lại thì tra đây)

| Hiện tượng | Nguyên nhân | Đã xử lý |
|---|---|---|
| Output rỗng / fitness rác | fp16 → logits NaN trên Gemma3 (T4) | Chuyển bf16 emulated (`configs/de.yaml`) |
| CUDA OOM lúc prefill 27B | 2×T4: transformers tự trừ 10% max_memory → layer rơi CPU; mảnh vỡ allocator | `max_memory {0:4GiB, 1:14GiB}`, pin target lên GPU0, `empty_cache` mỗi lần sinh, giảm batch |
| OOM SigLIP (vision tower) | bf16 trên GPU sm<80 rớt backend "math" (O(n²)) | `cuda_compat.py`: attention tính fp16, trọng số vẫn bf16 |
| Offspring "rác 0.58" | 27B bỏ thẻ `<prompt>`, trả bản phân tích CoT → fallback dùng nguyên response | `parse_final_prompt` trả `""` → de_loop giữ prompt cũ, bỏ eval |
| Cache trả output cũ dù đã đổi config | Cache key thiếu cấu hình generation/metric | Fingerprint cấu hình vào cache key + checkpoint `run_fp` |
| `ValueError: not enough values to unpack` khi decode | transformers 5.x trả list | `batch_decode(...)[0]` |

## 9. Việc còn treo / cần quyết định

1. **Chặn full run qua API**: endpoint thật (base_url + model name + api_key) → điền `.env`.
2. **Full run** — người giữ quota/máy thực hiện (Kaggle ≤30h/tuần; dev100 nếu còn trống).
3. Nếu chạy `local`: chốt repo chính xác của 27B (config đang ghi ⚠️).
4. Nhóm review 2 điểm lệch official (§7) + dòng "Important" thêm vào template EvoPrompt.
5. Xác nhận `num_samples = 200` đầu val là đúng thỏa thuận nhóm.
6. PR `feat/de-med-findings` → `main` sau khi review.
