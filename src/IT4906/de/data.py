"""Load dữ liệu ViX-Ray (VNMedical_bv175) + render prompt cho target LLM."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from PIL import Image

# Prompt hệ thống theo Appendix A (paper ViX-Ray)
SYSTEM_PROMPT = "Bạn là trợ lý bác sĩ"

# Câu hỏi mặc định theo format Stage 1 của paper (dùng khi prompt không chứa placeholder)
DEFAULT_TASK_QUESTION = "Cho biết bệnh nhân bị gì?"

# Mapping placeholder trong prompt mẫu -> tên cột dataset.
# ⚠️ Prompt mẫu ghi "Kĩ thuật" (kẻ-short i) trong khi cột dataset là "Kỹ thuật".
PLACEHOLDER_ALIASES = {
    "Ages": "Ages",
    "sex": "sex",
    "View": "view",
    "Ghi chú": "Ghi chú",
    "Kĩ thuật": "Kỹ thuật",  # placeholder -> cột
    "Kỹ thuật": "Kỹ thuật",
}


@dataclass
class Sample:
    idx: int
    meta: dict[str, Any]  # Ages, sex, view, Ghi chú, Kỹ thuật
    image: Image.Image
    findings_gt: str


def find_parquet_files(dataset_dir: str, split: str) -> list[str]:
    """Tìm các shard parquet của một split, sắp theo tên tăng dần.

    dataset_dir có thể là HF snapshot (parquet nằm ở <dir>/data/) hoặc thư mục phẳng
    (Kaggle dataset). Thứ tự file + thứ tự dòng trong file = "thứ tự gốc" của split
    (cần xác nhận lại với nhóm — item mở trong TODO).
    """
    candidates = [
        os.path.join(dataset_dir, "data", f"{split}-*.parquet"),
        os.path.join(dataset_dir, f"{split}-*.parquet"),
        os.path.join(dataset_dir, split, "*.parquet"),
    ]
    for pattern in candidates:
        files = sorted(glob.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(
        f"Không tìm thấy parquet cho split '{split}' trong '{dataset_dir}'. "
        f"Đã thử: {candidates}"
    )


def load_split(dataset_dir: str, split: str, num_samples: int | None = None) -> list[Sample]:
    """Load split theo thứ tự parquet gốc, lấy num_samples mẫu đầu (mặc định: hết)."""
    files = find_parquet_files(dataset_dir, split)
    ds = load_dataset("parquet", data_files={split: files}, split=split)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        meta = {
            "Ages": str(row["Ages"]),
            "sex": str(row["sex"]),
            "view": str(row["view"]),
            "Ghi chú": str(row["Ghi chú"]),
            "Kỹ thuật": str(row["Kỹ thuật"]),
        }
        samples.append(
            Sample(
                idx=i,
                meta=meta,
                image=row["image"],
                findings_gt=str(row["findings"]).strip(),
            )
        )
    return samples


def render_prompt(template: str, sample: Sample, append_metadata: bool = True) -> str:
    """Render prompt evolve + metadata của sample.

    - Nếu template chứa placeholder {Ages} {sex} {View} {Ghi chú} {Kĩ thuật} -> substitute
      (format_map an toàn: placeholder lạ giữ nguyên).
    - Nếu template KHÔNG chứa placeholder nào -> nối thêm dòng metadata + câu hỏi
      theo format Stage 1 của paper ViX-Ray.
    """
    meta = dict(sample.meta)
    if "{" in template:
        filled = template.format_map(_SafeDict(meta))
        return filled.strip()
    rendered = template.strip()
    if append_metadata:
        meta_line = (
            f"Thông tin bệnh nhân: {meta['sex']}, {meta['Ages']} tuổi. "
            f"Chụp X-quang {meta['view']} ({meta['Kỹ thuật']}; {meta['Ghi chú']})."
        )
        rendered = f"{rendered}\n\n{meta_line}\n\n{DEFAULT_TASK_QUESTION}"
    return rendered.strip()


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def substitute_placeholders(template: str, meta: dict[str, Any]) -> str:
    """Substitute placeholder theo alias (xử lý cả 'Kĩ thuật' vs 'Kỹ thuật')."""
    resolved = {alias: meta[col] for alias, col in PLACEHOLDER_ALIASES.items() if col in meta}
    return template.format_map(_SafeDict(resolved))
