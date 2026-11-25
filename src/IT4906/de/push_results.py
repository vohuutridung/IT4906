"""Upload an EvoPrompt-DE results folder to the IT4906 dataset repository."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi


REPO_ID = "vohuutridung/IT4906-TTTH"


def _read_result(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def _report_run_status(folder: Path) -> None:
    snapshot = _read_result(folder / "run_config.json")
    run_id = snapshot.get("run_id")
    if not run_id:
        return

    best = _read_result(folder / "best_prompt.json")
    if best.get("run_id") == run_id and best.get("run_fp") == snapshot.get("run_fp"):
        evaluation = _read_result(folder / "test_eval" / "metrics.json")
        if evaluation.get("run_id") != run_id:
            print("[push-results] Cảnh báo: run đã có best prompt nhưng thiếu eval test đầy đủ cho run này.")
        return

    checkpoint = _read_result(folder / "checkpoint.json")
    if checkpoint.get("run_id") == run_id and checkpoint.get("run_fp") == snapshot.get("run_fp"):
        progress = f"checkpoint sau thế hệ {checkpoint.get('step_done')}"
    else:
        initial = _read_result(folder / "evolution" / run_id / "initial_population.json")
        if "candidates" in initial:
            scored = len(initial["evaluated_prompts"])
            total = len(initial["candidates"])
            progress = f"đã chấm {scored}/{total} prompt khởi tạo"
        else:
            progress = "chưa có checkpoint P0"
    print(f"[push-results] Cảnh báo: đang upload kết quả dở dang ({progress}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-folder", type=Path, required=True)
    parser.add_argument("--hf-folder", required=True, help="Relative folder inside the dataset repository.")
    args = parser.parse_args()

    if not args.local_folder.is_dir():
        parser.error(f"Không tìm thấy thư mục local: {args.local_folder}")
    hf_folder = PurePosixPath(args.hf_folder.strip())
    if hf_folder.is_absolute() or not hf_folder.parts or ".." in hf_folder.parts:
        parser.error("--hf-folder phải là đường dẫn thư mục tương đối trong repo, ví dụ runs/qwen35-9b.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("Cần đặt HF_TOKEN có quyền ghi vào dataset repo.")

    local_folder = args.local_folder.resolve()
    _report_run_status(local_folder)
    print(f"[push-results] {local_folder} → {REPO_ID}/{hf_folder}")
    commit = HfApi().upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=local_folder,
        path_in_repo=hf_folder.as_posix(),
        token=token,
        commit_message=f"Upload DE results to {hf_folder}",
    )
    print(f"[push-results] Commit: {commit.commit_url}")


if __name__ == "__main__":
    main()
