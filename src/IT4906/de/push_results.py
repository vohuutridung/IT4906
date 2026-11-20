"""Upload an EvoPrompt-DE results folder to the IT4906 dataset repository."""

import argparse
import os
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi


REPO_ID = "vohuutridung/IT4906-TTTH"


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
