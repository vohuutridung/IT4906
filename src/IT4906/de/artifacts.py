"""Atomic writes for DE checkpoints and result files."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(text)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
