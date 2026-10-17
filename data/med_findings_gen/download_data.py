"""Download the VNMedical_bv175 dataset into data/med_findings_gen/VNMedical_bv175/."""

from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with: uv add huggingface_hub"
    ) from exc


REPO_ID = "MilitaryHospital175/VNMedical_bv175"
DESTINATION = Path(__file__).resolve().parent / "VNMedical_bv175"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DESTINATION,
    )
    print(f"Dataset downloaded to: {downloaded_path}")


if __name__ == "__main__":
    main()
