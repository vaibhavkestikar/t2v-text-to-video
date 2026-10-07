"""Re-download incomplete model weights to D: drive (local_dir, no symlinks)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from huggingface_hub import snapshot_download

from app.config import HF_HOME, MODELS_DIR

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HOME"] = str(HF_HOME)


def verify_safetensors(path: Path, min_files: int = 1) -> int:
    files = list(path.rglob("*.safetensors"))
    print(f"  {path.name}: {len(files)} safetensors files")
    return len(files)


def download_full(repo: str, local_dir: Path) -> None:
    print(f"\n=== Downloading {repo} -> {local_dir} ===")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    count = verify_safetensors(local_dir)
    if count < 1:
        raise RuntimeError(f"No weights downloaded for {repo}")


def main() -> None:
    cog_dir = MODELS_DIR / "cogvideo"
    ltx_dir = MODELS_DIR / "ltx-full"

    download_full("THUDM/CogVideoX-2b", cog_dir)
    download_full("Lightricks/LTX-Video", ltx_dir)

    print("\n=== Verification ===")
    verify_safetensors(cog_dir)
    verify_safetensors(ltx_dir)
    print("\nRepair complete.")


if __name__ == "__main__":
    main()
