"""Download all model weights to D: drive."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env.example")

from huggingface_hub import hf_hub_download, snapshot_download

from app.config import HF_HOME, MODELS_DIR

# Pin caches to D:
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))


def download_qwen() -> None:
    print("=== Downloading Qwen2.5-1.5B-Instruct ===")
    snapshot_download(
        "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir=str(HF_HOME),
    )
    print("Qwen download complete.")


def download_cogvideo() -> None:
    print("=== Downloading CogVideoX-2B ===")
    snapshot_download(
        "THUDM/CogVideoX-2b",
        cache_dir=str(HF_HOME),
    )
    print("CogVideoX download complete.")


def download_ltx_fp8() -> None:
    print("=== Downloading LTX-Video 2B distilled ===")
    ltx_dir = MODELS_DIR / "ltx"
    ltx_dir.mkdir(parents=True, exist_ok=True)

    # Symphone FP8 single-file checkpoint (lower VRAM)
    symphone_dir = MODELS_DIR / "ltx-fp8"
    symphone_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id="Symphone/ltx-video-2b-v0.9-fp8",
            filename="ltx-video-2b-v0.9-fp8_e4m3fn.safetensors",
            local_dir=str(symphone_dir),
        )
        print(f"  Downloaded Symphone FP8: {path}")
    except Exception as exc:
        print(f"  Symphone FP8 skipped: {exc}")

    repo = "Lightricks/LTX-Video"
    for filename in (
        "ltxv-2b-0.9.8-distilled-fp8.safetensors",
        "ltxv-2b-0.9.6-distilled-04-25.safetensors",
    ):
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(ltx_dir),
            )
            print(f"  Downloaded: {path}")
            break
        except Exception as exc:
            print(f"  Skipped {filename}: {exc}")

    # Full diffusers pipeline components (text encoder, vae, config)
    print("  Downloading LTX diffusers components...")
    snapshot_download(
        repo,
        cache_dir=str(HF_HOME),
        allow_patterns=[
            "*.json", "*.yaml", "*.txt",
            "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*",
        ],
    )
    print("LTX download complete.")


def download_realesrgan() -> None:
    print("=== Downloading Real-ESRGAN x2 weights (optional) ===")
    realesrgan_dir = MODELS_DIR / "realesrgan"
    realesrgan_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x2plus.pth",
            local_dir=str(realesrgan_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded: {path}")
    except Exception:
        try:
            # Alternative source
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            dest = realesrgan_dir / "RealESRGAN_x2plus.pth"
            if not dest.exists():
                print(f"  Downloading from {url}...")
                urllib.request.urlretrieve(url, dest)
                print(f"  Saved: {dest}")
        except Exception as exc:
            print(f"  Real-ESRGAN download skipped: {exc}")
    print("Real-ESRGAN step complete.")


def main() -> None:
    print(f"Models dir: {MODELS_DIR}")
    print(f"HF cache:   {HF_HOME}")
    download_qwen()
    download_cogvideo()
    download_ltx_fp8()
    download_realesrgan()
    print("\nAll downloads finished.")


if __name__ == "__main__":
    main()
