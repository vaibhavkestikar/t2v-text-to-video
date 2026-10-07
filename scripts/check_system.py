"""Verify system meets minimum requirements for local T2V."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import COGVIDEO_LOCAL, HF_HOME, LTX_COMPONENTS, MODELS_DIR, PROJECT_ROOT
from app.services.model_manager import get_vram_usage_mb


def check_disk() -> bool:
    ok = True
    for label, path in [("Project (D:)", PROJECT_ROOT), ("HF cache", HF_HOME), ("Models", MODELS_DIR)]:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        print(f"  {label}: {free_gb:.1f} GB free")
        if free_gb < 5:
            print(f"    WARNING: less than 5 GB free at {path}")
            ok = False
    c_usage = shutil.disk_usage("C:\\")
    c_free = c_usage.free / (1024**3)
    print(f"  C: drive: {c_free:.1f} GB free")
    if c_free < 15:
        print("    WARNING: C: drive low — Windows page file may fail during model load.")
        print("    Fix: free C: space or move page file to D: (System > Advanced > Performance > Virtual memory)")
        ok = False
    return ok


def check_gpu() -> bool:
    vram = get_vram_usage_mb()
    total = vram.get("total_mb")
    print(f"  GPU VRAM: {total} MB total, {vram.get('free_mb')} MB free")
    if total and total < 5000:
        print("    WARNING: less than 5 GB VRAM — use fast preset only")
        return False
    return True


def check_models() -> bool:
    ok = True
    qwen_ok = (HF_HOME / "models--Qwen--Qwen2.5-1.5B-Instruct").exists()
    print(f"  {'OK' if qwen_ok else 'MISSING'}: Qwen2.5-1.5B-Instruct")
    ok = ok and qwen_ok

    cog_ok = (COGVIDEO_LOCAL / "transformer" / "diffusion_pytorch_model.safetensors").exists()
    print(f"  {'OK' if cog_ok else 'MISSING'}: CogVideoX-2B (local)")
    ok = ok and cog_ok

    ltx_ok = (LTX_COMPONENTS / "text_encoder").exists() and list(MODELS_DIR.rglob("ltx-video-2b*fp8*.safetensors"))
    print(f"  {'OK' if ltx_ok else 'MISSING'}: LTX-Video 2B FP8 + text encoder")
    ok = ok and bool(ltx_ok)
    return ok


def main() -> None:
    print("=== Disk ===")
    disk_ok = check_disk()
    print("=== GPU ===")
    gpu_ok = check_gpu()
    print("=== Models ===")
    models_ok = check_models()
    print()
    if disk_ok and gpu_ok and models_ok:
        print("System check PASSED")
    else:
        print("System check WARNINGS — generation may fail until issues are resolved")
        sys.exit(1)


if __name__ == "__main__":
    main()
