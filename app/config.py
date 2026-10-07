"""Application configuration — paths default to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

# Keep models, cache, temp, and outputs inside the project directory
HF_HOME = Path(os.getenv("HF_HOME", PROJECT_ROOT / "hf_cache"))
TORCH_HOME = Path(os.getenv("TORCH_HOME", HF_HOME / "torch"))
TEMP_DIR = Path(os.getenv("TEMP", PROJECT_ROOT / "temp"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
COGVIDEO_LOCAL = Path(os.getenv("COGVIDEO_LOCAL", MODELS_DIR / "cogvideo"))
LTX_LOCAL = Path(os.getenv("LTX_LOCAL", MODELS_DIR / "ltx-full"))
LTX_COMPONENTS = Path(os.getenv("LTX_COMPONENTS", MODELS_DIR / "ltx-components"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

for path in (HF_HOME, TORCH_HOME, TEMP_DIR, MODELS_DIR, OUTPUTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("TEMP", str(TEMP_DIR))
os.environ.setdefault("TMP", str(TEMP_DIR))
# Windows without Developer Mode cannot create symlinks in HF cache
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
MAX_VRAM_MB = int(os.getenv("MAX_VRAM_MB", "5500"))

# Model identifiers
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COGVIDEO_MODEL = "THUDM/CogVideoX-2b"
LTX_FP8_REPO = "Symphone/ltx-video-2b-v0.9-fp8"
LTX_FALLBACK_REPO = "Lightricks/LTX-Video"

# Presets tuned for RTX 4050 6GB
PRESETS: dict[str, dict] = {
    "fast": {
        "model": "ltx",
        "width": 512,
        "height": 288,
        "num_frames": 49,
        "num_inference_steps": 8,
        "upscale": False,
    },
    "balanced": {
        "model": "ltx",
        "width": 704,
        "height": 480,
        "num_frames": 49,
        "num_inference_steps": 8,
        "upscale": False,
    },
    "quality": {
        "model": "cogvideo",
        "width": 720,
        "height": 480,
        "num_frames": 49,
        "num_inference_steps": 40,
        "upscale": False,
    },
    "upscaled": {
        "model": "ltx",
        "width": 512,
        "height": 288,
        "num_frames": 49,
        "num_inference_steps": 8,
        "upscale": True,
    },
}
