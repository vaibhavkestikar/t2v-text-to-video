"""Optional Real-ESRGAN x2 frame upscaler — loaded after T2V unloads."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video

from app.config import MODELS_DIR, OUTPUTS_DIR
from app.services.model_manager import ModelId, model_manager

logger = logging.getLogger(__name__)

_upsampler = None


def _load() -> None:
    global _upsampler
    if _upsampler is not None:
        return

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = MODELS_DIR / "realesrgan" / "RealESRGAN_x2plus.pth"
        if not model_path.exists():
            logger.warning("Real-ESRGAN weights not found at %s — skipping upscale", model_path)
            return

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        _upsampler = RealESRGANer(
            scale=2,
            model_path=str(model_path),
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        model_manager.register_unload(ModelId.UPSCALER, _unload)
        logger.info("Real-ESRGAN upscaler loaded")
    except ImportError as exc:
        logger.warning("Real-ESRGAN dependencies missing: %s", exc)


def _unload() -> None:
    global _upsampler
    _upsampler = None
    logger.info("Upscaler unloaded")


def _read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def upscale_video(input_path: Path, output_name: str = "upscaled") -> Path:
    """Upscale each frame 2x with Real-ESRGAN."""
    with model_manager.load(ModelId.UPSCALER):
        _load()
        if _upsampler is None:
            logger.warning("Upscaler unavailable — returning original video")
            return input_path

        frames, fps = _read_video_frames(input_path)
        upscaled = []
        for i, frame in enumerate(frames):
            result, _ = _upsampler.enhance(frame, outscale=2)
            upscaled.append(result)
            if i % 10 == 0:
                logger.info("Upscaled frame %d/%d", i + 1, len(frames))

        output_path = OUTPUTS_DIR / f"{output_name}.mp4"
        export_to_video(upscaled, str(output_path), fps=fps)
        logger.info("Upscaled video saved: %s", output_path)
        return output_path
