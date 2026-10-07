"""End-to-end ensemble pipeline: enhancer → T2V → optional upscale."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from app.config import PRESETS
from app.services.cogvideo import generate_cogvideo
from app.services.enhancer import enhance_prompt
from app.services.ltx_video import generate_ltx_video
from app.services.model_manager import ModelId, model_manager
from app.services.preflight import require_preflight
from app.services.upscaler import upscale_video

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


class ModelChoice(str, Enum):
    AUTO = "auto"
    LTX = "ltx"
    COGVIDEO = "cogvideo"


@dataclass
class GenerationRequest:
    prompt: str
    preset: str = "fast"
    model: ModelChoice = ModelChoice.AUTO
    seed: int = 42
    enhance: bool = True


@dataclass
class GenerationResult:
    job_id: str
    original_prompt: str
    enhanced_prompt: str
    video_path: Path
    model_used: str
    preset: str


@dataclass
class JobState:
    job_id: str
    request: GenerationRequest | None = None
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    result: GenerationResult | None = None
    error: str | None = None


def _noop_progress(stage: str, pct: float) -> None:
    logger.info("[%s] %.0f%%", stage, pct * 100)


def _resolve_model(preset_cfg: dict, choice: ModelChoice) -> str:
    if choice == ModelChoice.LTX:
        return "ltx"
    if choice == ModelChoice.COGVIDEO:
        return "cogvideo"
    return preset_cfg.get("model", "ltx")


def run_ensemble(
    request: GenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> GenerationResult:
    """Run the full ensemble pipeline synchronously."""
    progress = on_progress or _noop_progress
    job_id = str(uuid.uuid4())[:8]
    preset_cfg = PRESETS.get(request.preset, PRESETS["fast"])
    model_used = _resolve_model(preset_cfg, request.model)

    progress("init", 0.05)
    require_preflight()

    # Stage 1: Prompt enhancement
    enhanced = request.prompt
    if request.enhance:
        progress("enhancing_prompt", 0.1)
        enhanced = enhance_prompt(request.prompt)
        model_manager.unload_current()  # Free RAM before T2V

    progress("prompt_ready", 0.2)

    # Stage 2: T2V generation
    output_name = f"{job_id}_{model_used}"
    width = preset_cfg["width"]
    height = preset_cfg["height"]
    num_frames = preset_cfg["num_frames"]
    steps = preset_cfg["num_inference_steps"]

    progress("generating_video", 0.25)

    if model_used == "cogvideo":
        video_path = generate_cogvideo(
            prompt=enhanced,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            seed=request.seed,
            output_name=output_name,
        )
    else:
        video_path = generate_ltx_video(
            prompt=enhanced,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            seed=request.seed,
            output_name=output_name,
        )

    model_manager.unload_current()
    progress("video_generated", 0.85)

    # Stage 3: Optional upscale
    if preset_cfg.get("upscale"):
        progress("upscaling", 0.9)
        video_path = upscale_video(video_path, output_name=f"{output_name}_upscaled")
        model_manager.unload_current()

    progress("complete", 1.0)

    return GenerationResult(
        job_id=job_id,
        original_prompt=request.prompt,
        enhanced_prompt=enhanced,
        video_path=video_path,
        model_used=model_used,
        preset=request.preset,
    )
