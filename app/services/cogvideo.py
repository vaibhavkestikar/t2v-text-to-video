"""Fallback T2V using CogVideoX-2B with memory-efficient loading."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

from app.config import COGVIDEO_LOCAL, COGVIDEO_MODEL, HF_HOME, OUTPUTS_DIR
from app.services.model_manager import ModelId, check_vram_guard, clear_gpu_memory, model_manager
from app.services.text_encoder_utils import encode_t5_prompt

logger = logging.getLogger(__name__)

_pipeline: CogVideoXPipeline | None = None


def _model_source() -> str:
    """Prefer fully-downloaded local copy on D: over incomplete HF cache."""
    if (COGVIDEO_LOCAL / "model_index.json").exists():
        return str(COGVIDEO_LOCAL)
    return COGVIDEO_MODEL


def _load_pipeline() -> CogVideoXPipeline:
    """Load transformer + VAE only (text encoder loaded separately per prompt)."""
    clear_gpu_memory()
    gc.collect()
    dtype = torch.float16
    source = _model_source()
    logger.info("Loading CogVideoX transformer + VAE from %s", source)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        source,
        subfolder="transformer",
        torch_dtype=dtype,
        cache_dir=str(HF_HOME),
        local_files_only=source == str(COGVIDEO_LOCAL),
        low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        source,
        subfolder="vae",
        torch_dtype=dtype,
        cache_dir=str(HF_HOME),
        local_files_only=source == str(COGVIDEO_LOCAL),
        low_cpu_mem_usage=True,
    )

    pipe = CogVideoXPipeline.from_pretrained(
        source,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
        cache_dir=str(HF_HOME),
        local_files_only=source == str(COGVIDEO_LOCAL),
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return pipe


def _load() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    _pipeline = _load_pipeline()
    model_manager.register_unload(ModelId.COGVIDEO, _unload)
    logger.info("CogVideoX pipeline loaded")


def _unload() -> None:
    global _pipeline
    _pipeline = None
    clear_gpu_memory()
    gc.collect()
    logger.info("CogVideoX pipeline unloaded")


def generate_cogvideo(
    prompt: str,
    width: int = 720,
    height: int = 480,
    num_frames: int = 49,
    num_inference_steps: int = 40,
    seed: int = 42,
    output_name: str = "cogvideo_output",
) -> Path:
    """Generate video with CogVideoX-2B."""
    if not check_vram_guard():
        width, height = 512, 320
        logger.warning("VRAM high — falling back to 512x320")

    # Encode prompt first, then unload T5 before loading transformer
    src = _model_source()
    prompt_embeds, _ = encode_t5_prompt(src, prompt, local_files_only=(src == str(COGVIDEO_LOCAL)))

    with model_manager.load(ModelId.COGVIDEO):
        _load()
        assert _pipeline is not None

        generator = torch.Generator(device="cpu").manual_seed(seed)
        logger.info(
            "CogVideoX generate: %dx%d, %d frames, %d steps",
            width, height, num_frames, num_inference_steps,
        )

        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        result = _pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
            generator=generator,
        )

        frames = result.frames[0]
        output_path = OUTPUTS_DIR / f"{output_name}.mp4"
        export_to_video(frames, str(output_path), fps=8)
        logger.info("CogVideoX video saved: %s", output_path)
        return output_path
