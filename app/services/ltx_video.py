"""Primary T2V using LTX-Video 2B — falls back to CogVideoX on 16 GB RAM if T5 load fails."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

from app.config import HF_HOME, LTX_COMPONENTS, LTX_FALLBACK_REPO, LTX_LOCAL, MODELS_DIR, OUTPUTS_DIR
from app.services.model_manager import ModelId, check_vram_guard, clear_gpu_memory, model_manager
from app.services.text_encoder_utils import encode_t5_prompt

logger = logging.getLogger(__name__)

_pipeline: LTXPipeline | None = None
_using_fp8 = False


def _ltx_text_source() -> str:
    if (LTX_COMPONENTS / "text_encoder").exists():
        return str(LTX_COMPONENTS)
    if (LTX_LOCAL / "model_index.json").exists():
        return str(LTX_LOCAL)
    return LTX_FALLBACK_REPO


def _ltx_config_source() -> str:
    if (LTX_LOCAL / "model_index.json").exists():
        return str(LTX_LOCAL)
    return LTX_FALLBACK_REPO


def _find_ltx_checkpoint() -> Path | None:
    for subdir in ("ltx-fp8", "ltx"):
        ltx_dir = MODELS_DIR / subdir
        if not ltx_dir.exists():
            continue
        for pattern in (
            "ltx-video-2b*fp8*.safetensors",
            "*distilled*fp8*.safetensors",
            "*distilled*.safetensors",
        ):
            matches = sorted(ltx_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _encode_ltx_prompt_int8(prompt: str) -> torch.Tensor:
    """Last-resort: INT8 T5 to fit 16 GB RAM."""
    from transformers import T5EncoderModel, T5TokenizerFast

    dtype = torch.float16
    source = _ltx_text_source()
    local_only = source in (str(LTX_LOCAL), str(LTX_COMPONENTS))
    tokenizer = T5TokenizerFast.from_pretrained(
        source, subfolder="tokenizer", cache_dir=str(HF_HOME), local_files_only=local_only
    )
    text_encoder = T5EncoderModel.from_pretrained(
        source,
        subfolder="text_encoder",
        torch_dtype=dtype,
        cache_dir=str(HF_HOME),
        low_cpu_mem_usage=True,
    )
    try:
        from torchao.quantization import int8_weight_only, quantize_
        quantize_(text_encoder, int8_weight_only())
        logger.info("Using INT8 quantized T5 for LTX")
    except ImportError:
        pass

    text_encoder.eval()
    inputs = tokenizer(prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds = text_encoder(**inputs).last_hidden_state.cpu()
    del text_encoder, tokenizer
    gc.collect()
    clear_gpu_memory()
    return embeds


def _load() -> None:
    global _pipeline, _using_fp8
    if _pipeline is not None:
        return

    clear_gpu_memory()
    gc.collect()

    checkpoint = _find_ltx_checkpoint()
    dtype = torch.float16
    # Use local components dir as config to avoid HF hub symlink issues on Windows
    config_source = str(LTX_COMPONENTS) if (LTX_COMPONENTS / "model_index.json").exists() else _ltx_config_source()

    if checkpoint:
        logger.info("Loading LTX checkpoint: %s (config=%s)", checkpoint.name, config_source)
        _pipeline = LTXPipeline.from_single_file(
            str(checkpoint),
            text_encoder=None,
            tokenizer=None,
            torch_dtype=dtype,
            config=config_source,
        )
        _using_fp8 = "fp8" in checkpoint.name.lower()
    else:
        logger.info("Loading LTX from %s", config_source)
        _pipeline = LTXPipeline.from_pretrained(
            config_source,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=dtype,
            cache_dir=str(HF_HOME),
            low_cpu_mem_usage=True,
        )
        _using_fp8 = False

    _pipeline.enable_sequential_cpu_offload()
    if hasattr(_pipeline, "vae") and _pipeline.vae is not None:
        _pipeline.vae.enable_tiling()
    model_manager.register_unload(ModelId.LTX, _unload)
    logger.info("LTX pipeline loaded (fp8=%s)", _using_fp8)


def _unload() -> None:
    global _pipeline, _using_fp8
    _pipeline = None
    _using_fp8 = False
    clear_gpu_memory()
    gc.collect()
    logger.info("LTX pipeline unloaded")


def generate_ltx_video(
    prompt: str,
    width: int = 512,
    height: int = 288,
    num_frames: int = 49,
    num_inference_steps: int = 8,
    seed: int = 42,
    output_name: str = "ltx_output",
) -> Path:
    """Generate video with LTX-Video 2B."""
    if not check_vram_guard():
        width, height = 512, 288
        logger.warning("VRAM high — falling back to 512x288")

    src = _ltx_text_source()
    local = src == str(LTX_COMPONENTS)
    try:
        prompt_embeds, prompt_attention_mask = encode_t5_prompt(
            src, prompt, max_length=128, local_files_only=local
        )
    except Exception as exc:
        logger.warning("LTX T5 encode failed (%s), using INT8 fallback", exc)
        prompt_embeds = _encode_ltx_prompt_int8(prompt)
        prompt_attention_mask = torch.ones(
            prompt_embeds.shape[:2], dtype=torch.long
        )

    with model_manager.load(ModelId.LTX):
        _load()
        assert _pipeline is not None

        generator = torch.Generator(device="cpu").manual_seed(seed)
        logger.info(
            "LTX generate: %dx%d, %d frames, %d steps",
            width, height, num_frames, num_inference_steps,
        )

        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_prompt_attention_mask = torch.zeros_like(prompt_attention_mask)
        result = _pipeline(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        frames = result.frames[0]
        output_path = OUTPUTS_DIR / f"{output_name}.mp4"
        export_to_video(frames, str(output_path), fps=24)
        logger.info("LTX video saved: %s", output_path)
        return output_path
