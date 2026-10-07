"""Load T5 text encoder, encode prompt, then unload to save RAM."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig, T5EncoderModel, T5TokenizerFast

from app.config import HF_HOME
from app.services.model_manager import clear_gpu_memory

logger = logging.getLogger(__name__)


def _is_local(path: str) -> bool:
    return Path(path).exists()


def encode_t5_prompt(
    repo_id: str,
    prompt: str,
    subfolder: str = "text_encoder",
    tokenizer_subfolder: str = "tokenizer",
    max_length: int = 226,
    local_files_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompt with T5-XXL using 8-bit loading (~5 GB RAM instead of ~9.5 GB).
    """
    local = local_files_only or _is_local(repo_id)
    logger.info("Loading T5 (8-bit) from %s [local=%s]", repo_id, local)

    tokenizer = T5TokenizerFast.from_pretrained(
        repo_id,
        subfolder=tokenizer_subfolder,
        cache_dir=str(HF_HOME),
        local_files_only=local,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    try:
        text_encoder = T5EncoderModel.from_pretrained(
            repo_id,
            subfolder=subfolder,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(HF_HOME),
            local_files_only=local,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        logger.warning("8-bit load failed (%s), trying fp16", exc)
        text_encoder = T5EncoderModel.from_pretrained(
            repo_id,
            subfolder=subfolder,
            torch_dtype=torch.float16,
            cache_dir=str(HF_HOME),
            local_files_only=local,
            low_cpu_mem_usage=True,
        )

    text_encoder.eval()

    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    device = next(text_encoder.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        prompt_embeds = outputs.last_hidden_state.to(torch.float16).cpu()
        attn_cpu = inputs.attention_mask.cpu()

    del text_encoder, tokenizer, outputs, inputs
    gc.collect()
    clear_gpu_memory()
    logger.info("T5 encoded shape=%s, unloaded", tuple(prompt_embeds.shape))
    return prompt_embeds, attn_cpu
