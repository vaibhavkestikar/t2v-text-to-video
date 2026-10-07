"""Prompt enhancement using Qwen2.5-1.5B-Instruct on CPU."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import HF_HOME, QWEN_MODEL
from app.services.model_manager import ModelId, model_manager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a cinematic prompt engineer for AI video generation. "
    "Expand the user's short prompt into a detailed video description. "
    "Include: subject, action/motion, camera movement, lighting, and atmosphere. "
    "Keep it under 120 words. Output only the enhanced prompt, no preamble."
)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None


def _load() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    logger.info("Loading prompt enhancer: %s", QWEN_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, cache_dir=str(HF_HOME))
    _model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float32,
        cache_dir=str(HF_HOME),
        low_cpu_mem_usage=True,
    )
    _model.eval()
    model_manager.register_unload(ModelId.ENHANCER, _unload)


def _unload() -> None:
    global _tokenizer, _model
    _model = None
    _tokenizer = None
    logger.info("Prompt enhancer unloaded")


def enhance_prompt(prompt: str, max_new_tokens: int = 150) -> str:
    """Expand a short user prompt into a cinematic description."""
    with model_manager.load(ModelId.ENHANCER):
        _load()
        assert _tokenizer is not None and _model is not None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        enhanced = _tokenizer.decode(generated, skip_special_tokens=True).strip()
        logger.info("Enhanced prompt: %s", enhanced[:120])
        return enhanced or prompt
