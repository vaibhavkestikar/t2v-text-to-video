"""Sequential model load/unload with VRAM monitoring."""

from __future__ import annotations

import gc
import logging
import subprocess
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Iterator

import torch

from app.config import MAX_VRAM_MB

logger = logging.getLogger(__name__)


class ModelId(str, Enum):
    ENHANCER = "enhancer"
    LTX = "ltx"
    COGVIDEO = "cogvideo"
    UPSCALER = "upscaler"


def get_vram_usage_mb() -> dict[str, float | None]:
    """Return GPU memory stats in MB via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        used, total, free = [float(x.strip()) for x in result.stdout.strip().split(",")]
        return {"used_mb": used, "total_mb": total, "free_mb": free}
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024**2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            return {"used_mb": used, "total_mb": total, "free_mb": total - used}
        return {"used_mb": None, "total_mb": None, "free_mb": None}


def clear_gpu_memory() -> None:
    """Force release of GPU memory between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def check_vram_guard(max_mb: int = MAX_VRAM_MB) -> bool:
    """Return True if VRAM usage is below the guard threshold."""
    stats = get_vram_usage_mb()
    used = stats.get("used_mb")
    if used is None:
        return True
    if used > max_mb:
        logger.warning("VRAM guard triggered: %.0f MB used (limit %d MB)", used, max_mb)
        return False
    return True


class ModelManager:
    """Ensures only one heavy model resides on GPU at a time."""

    def __init__(self) -> None:
        self._loaded: ModelId | None = None
        self._unload_callbacks: dict[ModelId, Callable[[], None]] = {}

    def register_unload(self, model_id: ModelId, callback: Callable[[], None]) -> None:
        self._unload_callbacks[model_id] = callback

    @property
    def loaded_model(self) -> ModelId | None:
        return self._loaded

    def unload_current(self) -> None:
        if self._loaded is None:
            return
        logger.info("Unloading model: %s", self._loaded.value)
        callback = self._unload_callbacks.get(self._loaded)
        if callback:
            callback()
        self._loaded = None
        clear_gpu_memory()
        stats = get_vram_usage_mb()
        logger.info("VRAM after unload: %s MB used", stats.get("used_mb"))

    @contextmanager
    def load(self, model_id: ModelId) -> Iterator[None]:
        """Context manager — unloads previous model before yielding."""
        if self._loaded is not None and self._loaded != model_id:
            self.unload_current()
        self._loaded = model_id
        logger.info("Loading model: %s (VRAM: %s MB)", model_id.value, get_vram_usage_mb().get("used_mb"))
        try:
            yield
        finally:
            pass  # Caller decides when to unload via unload_current()


# Singleton used across the app
model_manager = ModelManager()
