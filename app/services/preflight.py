"""Pre-flight checks before loading heavy T2V models."""

from __future__ import annotations

import shutil

from app.config import PROJECT_ROOT
from app.services.model_manager import get_vram_usage_mb

MIN_C_DRIVE_GB = 10.0
MIN_D_DRIVE_GB = 1.0
MIN_VRAM_MB = 4500


def run_preflight() -> list[str]:
    """Return list of warnings/errors. Empty list means OK."""
    issues: list[str] = []

    c_usage = shutil.disk_usage("C:\\")
    c_free = c_usage.free / (1024**3)
    if c_free < MIN_C_DRIVE_GB:
        issues.append(
            f"C: drive has only {c_free:.1f} GB free. Windows page file needs ~15+ GB "
            "to load T5 text encoders. Free C: space or move the page file to D:."
        )

    d_usage = shutil.disk_usage(PROJECT_ROOT)
    d_free = d_usage.free / (1024**3)
    if d_free < MIN_D_DRIVE_GB:
        issues.append(f"Project drive has only {d_free:.1f} GB free (need {MIN_D_DRIVE_GB}+ GB).")

    vram = get_vram_usage_mb()
    total = vram.get("total_mb")
    if total and total < MIN_VRAM_MB:
        issues.append(f"GPU VRAM {total:.0f} MB is below recommended {MIN_VRAM_MB} MB.")

    return issues


def require_preflight() -> None:
    issues = run_preflight()
    if issues:
        raise RuntimeError("Pre-flight check failed:\n- " + "\n- ".join(issues))
