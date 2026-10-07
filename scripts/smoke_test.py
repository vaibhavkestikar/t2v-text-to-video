"""Smoke tests for API, imports, and optional GPU inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example")


def test_imports() -> None:
    print("=== Test: imports ===")
    from diffusers import CogVideoXPipeline, LTXPipeline
    from app.services.model_manager import get_vram_usage_mb, model_manager
    from app.config import PRESETS
    print("  LTXPipeline:", LTXPipeline)
    print("  CogVideoXPipeline:", CogVideoXPipeline)
    print("  Presets:", list(PRESETS.keys()))
    print("  VRAM:", get_vram_usage_mb())
    print("  PASS")


def test_enhancer() -> None:
    print("=== Test: prompt enhancer ===")
    from app.services.enhancer import enhance_prompt
    from app.services.model_manager import model_manager

    result = enhance_prompt("A cat in a garden")
    assert len(result) > 10, "Enhancer returned empty prompt"
    print("  Enhanced:", result[:100], "...")
    model_manager.unload_current()
    print("  PASS")


def test_cogvideo() -> None:
    print("=== Test: CogVideoX generation ===")
    from app.pipeline.ensemble import GenerationRequest, ModelChoice, run_ensemble

    result = run_ensemble(
        GenerationRequest(
            prompt="An orange cat walking slowly through green grass",
            preset="quality",
            model=ModelChoice.COGVIDEO,
            seed=42,
            enhance=False,
        )
    )
    assert result.video_path.exists(), f"Video missing: {result.video_path}"
    print("  Video:", result.video_path)
    print("  PASS")


def test_ltx_fast() -> None:
    print("=== Test: LTX fast generation ===")
    from app.pipeline.ensemble import GenerationRequest, ModelChoice, run_ensemble

    result = run_ensemble(
        GenerationRequest(
            prompt="A orange cat walking slowly through green grass",
            preset="fast",
            model=ModelChoice.LTX,
            seed=42,
            enhance=False,
        )
    )
    assert result.video_path.exists(), f"Video missing: {result.video_path}"
    print("  Video:", result.video_path)
    print("  PASS")


def test_ensemble() -> None:
    print("=== Test: full ensemble pipeline ===")
    from app.pipeline.ensemble import GenerationRequest, run_ensemble

    result = run_ensemble(
        GenerationRequest(
            prompt="Sunset over ocean waves",
            preset="fast",
            seed=7,
            enhance=True,
        )
    )
    assert result.video_path.exists()
    assert result.enhanced_prompt != result.original_prompt
    print("  Enhanced:", result.enhanced_prompt[:80], "...")
    print("  Video:", result.video_path)
    print("  PASS")


def test_api_health() -> None:
    print("=== Test: API health ===")
    import httpx
    from app.config import API_HOST, API_PORT

    try:
        resp = httpx.get(f"http://{API_HOST}:{API_PORT}/api/health", timeout=5)
        resp.raise_for_status()
        print("  Health:", resp.json())
        print("  PASS")
    except httpx.ConnectError:
        print("  SKIP (API not running — start with: python -m app.main)")


def test_preflight() -> None:
    print("=== Test: preflight ===")
    from app.services.preflight import run_preflight
    issues = run_preflight()
    if issues:
        print("  Warnings:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  All checks passed")
    print("  PASS")


if __name__ == "__main__":
    tests = sys.argv[1:] or ["imports", "enhancer", "preflight", "api"]
    runners = {
        "imports": test_imports,
        "enhancer": test_enhancer,
        "cogvideo": test_cogvideo,
        "ltx": test_ltx_fast,
        "ensemble": test_ensemble,
        "preflight": test_preflight,
        "api": test_api_health,
    }
    for name in tests:
        if name in runners:
            runners[name]()
        else:
            print(f"Unknown test: {name}")
