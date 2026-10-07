"""Gradio web UI for the local T2V generator."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example")

import gradio as gr
import httpx

from app.config import API_HOST, API_PORT, GRADIO_PORT, PRESETS

API_BASE = f"http://{API_HOST}:{API_PORT}"


def _poll_job(client: httpx.Client, job_id: str) -> dict:
    """Poll job status until complete or failed."""
    while True:
        resp = client.get(f"{API_BASE}/api/jobs/{job_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(2)


def generate_video(
    prompt: str,
    preset: str,
    model: str,
    seed: int,
    enhance: bool,
    progress=gr.Progress(),
) -> tuple[str | None, str, str]:
    """Submit generation job and poll for result."""
    if not prompt.strip():
        return None, "", "Please enter a prompt."

    progress(0.05, desc="Submitting job...")
    try:
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                f"{API_BASE}/api/generate",
                json={
                    "prompt": prompt,
                    "preset": preset,
                    "model": model,
                    "seed": seed,
                    "enhance": enhance,
                },
            )
            resp.raise_for_status()
            job = resp.json()
            job_id = job["job_id"]

            progress(0.1, desc="Generating...")
            while True:
                status_resp = client.get(f"{API_BASE}/api/jobs/{job_id}", timeout=30)
                status_resp.raise_for_status()
                status = status_resp.json()
                progress(status["progress"], desc=status["message"])

                if status["status"] == "completed":
                    video_resp = client.get(
                        f"{API_BASE}/api/jobs/{job_id}/video",
                        timeout=60,
                    )
                    video_resp.raise_for_status()
                    video_path = ROOT / "outputs" / f"gradio_{job_id}.mp4"
                    video_path.write_bytes(video_resp.content)
                    enhanced = status.get("enhanced_prompt", "")
                    return str(video_path), enhanced, f"Done — job {job_id}"

                if status["status"] == "failed":
                    return None, "", f"Failed: {status.get('error', 'unknown error')}"

                time.sleep(2)

    except httpx.ConnectError:
        return None, "", f"Cannot connect to API at {API_BASE}. Start the API first: python -m app.main"
    except Exception as exc:
        return None, "", f"Error: {exc}"


def build_ui() -> gr.Blocks:
    preset_choices = list(PRESETS.keys())
    with gr.Blocks(title="Local Text-to-Video Generator") as demo:
        gr.Markdown("# Local Text-to-Video Generator")
        gr.Markdown(
            "Ensemble pipeline: **Qwen2.5 prompt enhancer** → **LTX-Video 2B** (fast) / **CogVideoX-2B** (quality) → optional upscale. "
            "Optimized for RTX 4050 6GB."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A cat walking through a sunlit garden...",
                    lines=3,
                )
                with gr.Row():
                    preset = gr.Dropdown(preset_choices, value="fast", label="Preset")
                    model = gr.Dropdown(
                        ["auto", "ltx", "cogvideo"],
                        value="auto",
                        label="Model",
                    )
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    enhance = gr.Checkbox(value=True, label="Enhance prompt")
                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=2):
                video_out = gr.Video(label="Generated Video")
                enhanced_out = gr.Textbox(label="Enhanced Prompt", lines=3)
                status_out = gr.Textbox(label="Status", lines=2)

        generate_btn.click(
            fn=generate_video,
            inputs=[prompt, preset, model, seed, enhance],
            outputs=[video_out, enhanced_out, status_out],
        )

        gr.Markdown(
            f"**API:** `{API_BASE}` | **Presets:** fast (512×288), balanced (704×480), quality (CogVideoX), upscaled"
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT, share=False)
