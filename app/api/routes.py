"""FastAPI routes for video generation jobs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import OUTPUTS_DIR, PRESETS
from app.pipeline.ensemble import GenerationRequest, JobState, ModelChoice, run_ensemble
from app.services.model_manager import get_vram_usage_mb

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# In-memory job store (single-user local app)
_jobs: dict[str, JobState] = {}
_job_lock = asyncio.Lock()
_queue: asyncio.Queue[str | None] = asyncio.Queue()
_worker_started = False


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    preset: str = "fast"
    model: ModelChoice = ModelChoice.AUTO
    seed: int = 42
    enhance: bool = True


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    enhanced_prompt: str | None = None
    video_url: str | None = None
    error: str | None = None


async def _process_job(job_id: str) -> None:
    job = _jobs[job_id]
    job.status = "running"
    job.message = "Starting pipeline"

    def on_progress(stage: str, pct: float) -> None:
        job.progress = pct
        job.message = stage.replace("_", " ").title()

    try:
        request = job.request
        if request is None:
            raise ValueError("Job has no request")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_ensemble, request, on_progress)
        job.result = result
        job.status = "completed"
        job.progress = 1.0
        job.message = "Done"
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"


async def _queue_worker() -> None:
    while True:
        job_id = await _queue.get()
        if job_id is None:
            break
        async with _job_lock:
            await _process_job(job_id)
        _queue.task_done()


async def start_worker() -> None:
    """Start background job worker (call from FastAPI lifespan)."""
    global _worker_started
    if not _worker_started:
        asyncio.create_task(_queue_worker())
        _worker_started = True
        logger.info("Job queue worker started")


@router.post("/generate", response_model=JobResponse)
async def generate(body: GenerateBody) -> JobResponse:
    if body.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {body.preset}. Choose from {list(PRESETS)}")

    job_id = str(uuid.uuid4())[:8]
    job = JobState(
        job_id=job_id,
        status="queued",
        message="Queued",
        request=GenerationRequest(
            prompt=body.prompt,
            preset=body.preset,
            model=body.model,
            seed=body.seed,
            enhance=body.enhance,
        ),
    )
    _jobs[job_id] = job
    await _queue.put(job_id)

    return JobResponse(job_id=job_id, status=job.status, progress=0, message=job.message)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    resp = JobResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error=job.error,
    )
    if job.result:
        resp.enhanced_prompt = job.result.enhanced_prompt
        resp.video_url = f"/api/jobs/{job_id}/video"
    return resp


@router.get("/jobs/{job_id}/video")
async def get_video(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "Video not ready")
    path: Path = job.result.video_path
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/health")
async def health() -> dict:
    import shutil
    from app.config import PROJECT_ROOT

    vram = get_vram_usage_mb()
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "status": "ok",
        "vram": vram,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "presets": list(PRESETS.keys()),
        "outputs_dir": str(OUTPUTS_DIR),
    }


@router.get("/presets")
async def list_presets() -> dict:
    return PRESETS
