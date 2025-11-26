from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import os
import shutil
import tempfile
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import AnyHttpUrl, BaseModel

try:
    import essentia.standard as es  # type: ignore[import]
except ImportError:  # Essentia may not be installed yet
    es = None  # type: ignore[assignment]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class JobWithResult(Job):
    result: Optional[Dict[str, Any]] = None


class Meta(BaseModel):
    name: str
    version: str
    license: str
    repo: str
    description: str


SERVICE_VERSION = "0.1.0"
REPO_URL = "https://github.com/your-org/essentia-analysis-service"
API_LICENSE = "AGPL-3.0-only"

app = FastAPI(
    title="Essentia Analysis Service",
    version=SERVICE_VERSION,
    description=(
        "Thin HTTP API wrapper around the Essentia audio analysis library. "
        "This service is intended to be published under the AGPL-3.0-only license."
    ),
)

# In-memory storage for demo purposes only
_JOBS: Dict[str, JobWithResult] = {}


def _extract_features(audio_path: str) -> Dict[str, Any]:
    """
    Minimal Essentia-based feature extraction.

    Replace / extend this with whatever Essentia pipeline you actually need.
    """
    if es is None:
        raise RuntimeError(
            "Essentia is not installed. Install `essentia` and its dependencies "
            "in this environment to enable feature extraction."
        )

    loader = es.MonoLoader(filename=audio_path)
    audio = loader()

    # Very simple example feature: total energy.
    energy = float(np.sum(np.square(audio)))

    return {
        "backend": "essentia",
        "features": {
            "energy": energy,
        },
    }


@app.post("/analyze", response_model=JobWithResult, tags=["analysis"])
async def analyze(
    source_url: Optional[AnyHttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> JobWithResult:
    """
    Start an analysis job.

    - Prefer `file` uploads for now; `source_url` is left as a future extension.
    """
    if source_url is None and file is None:
        raise HTTPException(status_code=400, detail="Provide either `file` or `source_url`.")

    job_id = str(uuid4())
    job = JobWithResult(
        job_id=job_id,
        status=JobStatus.running,
        created_at=datetime.utcnow(),
    )
    _JOBS[job_id] = job

    temp_path: Optional[str] = None
    try:
        if file is not None:
            suffix = os.path.splitext(file.filename or "")[1] or ".audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                shutil.copyfileobj(file.file, tmp)
        else:
            # You can implement URL downloading here using httpx/requests if desired.
            raise HTTPException(
                status_code=501,
                detail="`source_url` handling is not implemented yet. Upload a file instead.",
            )

        features = _extract_features(temp_path)

        job.status = JobStatus.completed
        job.completed_at = datetime.utcnow()
        job.result = features
        _JOBS[job_id] = job

        return job

    except HTTPException as http_exc:
        job.status = JobStatus.failed
        job.completed_at = datetime.utcnow()
        job.error = (
            http_exc.detail if isinstance(http_exc.detail, str) else str(http_exc.detail)
        )
        _JOBS[job_id] = job
        raise

    except Exception as exc:
        job.status = JobStatus.failed
        job.completed_at = datetime.utcnow()
        job.error = str(exc)
        _JOBS[job_id] = job
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/jobs/{job_id}", response_model=JobWithResult, tags=["analysis"])
async def get_job(job_id: str) -> JobWithResult:
    """Return status (and result, if available) for a given job."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/features/{job_id}", tags=["analysis"])
async def get_features(job_id: str) -> Dict[str, Any]:
    """Return just the feature payload for a completed job."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status is not JobStatus.completed or job.result is None:
        raise HTTPException(status_code=400, detail="Job is not completed yet.")
    return job.result


@app.get("/meta", response_model=Meta, tags=["meta"])
async def meta() -> Meta:
    """Service metadata, including license info and repo URL."""
    return Meta(
        name="essentia-analysis-service",
        version=SERVICE_VERSION,
        license=API_LICENSE,
        repo=REPO_URL,
        description="Audio feature extraction microservice built on Essentia.",
    )


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
