from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import os
import json
import shutil
import tempfile
from uuid import uuid4

import math
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
ANALYZER_NAME = (
    f"essentia-{getattr(es, '__version__', '2.1-beta5')}" if es is not None else "essentia-missing"
)
TF_MODEL_PATH = os.getenv("ESSENTIA_TF_MODEL")
TF_LABELS_PATH = os.getenv("ESSENTIA_TF_MODEL_LABELS")

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


def _extract_features(audio_path: str, track_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Rich Essentia-based feature extraction using the built-in MusicExtractor.

    Returns a payload shaped for the external API contract (tempo, key, MFCCs, tags, etc.).
    """
    if es is None:
        raise RuntimeError(
            "Essentia is not installed. Install `essentia` and its dependencies "
            "in this environment to enable feature extraction."
        )

    # Load raw audio for RMS/energy and TensorFlow tagging
    raw_loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    raw_audio = raw_loader()
    tf_audio: Optional[List[float]] = None
    if TF_MODEL_PATH:
        tf_loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
        tf_audio = list(tf_loader())

    # Run Essentia's high-level extractor (computes lowlevel, rhythm, tonal, and highlevel)
    music_extractor = es.MusicExtractor(
        lowlevelStats=["mean", "var"],
        rhythmStats=["mean", "var"],
        tonalStats=["mean", "var"],
        analysisSampleRate=44100,
    )
    pool, audio = music_extractor(audio_path)
    descriptor_names = set(pool.descriptorNames())

    def _get(key: str, default: Any = None) -> Any:
        try:
            return pool[key]
        except Exception:
            return default

    def _flatten_numeric(value: Any) -> List[float]:
        """Return a flat list of floats, best-effort, ignoring non-numerics."""
        out: List[float] = []

        def _walk(v: Any) -> None:
            if v is None:
                return
            if isinstance(v, (float, int, np.floating, np.integer)):
                try:
                    out.append(float(v))
                except Exception:
                    return
                return
            # Skip plain strings/bytes
            if isinstance(v, (str, bytes)):
                return
            if isinstance(v, (list, tuple)):
                for item in v:
                    _walk(item)
                return
            if isinstance(v, np.ndarray):
                for item in v.ravel():
                    _walk(item)
                return
            if isinstance(v, dict):
                for item in v.values():
                    _walk(item)
                return
            # Generic iterable (e.g., Essentia vector types)
            if hasattr(v, "__iter__"):
                try:
                    for item in v:
                        _walk(item)
                    return
                except Exception:
                    pass
            try:
                out.append(float(v))
            except Exception:
                return

        _walk(value)
        return out

    def _to_scalar(value: Any, default: float = 0.0) -> float:
        vals = _flatten_numeric(value)
        if not vals:
            return float(default)
        return float(vals[0])

    # Helper functions for shaping values
    def _safe_mean(values: Any) -> float:
        arr = _flatten_numeric(values)
        return float(sum(arr) / len(arr)) if len(arr) else 0.0

    def _swing(beats: List[float]) -> float:
        if len(beats) < 3:
            return 0.0
        intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
        odd = intervals[0::2]
        even = intervals[1::2]
        if not odd or not even:
            return 0.0
        odd_mean = sum(odd) / len(odd)
        even_mean = sum(even) / len(even)
        denom = max((odd_mean + even_mean) / 2.0, 1e-9)
        return float(abs(odd_mean - even_mean) / denom)

    def _top_labels(probabilities: Dict[str, float], k: int = 2) -> List[str]:
        return [label for label, _ in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    # Core scalar features
    tempo = _to_scalar(_get("rhythm.bpm", 0.0))
    beat_strength = _safe_mean(_flatten_numeric(_get("rhythm.beats_loudness", [])))
    swing = _swing(_flatten_numeric(_get("rhythm.beats_position", [])))
    danceability = _to_scalar(_get("rhythm.danceability", 0.0))

    key_key = _get("tonal.key_key", "C")
    key_scale = _get("tonal.key_scale", "major")
    key_strength = _to_scalar(_get("tonal.key_strength", 0.0))
    key_map = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    key = key_map.get(str(key_key), 0)
    mode = 1 if str(key_scale).lower() == "major" else 0

    hpcp_mean = [float(x) for x in _flatten_numeric(_get("tonal.hpcp.mean", []))]
    mfcc_mean = [float(x) for x in _flatten_numeric(_get("lowlevel.mfcc.mean", []))]
    mfcc_var = [float(x) for x in _flatten_numeric(_get("lowlevel.mfcc.var", []))]

    spectral_centroid = _to_scalar(_get("lowlevel.spectral_centroid.mean", 0.0))
    brightness = _to_scalar(
        _get("lowlevel.spectral_energyband_high.mean", spectral_centroid)
    )

    # Loudness and energy from raw audio
    audio_vals = _flatten_numeric(raw_audio)
    sum_sq = sum(x * x for x in audio_vals)
    rms = float(math.sqrt(sum_sq / len(audio_vals)) if audio_vals else 0.0)
    rms_loudness = float(20 * math.log10(max(rms, 1e-12)))  # dBFS-ish scalar
    dynamic_range = _to_scalar(_get("dynamic_complexity", 0.0))
    energy = float(sum_sq)

    # Tensorflow model-based tags (musicnn-style)
    def _tf_top_tags(audio_seq: Optional[List[float]]) -> Dict[str, Any]:
        if es is None or TF_MODEL_PATH is None or audio_seq is None:
            return {"labels": [], "scores": [], "mood_labels": [], "shape": None, "error": None}
        if not os.path.exists(TF_MODEL_PATH):
            return {"labels": [], "scores": [], "mood_labels": [], "shape": None, "error": "model_not_found"}
        if not (hasattr(es, "TensorflowPredictMusiCNN") or hasattr(es, "TensorflowPredict")):
            return {
                "labels": [],
                "scores": [],
                "mood_labels": [],
                "shape": None,
                "error": "tensorflow_not_available_in_this_essentia_build",
            }
        try:
            if hasattr(es, "TensorflowPredictMusiCNN"):
                tf_predict = es.TensorflowPredictMusiCNN(
                    graphFilename=TF_MODEL_PATH,
                    output="model/Sigmoid",
                )
            else:
                tf_predict = es.TensorflowPredict(
                    graphFilename=TF_MODEL_PATH,
                    input="model/Placeholder",
                    output="model/Sigmoid",
                    m2f="melspectrogram",
                )

            activations = tf_predict(np.asarray(audio_seq, dtype=np.float32))
            act_arr = np.asarray(activations, dtype=np.float32)
            if act_arr.ndim > 1:
                act_arr = act_arr.mean(axis=0)
            labels: List[str] = []
            if TF_LABELS_PATH and os.path.exists(TF_LABELS_PATH):
                if TF_LABELS_PATH.endswith(".json"):
                    with open(TF_LABELS_PATH, "r", encoding="utf-8") as f:
                        try:
                            payload = json.load(f)
                            if isinstance(payload, dict):
                                if "labels" in payload:
                                    payload = payload["labels"]
                                elif "classes" in payload:
                                    payload = payload["classes"]
                            if isinstance(payload, list):
                                labels = [str(x) for x in payload]
                        except Exception:
                            labels = []
                else:
                    with open(TF_LABELS_PATH, "r", encoding="utf-8") as f:
                        labels = [line.strip() for line in f if line.strip()]
            if not labels:
                labels = [f"tag_{i}" for i in range(act_arr.shape[-1])]

            all_scores = []
            for i, lbl in enumerate(labels):
                if i >= act_arr.shape[0]:
                    break
                all_scores.append({"label": lbl, "score": float(act_arr[i])})

            top_idx = np.argsort(act_arr)[::-1][:5]
            top_labels = [labels[i] for i in top_idx if i < len(labels)]
            top_scores = [float(act_arr[i]) for i in top_idx if i < len(labels)]

            # Mood-oriented labels across all outputs above a threshold
            mood_keywords = {
                "happy",
                "sad",
                "mellow",
                "chill",
                "chillout",
                "beautiful",
                "sexy",
                "party",
                "energetic",
                "dark",
                "aggressive",
                "relaxing",
                "calm",
                "uplifting",
                "melancholic",
                "romantic",
                "catchy",
                "easy listening",
            }
            mood_candidates: List[Tuple[str, float]] = []
            for i, lbl in enumerate(labels):
                if i >= act_arr.shape[0]:
                    continue
                score = float(act_arr[i])
                if lbl.lower().strip() in mood_keywords:
                    mood_candidates.append((lbl, score))

            # Always surface the top few mood-like labels, even if scores are low
            mood_candidates.sort(key=lambda x: x[1], reverse=True)
            mood_top = mood_candidates[:3]
            mood_labels = [lbl for lbl, _ in mood_top]
            mood_scores = [score for _, score in mood_top]

            return {
                "labels": top_labels,
                "scores": top_scores,
                "mood_labels": mood_labels,
                "mood_scores": mood_scores,
                "all_scores": all_scores,
                "shape": list(act_arr.shape),
                "error": None,
            }
        except Exception as exc:
            return {
                "labels": [],
                "scores": [],
                "mood_labels": [],
                "mood_scores": [],
                "all_scores": [],
                "shape": None,
                "error": str(exc),
            }

    # Tags from highlevel classifiers (if available)
    tags: Dict[str, Any] = {"genres": [], "moods": [], "vocals": None, "instruments": []}

    # Collect top labels from any highlevel.* descriptors that expose labels/probability
    for name in descriptor_names:
        if not name.startswith("highlevel.") or not name.endswith(".labels"):
            continue
        base = name[: -len(".labels")]
        labels = _get(name, [])
        probs = _get(f"{base}.probability", [])
        if not labels:
            continue
        label_list = [str(lbl) for lbl in labels]
        prob_list: List[float]
        if isinstance(probs, dict):
            prob_list = [float(probs.get(lbl, 0.0)) for lbl in label_list]
        else:
            flat_probs = _flatten_numeric(probs)
            if len(flat_probs) >= len(label_list):
                prob_list = [float(p) for p in flat_probs[: len(label_list)]]
            else:
                prob_list = [float(p) for p in flat_probs] + [0.0] * (
                    len(label_list) - len(flat_probs)
                )
        pairs = list(zip(label_list, prob_list))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_labels = [lbl for lbl, _ in pairs[:2]]
        if "genre" in base:
            tags["genres"].extend(top_labels)
        elif "mood" in base:
            tags["moods"].extend(top_labels)
        elif "voice_instrumental" in base:
            tags["vocals"] = "voice" in top_labels
        else:
            tags["instruments"].extend(top_labels)

    # Voice/instrumental boolean fallback from value if available
    vi_val = _get("highlevel.voice_instrumental.value", None)
    if vi_val is not None:
        tags["vocals"] = str(vi_val) == "voice"

    # TensorFlow musicnn-style top tags appended to genres bucket
    tf_tags = _tf_top_tags(tf_audio)
    if tf_tags["labels"]:
        tags["genres"].extend(tf_tags["labels"])
    if tf_tags.get("mood_labels"):
        tags["moods"].extend(tf_tags["mood_labels"])
    tf_debug = {
        "tf_model": TF_MODEL_PATH,
        "tf_labels_file": TF_LABELS_PATH,
        "tf_scores": tf_tags["scores"],
        "tf_mood_labels": tf_tags.get("mood_labels", []),
        "tf_mood_scores": tf_tags.get("mood_scores", []),
        "tf_all_scores": tf_tags.get("all_scores", []),
        "tf_shape": tf_tags.get("shape"),
        "tf_error": tf_tags.get("error"),
    }

    # Deduplicate while keeping order
    tags["genres"] = list(dict.fromkeys(tags["genres"]))
    tags["moods"] = list(dict.fromkeys(tags["moods"]))
    tags["instruments"] = list(dict.fromkeys(tags["instruments"]))

    # Simple embedding built from stacked descriptors (real values, padded/truncated to 128 dims)
    emb_components: List[float] = (
        hpcp_mean
        + mfcc_mean
        + mfcc_var
        + [
            spectral_centroid,
            brightness,
            rms_loudness,
            dynamic_range,
            energy,
            tempo,
            beat_strength,
            swing,
            danceability,
            key_strength,
            float(mode),
            float(key),
        ]
    )
    if len(emb_components) >= 128:
        embedding_list = emb_components[:128]
    else:
        embedding_list = emb_components + [0.0] * (128 - len(emb_components))

    return {
        "trackId": track_id or os.path.splitext(os.path.basename(audio_path))[0],
        "analyzer": ANALYZER_NAME,
        "features": {
            "tempo": tempo,
            "beat_strength": beat_strength,
            "swing": swing,
            "key": key,
            "mode": mode,
            "key_strength": key_strength,
            "hpcp_mean": hpcp_mean,
            "mfcc_mean": mfcc_mean,
            "mfcc_var": mfcc_var,
            "spectral_centroid": spectral_centroid,
            "brightness": brightness,
            "rms_loudness": rms_loudness,
            "dynamic_range": dynamic_range,
            "energy": energy,
            "danceability": danceability,
            "tags": tags,
            "debug": tf_debug,
            "embedding": [float(x) for x in embedding_list],
        },
    }


@app.post("/analyze", response_model=JobWithResult, tags=["analysis"])
async def analyze(
    source_url: Optional[AnyHttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None),
    track_id: Optional[str] = Form(None),
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

        features = _extract_features(temp_path, track_id=track_id)

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
