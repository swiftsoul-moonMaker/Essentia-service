# Essentia Analysis Service

A small FastAPI-based HTTP service that wraps the
[Essentia](https://essentia.upf.edu/) audio analysis library and exposes
its results over a simple JSON API.

This service is designed to be deployed as a stand-alone container or
microservice and then called from your main application (for example,
from a backend that forwards audio to analyzers and polls for job
status).

## Quickstart

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Make sure Essentia and its native dependencies are correctly
   installed on your platform (see Essentia's documentation for
   platform-specific notes).
4. Run the API locally:

   uvicorn app.main:app --reload

5. Open http://127.0.0.1:8000/docs to explore the automatically
   generated OpenAPI/Swagger UI.

## API endpoints

All responses are JSON.

| Method | Path                 | Description                                           |
|--------|----------------------|-------------------------------------------------------|
| POST   | `/analyze`           | Upload an audio file and start an analysis job.      |
| GET    | `/jobs/{job_id}`     | Get status and (if ready) the result for a job.      |
| GET    | `/features/{job_id}` | Return only the feature payload for a job.           |
| GET    | `/meta`              | Service metadata, repo URL, and license information. |
| GET    | `/health`            | Basic health check for monitoring.                   |

### POST `/analyze`

Accepts `multipart/form-data` with:

- `file`: binary audio file (mp3, wav, flac, etc.).
- `source_url` (optional, string): placeholder for URL-based ingestion;
  not yet implemented in the starter code.

Returns a JSON object representing the job, including a `job_id` and a
`status` field. In this starter stack the analysis runs synchronously,
so successful requests typically return with `status: "completed"`.

### GET `/jobs/{job_id}`

Returns the full job object:

- `job_id`
- `status` (`queued`, `running`, `completed`, `failed`)
- `created_at`, `completed_at`
- `error` (if any)
- `result` (feature payload, if available)

### GET `/features/{job_id}`

Returns just the `result` field for a completed job. If the job is
missing or not yet completed, an error is returned.

### GET `/meta`

Returns metadata about the running service, including:

- service name and version
- API license identifier
- GitHub repository URL (configure `REPO_URL` in `app/main.py`)
- short description

Use this endpoint in your UI or docs to clearly advertise where users
can find the source code for the service.

### GET `/health`

Returns a simple JSON object indicating that the process is running.
Useful for Kubernetes liveness / readiness checks or uptime monitors.

## License

This API implementation is intended to be released under the
**GNU Affero General Public License v3.0-only (AGPL-3.0-only)**, in line
with Essentia's own AGPLv3 licensing.

Running this service over a network means users interact with Essentia
through it; under the AGPL you must provide those users with access to
the complete corresponding source code of this service. Hosting this
repository publicly (for example on GitHub) and linking it from the
`/meta` endpoint is one practical way to honor that obligation.

See the `LICENSE` file for additional details and make sure to consult
your own legal counsel for production use.

## Notes

- The `_extract_features` function in `app/main.py` is deliberately
  minimal and only computes a simple energy feature. Extend this with
  your actual Essentia-based feature extraction pipeline.
- In-memory job storage is fine for development; for production, replace
  `_JOBS` with a durable store (SQL/NoSQL/Redis, etc.).
- If you intend to consume this from an existing app, configure the base
  URL and authentication (if any) in that app's backend and call the
  endpoints above as needed.
