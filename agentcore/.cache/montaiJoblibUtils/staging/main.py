"""AgentCore entrypoint — starts the FastAPI server on port 8080.

AgentCore Runtime invokes this file directly when using the CodeZip build.
The server binds to 0.0.0.0:8080 as required by the HTTP protocol contract.

The package lives under src/ (editable install locally), so we add that
directory to sys.path before importing to cover the CodeZip runtime where
only the raw source tree is available.

Observability is handled by AgentCore natively via aws-opentelemetry-distro.
Enable it in the AgentCore console: Agent Runtime → Tracing → Enable.
Metrics (Invocations, Latency, Errors, CPU/Memory) are automatic.
"""

import os
import sys

_src = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

import uvicorn  # noqa: E402

from montai_joblib_utils_ai_mcp.api import app  # noqa: E402, F401  (re-exported for gunicorn)

if __name__ == "__main__":
    uvicorn.run(
        "montai_joblib_utils_ai_mcp.api:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
