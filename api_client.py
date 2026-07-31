"""HTTP client the Streamlit dashboard uses to reach the scoring API.

The base URL comes from the ``CML_API_URL`` environment variable so the
dashboard can point at a container, a staging host or a local server
without editing code. The port previously moved between 8000 and 8002
across three commits precisely because it was hardcoded here.

This module deliberately raises instead of rendering errors: presentation
belongs to the dashboard, and keeping Streamlit out of here makes the
client testable and reusable from scripts.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_URL = "http://localhost:8000"
API_KEY_HEADER = "X-API-Key"
HEALTH_TIMEOUT_SECONDS = 5
SCORE_TIMEOUT_SECONDS = 120


class APIError(RuntimeError):
    """The API was unreachable or returned an error response."""


def get_api_base_url() -> str:
    """Return the configured API base URL, without a trailing slash."""
    return os.environ.get("CML_API_URL", DEFAULT_API_URL).rstrip("/")


def auth_headers() -> dict[str, str]:
    """Headers carrying the API key, or an empty dict when none is set.

    Public because the dashboard makes its own calls for model metadata
    and needs the same credentials.

    The API only requires it when its own API_KEY is set, so an
    unconfigured dashboard against an unconfigured API is unchanged.
    """
    api_key = os.environ.get("CML_API_KEY")
    return {API_KEY_HEADER: api_key} if api_key else {}


def check_api_health(timeout: int = HEALTH_TIMEOUT_SECONDS) -> bool:
    """Return True when the API answers its health probe."""
    try:
        response = requests.get(
            f"{get_api_base_url()}/health", timeout=timeout, headers=auth_headers()
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def score_cml_data(uploaded_file, timeout: int = SCORE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Score an uploaded CML file through the API.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (anything exposing
            ``name``, ``type`` and ``getvalue()`` works).
        timeout: Seconds to wait for the response.

    Returns:
        The decoded JSON response body.

    Raises:
        APIError: On connection failure, timeout, or a non-2xx response.
            The API's own error detail is included in the message when the
            body carries one.
    """
    base_url = get_api_base_url()
    mime_type = getattr(uploaded_file, "type", None) or "application/octet-stream"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}

    try:
        response = requests.post(f"{base_url}/score-cml-data", files=files, timeout=timeout)
    except requests.ConnectionError as exc:
        raise APIError(
            f"Cannot reach the API at {base_url}. Is it running? "
            f"Set CML_API_URL if it listens elsewhere."
        ) from exc
    except requests.Timeout as exc:
        raise APIError(f"The API did not respond within {timeout}s.") from exc
    except requests.RequestException as exc:
        raise APIError(f"Request to {base_url} failed: {exc}") from exc

    if response.status_code == 401:
        raise APIError(
            "The API rejected the request: it requires an API key. "
            "Set CML_API_KEY for the dashboard to match the API's API_KEY."
        )

    if not response.ok:
        raise APIError(f"API returned {response.status_code}: {_error_detail(response)}")

    return response.json()


def _error_detail(response: requests.Response) -> str:
    """Pull FastAPI's ``detail`` out of an error body, falling back to raw text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:500]
