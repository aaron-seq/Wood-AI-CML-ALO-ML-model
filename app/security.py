"""Optional API-key authentication.

Disabled by default. With ``API_KEY`` unset the dependencies below are
no-ops, so an existing deployment behaves exactly as before; setting the
variable is what turns enforcement on.

This is a shared-secret gate, not an identity system. It answers "is this
caller allowed to reach the API", not "who is this caller", so the
``sme_name`` recorded on an override is still self-declared. Attributing
overrides to a verified identity needs real authentication in front --
see docs/DEPLOYMENT.md.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


def _reject(request: Request, reason: str) -> None:
    """Log the rejection with correlation, then fail the request.

    The response body never says *why* the key was wrong -- missing and
    incorrect are reported identically, so a caller cannot use the error
    to probe for valid keys. The distinction is in the server log.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "Rejected request to %s (request_id=%s): %s", request.url.path, request_id, reason
    )
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        f"A valid {API_KEY_HEADER} header is required.",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def _check(request: Request, supplied: str | None) -> None:
    if settings.API_KEY is None:
        return
    if supplied is None:
        _reject(request, f"no {API_KEY_HEADER} header supplied")
    # compare_digest rather than ==: a plain comparison returns as soon as
    # it finds a differing byte, which leaks the key one byte at a time to
    # anyone who can time the response.
    if not secrets.compare_digest(supplied or "", settings.API_KEY):
        _reject(request, f"incorrect {API_KEY_HEADER} value")


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Guard a write endpoint.

    Enforced whenever ``API_KEY`` is set, at either scope.
    """
    _check(request, x_api_key)


async def require_api_key_for_reads(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """Guard a read or scoring endpoint.

    Enforced only when ``API_KEY_SCOPE`` is ``all``. The default scope,
    ``writes``, leaves scoring and reporting open so an internal
    dashboard keeps working while the endpoints that mutate the audit
    trail are protected.
    """
    if settings.API_KEY_SCOPE != "all":
        return
    _check(request, x_api_key)
