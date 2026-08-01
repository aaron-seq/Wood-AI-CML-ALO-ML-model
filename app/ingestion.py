"""Validation and parsing of uploaded CML files.

Every upload endpoint routes through :func:`read_upload`, so the size
limit, the format allow-list and the error contract are defined once
instead of being re-implemented (and diverging) per endpoint.

Rejections raise :class:`UploadError`, which the API translates into a
``400 Bad Request``. Previously each endpoint raised ``HTTPException``
inside a ``try`` block whose own ``except Exception`` caught it and
re-raised it as a ``500``, so every client-side mistake was reported as a
server fault.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import PurePosixPath

import pandas as pd

logger = logging.getLogger(__name__)

CSV_SUFFIXES = frozenset({".csv"})
EXCEL_SUFFIXES = frozenset({".xlsx", ".xls"})
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES


class UploadError(ValueError):
    """An uploaded file was missing, too large, or could not be parsed."""


def _safe_suffix(filename: str | None) -> str:
    """Return the lower-cased extension of a client-supplied filename.

    The name is treated as untrusted: only the final path component is
    considered, so a caller cannot smuggle a directory traversal through
    the ``filename`` field.
    """
    if not filename:
        raise UploadError("No filename was supplied with the upload.")
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()


def parse_bytes(payload: bytes, filename: str | None) -> pd.DataFrame:
    """Parse raw upload bytes into a DataFrame based on the file extension.

    Args:
        payload: Complete file contents.
        filename: Client-supplied name, used only to select a parser.

    Returns:
        The parsed records.

    Raises:
        UploadError: If the format is unsupported or the file is unreadable.
    """
    suffix = _safe_suffix(filename)

    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadError(
            f"Unsupported file format '{suffix or filename}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if not payload:
        raise UploadError("The uploaded file is empty.")

    try:
        if suffix in CSV_SUFFIXES:
            frame = pd.read_csv(BytesIO(payload))
        else:
            frame = pd.read_excel(BytesIO(payload))
    except UnicodeDecodeError as exc:
        raise UploadError(
            "The file could not be decoded as text. Save it as UTF-8 CSV or as .xlsx."
        ) from exc
    except ValueError as exc:
        # pandas raises ValueError/EmptyDataError for headerless, empty and
        # structurally broken files. These are caller mistakes, not faults.
        raise UploadError(f"The file could not be parsed: {exc}") from exc

    if frame.empty:
        raise UploadError("The uploaded file contains no data rows.")

    return frame


def parse_within_limits(
    payload: bytes, filename: str | None, max_bytes: int, max_rows: int
) -> pd.DataFrame:
    """Size-check and parse upload bytes.

    The whole upload contract in one synchronous function, so the API and
    the dashboard enforce identical limits and produce identical messages.
    The dashboard previously called pandas directly and enforced nothing.

    Raises:
        UploadError: If the upload is oversized, unsupported or unparseable.
    """
    if len(payload) > max_bytes:
        raise UploadError(
            f"File is too large ({len(payload) / 1_048_576:.1f} MB). "
            f"The limit is {max_bytes / 1_048_576:.0f} MB."
        )

    frame = parse_bytes(payload, filename)

    if len(frame) > max_rows:
        raise UploadError(
            f"File contains {len(frame):,} rows, which exceeds the {max_rows:,} row limit."
        )

    return frame


async def read_upload(
    file, max_bytes: int, max_rows: int, *, filename: str | None = None
) -> pd.DataFrame:
    """Read, size-check and parse a FastAPI ``UploadFile``.

    Args:
        file: The ``UploadFile`` to consume.
        max_bytes: Reject payloads larger than this. Uploads are parsed in
            memory, so this is the effective per-request memory bound.
        max_rows: Reject frames with more rows than this.
        filename: Overrides ``file.filename`` when supplied.

    Returns:
        The parsed records.

    Raises:
        UploadError: If the upload is oversized, unsupported or unparseable.
    """
    name = filename if filename is not None else getattr(file, "filename", None)

    payload = await file.read()
    frame = parse_within_limits(payload, name, max_bytes, max_rows)

    logger.info("Parsed upload %s: %d rows, %d columns", name, len(frame), len(frame.columns))
    return frame
