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
import zipfile
from io import BytesIO
from pathlib import PurePosixPath

import pandas as pd
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

CSV_SUFFIXES = frozenset({".csv"})
EXCEL_SUFFIXES = frozenset({".xlsx", ".xls"})
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES

# An .xlsx is a zip archive, so its size on the wire says little about the
# work needed to parse it. The decompressed size is checked against this
# multiple of the byte limit before parsing begins. A 100,000-row workbook
# in the documented schema measured 6.2 MB on disk and 56.5 MB inflated, so
# the default (4 x 25 MB) leaves 1.8x headroom over a legitimate file at the
# row limit while refusing archives built to expand.
INFLATION_ALLOWANCE = 4


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


def _check_inflated_size(payload: bytes, limit: int) -> None:
    """Reject a workbook whose members decompress past ``limit`` bytes.

    Uses the sizes declared in the zip directory, which is safe to trust as
    a bound: :mod:`zipfile` will not inflate a member past its declared size
    without failing the CRC check, so an archive cannot lie its way past.
    """
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            inflated = sum(member.file_size for member in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise UploadError("The file is not a valid .xlsx workbook.") from exc

    if inflated > limit:
        raise UploadError(
            f"The workbook would decompress to {inflated / 1_048_576:.0f} MB, "
            f"which exceeds the {limit / 1_048_576:.0f} MB limit. "
            "Split the data across smaller files or upload it as CSV."
        )


def parse_bytes(
    payload: bytes,
    filename: str | None,
    *,
    max_rows: int | None = None,
    max_inflated_bytes: int | None = None,
) -> pd.DataFrame:
    """Parse raw upload bytes into a DataFrame based on the file extension.

    Args:
        payload: Complete file contents.
        filename: Client-supplied name, used only to select a parser.
        max_rows: Stop reading after one row more than this, so an oversized
            file costs no more to reject than a file at the limit. ``None``
            reads everything.
        max_inflated_bytes: Reject an .xlsx that would decompress past this
            before parsing it. ``None`` skips the check.

    Returns:
        The parsed records -- at most ``max_rows + 1`` of them when
        ``max_rows`` is given, so the caller can tell the limit was passed.

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

    nrows = None if max_rows is None else max_rows + 1

    if suffix in EXCEL_SUFFIXES and max_inflated_bytes is not None:
        _check_inflated_size(payload, max_inflated_bytes)

    try:
        if suffix in CSV_SUFFIXES:
            frame = pd.read_csv(BytesIO(payload), nrows=nrows)
        else:
            frame = pd.read_excel(BytesIO(payload), nrows=nrows)
    except zipfile.BadZipFile as exc:
        # Not a ValueError, so it used to escape as a 500.
        raise UploadError("The file is not a valid .xlsx workbook.") from exc
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

    frame = parse_bytes(
        payload,
        filename,
        max_rows=max_rows,
        max_inflated_bytes=max_bytes * INFLATION_ALLOWANCE,
    )

    # Reading stopped one row past the limit, so the true count is unknown
    # -- and deliberately so: learning it would mean parsing the whole file.
    if len(frame) > max_rows:
        raise UploadError(f"File contains more than {max_rows:,} rows, the row limit.")

    return frame


async def read_upload(
    file, max_bytes: int, max_rows: int, *, filename: str | None = None
) -> pd.DataFrame:
    """Read, size-check and parse a FastAPI ``UploadFile``.

    Args:
        file: The ``UploadFile`` to consume.
        max_bytes: Reject payloads larger than this. At most this many bytes
            (plus one) are read, so it is the per-request memory bound for the
            raw upload; parsing is bounded separately by ``max_rows`` and the
            inflation allowance.
        max_rows: Reject frames with more rows than this.
        filename: Overrides ``file.filename`` when supplied.

    Returns:
        The parsed records.

    Raises:
        UploadError: If the upload is oversized, unsupported or unparseable.
    """
    name = filename if filename is not None else getattr(file, "filename", None)

    # One byte past the limit is enough to know it was exceeded; reading the
    # rest would hold an arbitrarily large body in memory to reject it.
    payload = await file.read(max_bytes + 1)
    # Parsing is CPU-bound and synchronous -- about 10 s for a 100,000-row
    # workbook. On the event loop it stalled every other request, including
    # the health probe; in a worker thread it stalls only this one.
    frame = await run_in_threadpool(parse_within_limits, payload, name, max_bytes, max_rows)

    logger.info("Parsed upload %s: %d rows, %d columns", name, len(frame), len(frame.columns))
    return frame
