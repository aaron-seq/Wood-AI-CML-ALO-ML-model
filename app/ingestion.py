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
from itertools import islice
from pathlib import PurePosixPath
from xml.etree.ElementTree import ParseError

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

CSV_SUFFIXES = frozenset({".csv"})
# .xlsx only. .xls was listed, but no .xls reader is installed, so every
# legacy workbook was accepted by name and then failed to parse.
EXCEL_SUFFIXES = frozenset({".xlsx"})
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES

# An .xlsx is a zip archive, so its size on the wire says little about the
# work needed to parse it. The decompressed size is checked against this
# multiple of the byte limit before parsing begins. A 100,000-row workbook
# in the documented schema measured 6.2 MB on disk and 56.5 MB inflated, so
# the default (4 x 25 MB) leaves 1.8x headroom over a legitimate file at the
# row limit while refusing archives built to expand.
INFLATION_ALLOWANCE = 4

# The documented schema has twelve columns. Rows were bounded but width was
# not: an 889 KB CSV with 100,000 columns parsed at +120 MB, and memory grows
# with the column count. Checked on the raw header before pandas allocates.
MAX_COLUMNS = 256

# A workbook's shared-strings table is parsed in full when it opens, before
# any row limit can apply, and costs about 20x more per byte than sheet XML:
# a 129 KB file whose table fit the whole-archive allowance took 20 s and
# 233 MB. 16 MiB holds 100,000 unique 40-character ids several times over.
MAX_SHARED_STRINGS_BYTES = 16 * 1024 * 1024
SHARED_STRINGS_PART = "xl/sharedStrings.xml"

# A spreadsheet has a few dozen parts; thousands are only useful to an attacker.
MAX_ARCHIVE_MEMBERS = 1000

# What openpyxl raises for a zip that is not a readable workbook. Mapped to a
# 400: each used to escape as a 500.
_WORKBOOK_ERRORS = (zipfile.BadZipFile, KeyError, OSError, ParseError, InvalidFileException)


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


def _check_archive(payload: bytes, limit: int) -> None:
    """Reject a workbook whose parts would be too costly to open.

    Uses the sizes declared in the zip directory, which is safe to trust as
    a bound: :mod:`zipfile` will not inflate a member past its declared size
    without failing the CRC check, so an archive cannot lie its way past.
    """
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise UploadError("The file is not a valid .xlsx workbook.") from exc

    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise UploadError(
            f"The workbook has {len(members):,} parts; at most {MAX_ARCHIVE_MEMBERS:,} are accepted."
        )

    inflated = sum(member.file_size for member in members)
    if inflated > limit:
        raise UploadError(
            f"The workbook would decompress to {inflated / 1_048_576:.0f} MB, "
            f"which exceeds the {limit / 1_048_576:.0f} MB limit. "
            "Split the data across smaller files or upload it as CSV."
        )

    shared = sum(m.file_size for m in members if m.filename == SHARED_STRINGS_PART)
    if shared > MAX_SHARED_STRINGS_BYTES:
        raise UploadError(
            f"The workbook's shared strings table is {shared / 1_048_576:.1f} MB; "
            f"at most {MAX_SHARED_STRINGS_BYTES / 1_048_576:.0f} MB is accepted. "
            "Upload the data as CSV instead."
        )


def _check_csv_width(payload: bytes, lines: int | None) -> None:
    """Reject a CSV with any row wider than MAX_COLUMNS fields.

    Every row that will be read is checked, not just the header: when a row
    is wider than its header, pandas promotes the surplus fields to index
    levels, and one 2,000,000-field row cost 187 s and 3.8 GB to accept.
    Counts delimiters in raw bytes, so nothing is allocated per field; a
    quoted comma counts too, which can only make the check stricter.

    Args:
        payload: The raw CSV.
        lines: How many lines pandas will read (header included); ``None``
            scans them all.
    """
    for line in islice(BytesIO(payload), lines):
        columns = line.count(b",") + 1
        if columns > MAX_COLUMNS:
            raise UploadError(
                f"A row has {columns:,} columns; at most {MAX_COLUMNS:,} are accepted."
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

    if suffix in CSV_SUFFIXES:
        # The header plus the rows pandas will read.
        _check_csv_width(payload, None if nrows is None else nrows + 1)
    elif max_inflated_bytes is not None:
        _check_archive(payload, max_inflated_bytes)

    try:
        if suffix in CSV_SUFFIXES:
            frame = pd.read_csv(BytesIO(payload), nrows=nrows)
        else:
            # Pinned: pandas otherwise picks a reader from the archive's
            # contents, so a renamed .ods or .xlsb reached readers that are
            # not installed and failed with ImportError.
            frame = pd.read_excel(BytesIO(payload), nrows=nrows, engine="openpyxl")
    except _WORKBOOK_ERRORS as exc:
        # None of these is a ValueError, so each used to escape as a 500.
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

    # A workbook's width is only known once parsed; its cost was already
    # bounded by the archive checks above.
    if frame.shape[1] > MAX_COLUMNS:
        raise UploadError(
            f"The file has {frame.shape[1]:,} columns; at most {MAX_COLUMNS:,} are accepted."
        )

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
