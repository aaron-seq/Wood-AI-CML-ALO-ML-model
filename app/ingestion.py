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
import re
import zipfile
from io import BytesIO
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
# with the column count. A CSV header is checked before pandas allocates;
# data rows wider than the header are refused by the parser itself.
MAX_COLUMNS = 256
# The header and first data row are measured before parsing (see
# _check_csv_width), so their length is bounded too, which caps that scan at
# ~10 ms: unbounded, a first row of 25 MB of quote characters took 4.6 s.
# A CML row is well under 1 KB; 256 column names fit many times over.
MAX_RECORD_BYTES = 64 * 1024

# A workbook's shared-strings table is parsed in full when it opens, before
# any row limit can apply, and costs about 20x more per byte than sheet XML:
# a 129 KB file whose table fit the whole-archive allowance took 20 s and
# 233 MB. 16 MiB holds 100,000 unique 40-character ids several times over.
MAX_SHARED_STRINGS_BYTES = 16 * 1024 * 1024
SHARED_STRINGS_PART = "xl/sharedStrings.xml"

# A spreadsheet has a few dozen parts; thousands are only useful to an attacker.
MAX_ARCHIVE_MEMBERS = 1000

# Identifiers are labels, not numbers: inferred as integers, "00042" became 42
# and no longer matched an override recorded against "00042".
_ID_AS_TEXT = {"id_number": str}

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


_CSV_SIGNIFICANT = re.compile(rb'[",\r\n]')


def _record_width(payload: bytes, start: int, what: str) -> tuple[int, int]:
    """Count the fields of the CSV record starting at ``start``.

    Quote-aware, ends at the first unquoted CR or LF (any line ending), and
    returns as soon as the record passes MAX_COLUMNS fields. Only quotes,
    commas and line breaks are visited, so long text fields cost nothing.

    Returns:
        ``(fields, end)`` where ``end`` is the offset of the terminator, or
        ``len(payload)`` if the record runs to the end.

    Raises:
        UploadError: If the record is too wide or longer than MAX_RECORD_BYTES.
    """
    fields, quoted = 1, False
    for match in _CSV_SIGNIFICANT.finditer(payload, start):
        if match.start() - start > MAX_RECORD_BYTES:
            break
        char = match.group()
        if char == b'"':
            quoted = not quoted
        elif quoted:
            continue
        elif char == b",":
            fields += 1
            if fields > MAX_COLUMNS:
                raise UploadError(
                    f"A row has more than {MAX_COLUMNS:,} columns; "
                    f"at most {MAX_COLUMNS:,} are accepted."
                )
        else:
            return fields, match.start()
    if len(payload) - start > MAX_RECORD_BYTES:
        raise UploadError(
            f"The {what} is longer than {MAX_RECORD_BYTES // 1024} KB. "
            "Check that the file is a CSV with one record per line."
        )
    return fields, len(payload)


def _check_csv_width(payload: bytes) -> None:
    """Reject a CSV whose header or first data row is too wide.

    Rows after the first are refused by the parser itself when wider than
    the header (``index_col=False``). The first data row is the exception:
    pandas truncates it with a warning instead, so it is measured here, as
    is the header, which pandas must fully materialise before anything else
    applies. A scan of physical lines was tried first and was wrong: pandas
    skips blank lines and quoted fields span lines, so a wide row could sit
    past the scan and be accepted with ~50,000 index levels.
    """
    _, end = _record_width(payload, 0, "header row")
    # pandas skips blank lines, so the first data row is the next non-blank one.
    while end < len(payload) and payload[end] in b"\r\n":
        end += 1
    if end < len(payload):
        _record_width(payload, end, "first data row")


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
        _check_csv_width(payload)
    elif max_inflated_bytes is not None:
        _check_archive(payload, max_inflated_bytes)

    try:
        if suffix in CSV_SUFFIXES:
            # index_col=False: a row with more fields than the header is a
            # parse error. By default pandas promoted the surplus to index
            # levels, so one 2,000,000-field row cost 187 s and 3.8 GB to
            # accept, and rows with a trailing delimiter were mis-shifted.
            frame = pd.read_csv(BytesIO(payload), nrows=nrows, index_col=False, dtype=_ID_AS_TEXT)
        else:
            # Pinned: pandas otherwise picks a reader from the archive's
            # contents, so a renamed .ods or .xlsb reached readers that are
            # not installed and failed with ImportError.
            frame = pd.read_excel(
                BytesIO(payload), nrows=nrows, engine="openpyxl", dtype=_ID_AS_TEXT
            )
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
