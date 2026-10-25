"""Tests for upload parsing and validation."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.ingestion import UploadError, parse_bytes, parse_within_limits


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


class TestParseBytes:
    def test_parses_csv(self, valid_cml_frame):
        parsed = parse_bytes(_csv_bytes(valid_cml_frame), "cml.csv")
        assert len(parsed) == 3

    def test_parses_xlsx(self, valid_cml_frame, tmp_path):
        path = tmp_path / "cml.xlsx"
        valid_cml_frame.to_excel(path, index=False)
        parsed = parse_bytes(path.read_bytes(), "cml.xlsx")
        assert len(parsed) == 3

    def test_extension_matching_is_case_insensitive(self, valid_cml_frame):
        assert len(parse_bytes(_csv_bytes(valid_cml_frame), "CML.CSV")) == 3

    @pytest.mark.parametrize("filename", ["notes.txt", "archive.zip", "noextension"])
    def test_rejects_unsupported_formats(self, filename):
        with pytest.raises(UploadError, match="Unsupported file format"):
            parse_bytes(b"a,b\n1,2", filename)

    def test_rejects_a_missing_filename(self):
        with pytest.raises(UploadError, match="No filename"):
            parse_bytes(b"a,b\n1,2", None)

    def test_rejects_empty_payload(self):
        with pytest.raises(UploadError, match="empty"):
            parse_bytes(b"", "cml.csv")

    def test_rejects_a_headerless_file(self):
        with pytest.raises(UploadError):
            parse_bytes(b"\n\n", "cml.csv")

    def test_rejects_a_header_only_file(self):
        with pytest.raises(UploadError, match="no data rows"):
            parse_bytes(b"id_number,thickness_mm\n", "cml.csv")

    def test_a_traversal_style_filename_is_reduced_to_its_extension(self, valid_cml_frame):
        """The client-supplied name selects a parser and nothing else."""
        parsed = parse_bytes(_csv_bytes(valid_cml_frame), "../../etc/passwd.csv")
        assert len(parsed) == 3

    def test_a_windows_path_is_handled(self, valid_cml_frame):
        parsed = parse_bytes(_csv_bytes(valid_cml_frame), r"C:\Users\a\cml.csv")
        assert len(parsed) == 3

    def test_binary_content_with_a_csv_name_is_reported_as_a_bad_upload(self):
        with pytest.raises(UploadError):
            parse_bytes(b"\x00\x81\xfe\xff" * 64, "cml.csv")


class TestRowLimit:
    def test_row_limit_is_enforced(self, client, csv_upload, monkeypatch):
        from app import main

        monkeypatch.setattr(main.settings, "MAX_UPLOAD_ROWS", 2)
        frame = pd.DataFrame(
            {
                "id_number": ["A", "B", "C"],
                "average_corrosion_rate": [0.1, 0.1, 0.1],
                "thickness_mm": [9.0, 9.0, 9.0],
                "commodity": ["Crude Oil"] * 3,
                "feature_type": ["Pipe"] * 3,
                "cml_shape": ["Both"] * 3,
            }
        )
        response = client.post("/upload-cml-data", files=csv_upload(frame))
        assert response.status_code == 400
        assert "row limit" in response.json()["detail"]


class TestParseWithinLimits:
    """The single upload contract, shared by the API and the dashboard.

    The dashboard used to call pandas directly and enforce no limits at
    all, so it accepted files the API rejected and worded the same failure
    differently.
    """

    def test_a_valid_file_parses(self, valid_cml_frame):
        parsed = parse_within_limits(_csv_bytes(valid_cml_frame), "cml.csv", 10_000_000, 1000)
        assert len(parsed) == 3

    def test_an_oversized_payload_is_rejected_before_parsing(self):
        with pytest.raises(UploadError, match="too large"):
            parse_within_limits(b"x" * 2048, "cml.csv", max_bytes=1024, max_rows=1000)

    def test_too_many_rows_is_rejected(self, valid_cml_frame):
        with pytest.raises(UploadError, match="row limit"):
            parse_within_limits(
                _csv_bytes(valid_cml_frame), "cml.csv", max_bytes=10_000_000, max_rows=2
            )

    def test_format_errors_still_surface(self, valid_cml_frame):
        with pytest.raises(UploadError, match="Unsupported file format"):
            parse_within_limits(_csv_bytes(valid_cml_frame), "notes.txt", 10_000_000, 1000)

    def test_the_size_check_runs_first(self):
        """An oversized file must not be parsed just to learn it is oversized."""
        with pytest.raises(UploadError, match="too large"):
            parse_within_limits(b"\x00" * 2048, "cml.csv", max_bytes=1024, max_rows=1000)


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


class TestParsingWorkIsBounded:
    """The limits must bound the work done, not just the answer.

    The row limit used to be checked after the whole file had been parsed,
    so a small, highly compressible workbook declaring far more rows than
    the limit cost seconds of CPU and memory before it was rejected, with
    the event loop blocked for all of it.
    """

    @pytest.mark.parametrize("to_bytes, name", [(_csv_bytes, "big.csv"), (_xlsx_bytes, "big.xlsx")])
    def test_reading_stops_once_the_row_limit_is_passed(self, to_bytes, name):
        payload = to_bytes(pd.DataFrame({"a": range(5_000)}))
        # "more than", not an exact count: an exact count would prove the
        # whole file was read just to reject it.
        with pytest.raises(UploadError, match=r"more than 10 rows.*row limit"):
            parse_within_limits(payload, name, max_bytes=10_000_000, max_rows=10)

    def test_a_file_at_exactly_the_row_limit_is_accepted(self, valid_cml_frame):
        parsed = parse_within_limits(_csv_bytes(valid_cml_frame), "cml.csv", 10_000_000, 3)
        assert len(parsed) == 3

    def test_a_workbook_that_inflates_past_the_limit_is_rejected_unparsed(self, monkeypatch):
        payload = _xlsx_bytes(pd.DataFrame({"a": ["x" * 200] * 2_000}))
        parsed = []
        monkeypatch.setattr(pd, "read_excel", lambda *a, **k: parsed.append(1))
        # 64 KB on the wire; the allowance is four times that, decompressed.
        with pytest.raises(UploadError, match="decompress"):
            parse_within_limits(payload, "bomb.xlsx", max_bytes=64 * 1024, max_rows=100_000)
        assert not parsed, "the workbook was parsed before its inflated size was checked"

    def test_a_corrupt_workbook_is_a_client_error(self):
        with pytest.raises(UploadError):
            parse_within_limits(b"PK\x03\x04 not really a zip", "cml.xlsx", 10_000_000, 1000)


@pytest.fixture
def anyio_backend():
    # AnyIO's pytest plugin ships with AnyIO, which Starlette already
    # depends on, so async tests need no extra test dependency.
    return "asyncio"


class TestParsingDoesNotBlockTheServer:
    @pytest.mark.anyio
    async def test_health_answers_while_an_upload_is_parsed(self, monkeypatch, valid_cml_frame):
        """A 100,000-row workbook takes ~10 s to parse. Parsed on the event
        loop, it stalled every other request -- including the health probe
        an orchestrator uses to decide whether to restart the container."""
        import asyncio
        import time

        import httpx2

        import app.ingestion as ingestion
        from app import main

        real = ingestion.parse_within_limits

        def slow_parse(*args, **kwargs):
            time.sleep(0.5)
            return real(*args, **kwargs)

        monkeypatch.setattr(ingestion, "parse_within_limits", slow_parse)
        finished: list[str] = []
        transport = httpx2.ASGITransport(app=main.app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:

            async def upload():
                files = {"file": ("cml.csv", _csv_bytes(valid_cml_frame), "text/csv")}
                await client.post("/upload-cml-data", files=files)
                finished.append("upload")

            async def health():
                await asyncio.sleep(0.1)
                await client.get("/health")
                finished.append("health")

            await asyncio.gather(upload(), health())

        assert finished == ["health", "upload"]


def _zip(members: dict[str, bytes | str]) -> bytes:
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class TestWidthAndMemberBounds:
    """Found by adversarial review of the row bound: rows were capped, but
    columns and individual archive members were not."""

    def test_a_csv_wider_than_the_column_limit_is_rejected_before_parsing(self, monkeypatch):
        """An 889 KB CSV with 100,000 columns was accepted at +120 MB RSS."""
        from app import ingestion

        header = ",".join(f"c{i}" for i in range(ingestion.MAX_COLUMNS + 1))
        parsed = []
        monkeypatch.setattr(pd, "read_csv", lambda *a, **k: parsed.append(1))
        with pytest.raises(UploadError, match="columns"):
            parse_within_limits(f"{header}\n".encode(), "w.csv", 10_000_000, 1000)
        assert not parsed

    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param("a,b,c\n1,2,3\n{row}\n", id="after-a-row"),
            # Found by review: blank lines (which pandas skips) or a quoted
            # newline pushed a wide row past a physical-line scan, and it was
            # accepted with ~50,000 index levels.
            pytest.param("a,b,c\n" + "\n" * 50 + "{row}\n", id="after-blank-lines"),
            pytest.param('a,b,c\n"1\n2",2,3\n{row}\n', id="after-a-quoted-newline"),
        ],
    )
    def test_a_data_row_wider_than_its_header_is_rejected(self, layout):
        """pandas turns surplus fields into index levels: a 2,000,000-field
        row cost 187 s and 3.8 GB to accept."""
        row = ",".join("1" for _ in range(50_000))
        with pytest.raises(UploadError, match="could not be parsed|columns"):
            parse_within_limits(layout.format(row=row).encode(), "w.csv", 10_000_000, 3)

    # pandas warns that index_col=False drops the empty trailing field, which
    # is exactly the intent here.
    @pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning")
    def test_a_trailing_delimiter_on_every_row_still_parses(self):
        parsed = parse_within_limits(b"a,b,c\n1,2,3,\n4,5,6,\n", "t.csv", 10_000_000, 10)
        assert parsed.to_dict("records") == [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
    def test_every_line_ending_parses(self, newline):
        """A CR-only file was rejected as one 306-column line."""
        payload = newline.join(["a,b,c,d,e,f", *["1,2,3,4,5,6"] * 60]).encode()
        assert parse_within_limits(payload, "e.csv", 10_000_000, 100).shape == (60, 6)

    def test_a_header_hidden_behind_a_quoted_newline_is_still_bounded(self):
        from app import ingestion

        header = '"a\nb",' + ",".join(f"c{i}" for i in range(ingestion.MAX_COLUMNS + 1))
        with pytest.raises(UploadError, match="columns"):
            parse_within_limits(f"{header}\n1\n".encode(), "h.csv", 10_000_000, 10)

    def test_an_endless_header_is_rejected_unparsed(self, monkeypatch):
        from app import ingestion

        parsed = []
        monkeypatch.setattr(pd, "read_csv", lambda *a, **k: parsed.append(1))
        payload = b"x" * (ingestion.MAX_RECORD_BYTES + 1)
        with pytest.raises(UploadError, match="header"):
            parse_within_limits(payload, "h.csv", 10_000_000, 10)
        assert not parsed

    def test_an_endless_first_data_row_is_rejected_quickly(self):
        """A first row of 25 MB of quote characters cost 4.6 s to scan."""
        import time

        from app import ingestion

        payload = b"a\n" + b'"' * (ingestion.MAX_RECORD_BYTES * 8)
        started = time.perf_counter()
        with pytest.raises(UploadError, match="first data row"):
            parse_within_limits(payload, "q.csv", 10_000_000, 10)
        assert time.perf_counter() - started < 1.0

    def test_identifiers_keep_their_leading_zeros(self):
        """id_number was type-inferred, so 00042 became 42 and missed any
        override recorded against 00042."""
        parsed = parse_within_limits(b"id_number,x\n00042,1\n", "i.csv", 10_000_000, 10)
        assert parsed.loc[0, "id_number"] == "00042"

    def test_a_csv_at_the_column_limit_is_accepted(self):
        from app import ingestion

        names = [f"c{i}" for i in range(ingestion.MAX_COLUMNS)]
        payload = (",".join(names) + "\n" + ",".join("1" for _ in names) + "\n").encode()
        assert parse_within_limits(payload, "w.csv", 10_000_000, 1000).shape == (1, len(names))

    def test_a_workbook_wider_than_the_column_limit_is_rejected(self):
        from app import ingestion

        frame = pd.DataFrame([range(ingestion.MAX_COLUMNS + 1)])
        with pytest.raises(UploadError, match="columns"):
            parse_within_limits(_xlsx_bytes(frame), "w.xlsx", 10_000_000, 1000)

    def test_an_oversized_shared_strings_part_is_rejected_unparsed(self, monkeypatch):
        """Shared strings are parsed in full when a workbook opens, about 20x
        more expensively per byte than sheet XML. A 129 KB workbook whose
        table fit inside the whole-archive allowance cost 20 s and 233 MB."""
        from app import ingestion

        table = (
            b"<sst>" + b"<si><t>y</t></si>" * (ingestion.MAX_SHARED_STRINGS_BYTES // 16) + b"</sst>"
        )
        payload = _zip({"xl/workbook.xml": "<workbook/>", "xl/sharedStrings.xml": table})
        parsed = []
        monkeypatch.setattr(pd, "read_excel", lambda *a, **k: parsed.append(1))
        with pytest.raises(UploadError, match="shared strings"):
            parse_within_limits(payload, "s.xlsx", 25 << 20, 1000)
        assert not parsed

    def test_an_archive_with_too_many_members_is_rejected(self):
        from app import ingestion

        members = {f"xl/m{i}.xml": "" for i in range(ingestion.MAX_ARCHIVE_MEMBERS + 1)}
        with pytest.raises(UploadError, match="parts"):
            parse_within_limits(_zip(members), "m.xlsx", 25 << 20, 1000)


class TestMalformedWorkbooksAreClientErrors:
    """Each of these zip files named .xlsx used to escape as a 500."""

    @pytest.mark.parametrize(
        "members",
        [
            pytest.param({"hello.txt": "hi"}, id="no-workbook"),
            pytest.param(
                {"[Content_Types].xml": "<not xml", "xl/workbook.xml": "<<<"}, id="bad-xml"
            ),
            pytest.param(
                {
                    "mimetype": "application/vnd.oasis.opendocument.spreadsheet",
                    "content.xml": "<x/>",
                },
                id="ods-renamed",
            ),
            pytest.param({"[Content_Types].xml": "<Types/>", "xl/workbook.bin": "\x00"}, id="xlsb"),
        ],
    )
    def test_is_a_400(self, members):
        with pytest.raises(UploadError):
            parse_within_limits(_zip(members), "m.xlsx", 25 << 20, 1000)

    def test_legacy_xls_is_refused_up_front(self):
        """.xls was on the allow-list, but no .xls reader is installed, so
        every .xls upload failed after the fact."""
        with pytest.raises(UploadError, match="Unsupported file format"):
            parse_bytes(b"\xd0\xcf\x11\xe0", "old.xls")
