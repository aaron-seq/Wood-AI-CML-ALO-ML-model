"""Tests for upload parsing and validation."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.ingestion import UploadError, parse_bytes


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
