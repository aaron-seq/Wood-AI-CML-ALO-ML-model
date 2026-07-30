# ADR-0004: Centralise upload validation; client errors are 400s

**Status:** Accepted · **Date:** 2026-07-30

## Context

All three upload endpoints shared this shape:

```python
try:
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file.file)
    else:
        raise HTTPException(400, "Unsupported file format")
    ...
except Exception as e:
    raise HTTPException(500, f"Error processing file: {e}")
```

`HTTPException` is an `Exception`, so the handler caught the endpoint's own
`400` and re-raised it as a `500`. Verified against the shipped code: an
unsupported extension, an empty file and a malformed CSV all returned
`500 Internal Server Error`. A client had no way to tell "your file is wrong"
from "the service is broken," and every user mistake looked like an outage in
the logs.

Three further problems in the same block:

- **No size limit.** Files were parsed entirely in memory, so a single large
  upload could exhaust the worker.
- **`file.filename` could be `None`**, making `.endswith` raise `AttributeError`
  → another 500.
- **The logic was triplicated**, so a fix had to be applied three times and
  had already been applied inconsistently.

## Decision

`app/ingestion.py` owns reading an upload. It raises `UploadError` — a plain
`ValueError` subclass, not an HTTP type — and a single registered
`@app.exception_handler(UploadError)` maps it to `400` with the reason and the
request id.

The rule this encodes: **`HTTPException` is never raised inside a `try` whose
handler catches broad exceptions.** Domain errors are domain types; only the
HTTP boundary knows about status codes. That inverts the dependency and makes
the bug structurally unavailable rather than merely fixed.

Also folded in, once instead of three times: a byte-size limit, a row limit, a
missing-filename check, and treating the filename as untrusted — reduced to its
final path component so a traversal-style name cannot influence anything but
parser selection.

## Alternatives considered

**Catch `HTTPException` and re-raise before the generic handler.** The
one-line fix. Rejected: it leaves the trap in place for the next endpoint and
does nothing about the duplication or the missing limits.

**Validate with a FastAPI dependency (`Depends`).** Attractive — validation
would appear in the signature and in the OpenAPI schema. Rejected because the
size and row limits come from `settings`, so the dependency would need its own
factory, and the endpoints need the parsed DataFrame rather than the raw file.
The net line count was higher for no behavioural gain.

**Sniff content by magic bytes rather than trusting the extension.** More
robust against a mislabelled file, and worth doing if users hit it. Deferred:
the current failure is a clear `400` explaining the file could not be parsed,
which is actionable. Extension matching is what the previous behaviour
implied, so this keeps the contract.

## Consequences

Callers can distinguish their errors from ours, and each `400` names what to
fix. One place defines the limits; a new endpoint gets all of it by calling
`read_upload`. Nine regression tests in `tests/test_regressions.py` assert the
status codes across all three endpoints so the trap cannot come back unnoticed.

Two behaviour changes worth flagging:

- Requests that used to return `500` now return `400`. Any client asserting on
  `500` for a bad file will need updating — but such a client was coding
  against a bug.
- `/forecast-remaining-life` previously proceeded past a failed validation and
  returned `200` with an empty array when required columns were missing. It now
  returns `400` naming them. Strictly more informative, but it is a change in
  contract.

The size limit is a real ceiling, not a soft one: legitimate large uploads now
fail rather than succeeding slowly. `MAX_UPLOAD_BYTES` is configurable, and
`docs/DEPLOYMENT.md` ties it to container memory sizing.
