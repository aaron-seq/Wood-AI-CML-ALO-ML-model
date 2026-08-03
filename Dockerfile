# Multi-stage build. Wheels are compiled in the builder so the runtime
# image never has to ship a compiler toolchain.
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Both requirement sets are copied because requirements-streamlit.txt
# includes requirements.txt by reference.
COPY requirements.txt requirements-streamlit.txt ./

# TARGET selects the dependency set: "api" (default) or "dashboard".
ARG TARGET=api
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ "$TARGET" = "dashboard" ]; then \
           /opt/venv/bin/pip install -r requirements-streamlit.txt; \
       else \
           /opt/venv/bin/pip install -r requirements.txt; \
       fi


FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser . /app

# Run unprivileged: a container that only reads uploads and serves
# predictions has no reason to hold root.
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
