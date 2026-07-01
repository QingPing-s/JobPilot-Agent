ARG NODE_VERSION=20.19.5
ARG PYTHON_VERSION=3.10.18

FROM node:${NODE_VERSION}-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JOBPILOT_DATA_DIR=/home/jobpilot-data \
    JOBPILOT_FRONTEND_DIST=/app/frontend/dist \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock && pip check

COPY src/ ./src/
COPY data/user_profile.json ./data/user_profile.json
COPY data/sample_jds/ ./data/sample_jds/
COPY data/job_seed.json ./data/job_seed.json
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

RUN useradd --create-home --uid 10001 jobpilot && \
    mkdir -p /home/jobpilot-data/jobs_csv \
    /home/jobpilot-data/sample_jds \
    /home/jobpilot-data/vector_store \
    /home/jobpilot-data/outputs \
    /home/jobpilot-data/traces && \
    chown -R jobpilot:jobpilot /home/jobpilot-data /app

USER jobpilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
