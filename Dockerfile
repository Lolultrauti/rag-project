# Production image for the Helix RAG API (FastAPI + pgvector + Gemini).
# psycopg2-binary ships its own libpq, so no system build tools are needed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + SQL (schema/migrations are handy to have in the image).
COPY app ./app
COPY schema.sql ./schema.sql
COPY migrations ./migrations

EXPOSE 8000

# Single worker by design: the daily-cost cap and rate limiter use in-process
# state, so multiple workers would each keep their own counters. For this app's
# scale one worker is correct; scale out behind a shared store if ever needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
