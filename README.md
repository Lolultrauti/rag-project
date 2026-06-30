# Helix RAG

A small, production-shaped Retrieval-Augmented Generation service. Upload PDF/TXT/MD
documents through a web UI, then ask questions answered **only** from those documents,
with the source passages shown. Built on FastAPI + PostgreSQL/pgvector + Google Gemini.

## How retrieval works (hybrid)

Every query runs two retrievers and fuses them:

- **Semantic** — Gemini embeddings + pgvector cosine similarity (good at meaning).
- **Keyword** — PostgreSQL full-text search (good at specific names/terms/entities).
- **Fusion** — Reciprocal Rank Fusion (RRF) combines both rankings.

This fixes the classic dense-only failure where a specific question ("who submitted
this project") misses the exact chunk. If the embedding call fails (e.g. quota), the
keyword retriever still answers — the system degrades instead of going down. It abstains
("I don't know based on the provided context") when neither retriever finds anything.

## ⚠️ Before you deploy: API quota

The app calls Google Gemini for embeddings and answer generation. The **free tier is not
deployable for real users** — its limits are very low:

| Call | Free-tier limit |
|------|-----------------|
| Embeddings (`embed_content`) | ~1000 / day, ~100 / min |
| Generation (`gemini-2.5-flash`) | **~20 / day** |

For anything beyond a personal demo, **enable billing on the Gemini API key** (pay-as-you-go).
No code change — same key, the limits just lift. This is the single operational prerequisite
for a usable deployment.

## Deploy with Docker (recommended, portable)

Runs the API + a pgvector Postgres together. Works on any host that runs Docker
(a VPS, or a PaaS that accepts a Dockerfile/compose).

```bash
# 1. Configure
cp .env.example .env
#    edit .env: set GEMINI_API_KEY. Leave DATABASE_URL as-is — compose overrides it
#    to reach Postgres on the internal network.

# 2. Build + run (schema is auto-applied on the first run)
docker compose up -d --build

# 3. Open the UI
#    http://localhost:8000
```

- App: http://localhost:8000  (UI at `/`, API docs at `/docs`)
- Postgres: host port `5434` (internal `5432`)
- Stop: `docker compose down`  ·  reset data: `docker compose down -v`

To deploy to a server, point your host at this repo (it has the `Dockerfile` and
`docker-compose.yml`). Set the same env vars in the host's secret store. Put a TLS
reverse proxy (Caddy/Nginx, or the PaaS's built-in) in front of port 8000.

## Local development (without Docker for the app)

```bash
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

docker compose up -d db          # just Postgres
cp .env.example .env             # set GEMINI_API_KEY; DATABASE_URL -> localhost:5434

uvicorn app.main:app --reload    # http://127.0.0.1:8000
```

## Configuration (`.env`)

| Var | Required | Purpose |
|-----|----------|---------|
| `GEMINI_API_KEY` | yes | Gemini API key (enable billing for real use) |
| `DATABASE_URL` | yes | Postgres connection (compose sets this for the app automatically) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | for compose | DB credentials |
| `RATE_LIMIT` | no | Per-IP limit, e.g. `10/minute` |
| `DAILY_COST_CAP` | no | Hard ceiling on `/query` calls per UTC day |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| POST | `/query` | Ask a question → answer + sources |
| POST | `/ingest` | Upload a document (PDF/TXT/MD) |
| GET | `/stats` | Document + passage counts |
| GET | `/health` | Liveness (no dependencies) |
| GET | `/ready` | Readiness (checks DB + key) |

## Tests

```bash
pytest                                   # unit + schema tests (no API calls)
RUN_INTEGRATION=1 pytest tests/integration/   # end-to-end (uses live Gemini quota)
```

## Project layout

```
app/
  api/routes.py          # /query, /ready
  main.py                # app, /ingest, /stats, /health, serves the UI
  ingestion/             # load → chunk → embed → store (batched, quota-aware)
  retrieval/             # dense_search, lexical_search, fusion (RRF), hybrid (public)
  generation/            # grounded prompt + Gemini answer
  static/index.html      # single-page UI
schema.sql               # full DB schema (tables, pgvector + full-text indexes)
migrations/              # incremental DB changes
docs/superpowers/        # design spec + implementation plan
```
