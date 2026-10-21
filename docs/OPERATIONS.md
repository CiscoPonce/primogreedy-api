# PrimoGreedy Operations

Operational notes for the PrimoGreedy data API and its analysis pipeline.
Everything here is derived from the live system; update it when the setup
changes.

## Deployment topology

```
Internet / LAN
     │  https://<host>:8080   (self-signed cert, /pg.crt to trust)
     ▼
   Caddy  ── terminates TLS on :8080
     │
     │  reverse_proxy 127.0.0.1:8081
     ▼
 Docker: primogreedy-api   (uvicorn api:app --host 0.0.0.0 --port 8080)
     │  volume: .:/app
     ▼
   /app/data.duckdb        (single-writer DuckDB)
```

- The container is defined in `docker-compose.yml`. It binds **only** to
  loopback (`127.0.0.1:8081`); Caddy is the sole public entry point.
- There is a **stale, disabled** systemd unit at
  `/etc/systemd/system/primogreedy-api.service`. It is *not* how the app runs
  — do not enable it. The container is the source of truth.
- Restart policy is `restart: always`, so a crash self-heals.

## Authentication

- Write/action endpoints require an `X-API-Key` header equal to `VPS_API_KEY`.
- Read-only dashboard and `/health`, `/metrics` are public.
- `VPS_API_KEY` is injected from the host environment / `.env` by
  `docker-compose.yml` — it is no longer hardcoded there.

## Analysis pipeline (the "crons")

Two jobs drive analysis. They run **outside** the container (in the operator's
own agent automation) and authenticate with `VPS_API_KEY`, posting to
`POST /analyses` with a `source` label:

| Source        | Observed window (UTC) | Purpose                                  |
|---------------|------------------------|------------------------------------------|
| `Morning Cron`| ~09:00–13:00            | Discover + first-pass analysis of new names |
| `reeval_cron` | ~17:00–22:00            | Re-evaluate existing watchlist / portfolio |

> These are **not** in Hermes' `cron/jobs.json`, nor in root/system crontabs.
> See `crontab.example` for the shape they should have if/when moved to a
> host cron. If the external trigger dies, analysis silently stops — which is
> exactly why `/health` now reports `crons_healthy`.

### Health signal

`GET /health` returns:

```json
{
  "status": "ok",              // or "degraded"
  "crons_healthy": true,
  "last_analysis_at": "2026-10-21 10:52:42",
  "last_analysis_age_seconds": 240,
  "price_cache_entries": 0,
  "price_ttl_seconds": 600
}
```

`status` becomes `degraded` when no analysis has been written within
`CRON_STALE_AFTER` (default 36h — long enough to survive one missed daily
run). Point your monitor at `/health` and alert on `status != "ok"`.

### Metrics

`GET /metrics` is a JSON observability snapshot (no auth): row counts per
table, `analyses_by_source`, cache stats and process uptime. Scrape it by hand
or from the monitoring agent; it is intentionally not Prometheus-format.

## Configuration (env vars)

| Var                  | Default | Meaning                                             |
|----------------------|---------|-----------------------------------------------------|
| `DUCKDB_PATH`        | `/home/ubuntu/primogreedy/data.duckdb` | Database file        |
| `VPS_API_KEY`        | —       | Required for write endpoints                        |
| `PRICE_TTL`          | `600`   | yfinance price cache TTL (seconds)                  |
| `PRICE_FETCH_WORKERS`| `8`     | Max concurrent yfinance fetches per evaluate call   |
| `CRON_STALE_AFTER`   | `129600`| Seconds after which a silent pipeline = degraded    |

## Rate limits (per client IP)

| Endpoint                          | Limit         |
|-----------------------------------|---------------|
| Global default                    | 300/min       |
| `GET /portfolio/evaluate`         | 30/min        |
| `POST /analyses`                  | 120/min       |
| `POST /seen-tickers`              | 120/min       |
| `POST /portfolio`                 | 60/min        |
| `PATCH /portfolio/{t}/fill`       | 60/min        |
| `DELETE /portfolio/{t}`           | 60/min        |
| `POST /watchlist/{t}/accept`      | 60/min        |
| `POST /watchlist/{t}/dismiss`     | 60/min        |
| `POST /portfolio/deduplicate`     | 10/min        |

Exceeding a limit returns `429 Too Many Requests`.

## Price fetching

`GET /portfolio/evaluate` looks up live prices via yfinance. To keep it from
blocking the event loop or stampeding yfinance:

1. Results are cached in-process for `PRICE_TTL` (default 10 min).
2. Fetches run concurrently in a bounded `ThreadPoolExecutor`
   (`PRICE_FETCH_WORKERS`) via `run_in_threadpool`.
3. A failed symbol yields `price: 0` (reported as `null` P&L), never a 500.

The cache is process-local; a container restart empties it (harmless).

## Tests

```bash
python -m pytest tests/ -v
```

Tests use a throwaway DuckDB pointed to by `DUCKDB_PATH` (see
`tests/conftest.py`) and never touch `data.duckdb`. Rate limiting is disabled
for most tests and asserted explicitly in the rate-limit test.

## Backup / restore

The database is a single file: `data.duckdb` (plus any `-wal`). Back up with
the container stopped, or copy while idle:

```bash
docker stop primogreedy-api
cp data.duckdb data.duckdb.bak-$(date +%Y%m%d)
docker start primogreedy-api
```

## Known log noise (not errors)

- `ERROR:yfinance:HTTP Error 404` for delisted/invalid symbols — external data
  lookups, handled gracefully.
- Internet scanners hitting `/.git/config`, `/mcp`, `/v1/models`, `/sse` — all
  return 404 by design.
