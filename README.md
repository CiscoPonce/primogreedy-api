# PrimoGreedy Data API

A lightweight FastAPI wrapper around [DuckDB](https://duckdb.org/) for managing financial data, including seen tickers, paper portfolio tracking, and agent run logs.

## Features

- **DuckDB Persistence:** Efficiently stores and queries data using a local DuckDB file.
- **FastAPI Interface:** Clean RESTful API for interaction with the data layer.
- **Portfolio Management:** Tracks entry prices, verdicts, and sources for stock positions.
- **Live Dashboard:** Built-in HTML dashboard for visualizing portfolio performance, win rates, and recent activity.
- **TTL-Aware Seen Tickers:** Intelligent caching of seen tickers with different TTLs based on verdicts (e.g., 30 days for BUY, 14 days for others).
- **Cached, non-blocking price lookups:** `/portfolio/evaluate` serves live prices from a TTL cache and fetches concurrently in a bounded thread pool.
- **Rate limiting:** Per-IP limits via [slowapi](https://github.com/laurentS/slowapi), tighter on writes.
- **Health & metrics:** `/health` reports pipeline (`cron`) freshness; `/metrics` exposes row counts and cache stats.
- **Kelly Criterion Support:** Tracks position sizing suggestions.
- **Dockerized:** Ready to deploy with Docker and Docker Compose.

See [`docs/OPERATIONS.md`](docs/OPERATIONS.md) for deployment topology, the analysis-cron schedule, config vars and rate limits.

## Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **Database:** DuckDB
- **Data Fetching:** yfinance (for live price evaluation)
- **Frontend:** Vanilla JS, Chart.js (embedded in the API)

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CiscoPonce/primogreedy-api.git
   cd primogreedy-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```env
   DUCKDB_PATH=/home/ubuntu/primogreedy/data.duckdb
   VPS_API_KEY=your_secure_api_key
   ```

### Running the API

#### Locally
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### With Docker
```bash
docker-compose up --build
```

## API Endpoints

- `GET /health`: Liveness + dependency health (DB, cron freshness, cache stats). Returns `degraded` if the analysis pipeline has gone stale.
- `GET /metrics`: Observability snapshot — row counts, `analyses_by_source`, cache stats, uptime.
- `GET /dashboard`: View the live HTML dashboard.
- `GET /portfolio/summary`: Aggregated portfolio statistics.
- `GET /portfolio/evaluate`: Real-time P&L calculation using `yfinance` (TTL-cached, fetched concurrently).
- `GET /seen-tickers`: List recently analyzed tickers.
- `POST /analyses`: Store a structured analysis and stage it on the watchlist.
- `POST /portfolio`: Record a new trade entry.

## Testing

```bash
python -m pytest tests/ -v
```

Unit tests cover the analysis helpers (date coercion, stage/verdict derivation,
TTL cache logic); integration tests exercise the endpoints against a
throwaway DuckDB. Rate limiting is asserted explicitly.

## Security

The API is secured with an `X-API-Key` header. Ensure all write/action requests
include this header. Public: `/health`, `/metrics`, `/dashboard`. Requests are
rate-limited per client IP.

## License

MIT
