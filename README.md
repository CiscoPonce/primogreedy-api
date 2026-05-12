# PrimoGreedy Data API

A lightweight FastAPI wrapper around [DuckDB](https://duckdb.org/) for managing financial data, including seen tickers, paper portfolio tracking, and agent run logs.

## Features

- **DuckDB Persistence:** Efficiently stores and queries data using a local DuckDB file.
- **FastAPI Interface:** Clean RESTful API for interaction with the data layer.
- **Portfolio Management:** Tracks entry prices, verdicts, and sources for stock positions.
- **Live Dashboard:** Built-in HTML dashboard for visualizing portfolio performance, win rates, and recent activity.
- **TTL-Aware Seen Tickers:** Intelligent caching of seen tickers with different TTLs based on verdicts (e.g., 30 days for BUY, 14 days for others).
- **Kelly Criterion Support:** Tracks position sizing suggestions.
- **Dockerized:** Ready to deploy with Docker and Docker Compose.

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

- `GET /health`: System health check.
- `GET /dashboard`: View the live HTML dashboard.
- `GET /portfolio/summary`: Aggregated portfolio statistics.
- `GET /portfolio/evaluate`: Real-time P&L calculation using `yfinance`.
- `GET /seen-tickers`: List recently analyzed tickers.
- `POST /portfolio`: Record a new trade entry.

## Security

The API is secured with an `X-API-Key` header. Ensure all requests (except the dashboard and health check) include this header.

## License

MIT
