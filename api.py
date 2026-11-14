"""PrimoGreedy Data API — FastAPI wrapper around DuckDB.

Serves seen tickers, paper portfolio, and agent run logs over HTTP.
Secured with X-API-Key header. Runs behind Tailscale (private network).

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Optional

import duckdb
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DuckDB's in-process file handle registry can raise ResourceInUseException
# ("Unique file handle conflict") when multiple threads concurrently
# open/close the same database file. Serialize all DB connections on a
# re-entrant lock so concurrent reads/writes cannot collide.
_db_lock = threading.RLock()
_db_write_lock = _db_lock

DB_PATH = os.getenv("DUCKDB_PATH", "/home/ubuntu/primogreedy/data.duckdb")
API_KEY = os.getenv("VPS_API_KEY", "")
if not API_KEY:
    logger.warning("VPS_API_KEY environment variable is not set!")

MEMORY_TTL_LONG = 30 * 24 * 60 * 60   # 30 days — BUY / STRONG BUY
MEMORY_TTL_SHORT = 14 * 24 * 60 * 60  # 14 days — AVOID / WATCH / unknown

# Live-price cache TTL for yfinance lookups (seconds). Quotes change slowly
# relative to request volume; 10 minutes keeps /portfolio/evaluate off the
# yfinance API for repeat hits while staying fresh enough for P&L display.
PRICE_TTL = int(os.getenv("PRICE_TTL", str(10 * 60)))
# Longer negative TTL for failed/delisted symbols (price == 0) to avoid
# repeatedly hitting yfinance 404s every 10 minutes.
NEGATIVE_PRICE_TTL = int(os.getenv("NEGATIVE_PRICE_TTL", str(6 * 60 * 60)))
# Max concurrent yfinance fetches: bounded so a large portfolio cannot
# stampede yfinance (which rate-limits aggressively) during one request.
PRICE_FETCH_WORKERS = int(os.getenv("PRICE_FETCH_WORKERS", "8"))
# A cron run is considered "stale" (and therefore unhealthy) if it has not
# produced an analysis within this window. The Morning Cron fires daily.
CRON_STALE_AFTER = int(os.getenv("CRON_STALE_AFTER", str(36 * 60 * 60)))

# Wall-clock start of this process, for the /metrics uptime gauge.
_PROCESS_STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# Per-client-IP limits. Reads are cheap; writes are capped harder because
# DuckDB serializes on a module-level lock and each write is contended.
limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> duckdb.DuckDBPyConnection:
    """Return a fresh connection (DuckDB is single-writer, but reads are fine)."""
    return duckdb.connect(DB_PATH)


@contextmanager
def get_db_ctx():
    """Context manager for DuckDB connections to ensure proper closure."""
    with _db_lock:
        con = get_db()
        try:
            yield con
        finally:
            con.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db_ctx() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS portfolio_seq START 1;

        CREATE TABLE IF NOT EXISTS seen_tickers (
            ticker    VARCHAR NOT NULL PRIMARY KEY,
            region    VARCHAR DEFAULT 'USA',
            seen_at   TIMESTAMP DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS paper_portfolio (
            id            INTEGER PRIMARY KEY DEFAULT nextval('portfolio_seq'),
            ticker        VARCHAR NOT NULL,
            entry_price   DOUBLE NOT NULL,
            date          DATE NOT NULL,
            verdict       VARCHAR NOT NULL,
            source        VARCHAR DEFAULT 'unknown',
            position_size DOUBLE DEFAULT 0,
            order_id      VARCHAR,
            fill_price    DOUBLE,
            broker_status VARCHAR DEFAULT 'none',
            created_at    TIMESTAMP DEFAULT current_timestamp,
            UNIQUE (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id          VARCHAR PRIMARY KEY,
            ticker      VARCHAR NOT NULL,
            timestamp   TIMESTAMP DEFAULT current_timestamp,
            status      VARCHAR NOT NULL,
            model       VARCHAR,
            latency_ms  INTEGER,
            region      VARCHAR DEFAULT 'USA'
        );

        CREATE SEQUENCE IF NOT EXISTS analysis_seq START 1;

        -- Structured investment analyses (proposal.md framework: 3-layer method).
        CREATE TABLE IF NOT EXISTS analyses (
            id                INTEGER PRIMARY KEY DEFAULT nextval('analysis_seq'),
            ticker            VARCHAR NOT NULL,
            analyzed_at       TIMESTAMP DEFAULT current_timestamp,
            stage             VARCHAR NOT NULL,   -- RESEARCH | WATCHLIST | BUY_CANDIDATE | DISCARD
            verdict           VARCHAR,            -- STRONG BUY | BUY | WATCH | AVOID (legacy label)
            source            VARCHAR DEFAULT 'unknown',
            price             DOUBLE,
            currency          VARCHAR DEFAULT 'USD',
            business_summary       TEXT,
            financial_evolution    TEXT,
            cash_flow              TEXT,
            debt_liquidity         TEXT,
            dilution               TEXT,
            valuation_method       TEXT,
            valuation_bear         TEXT,
            valuation_base         TEXT,
            valuation_bull         TEXT,
            valuation_range_low    DOUBLE,
            valuation_range_high   DOUBLE,
            margin_of_safety       TEXT,
            risks                  TEXT,
            invalidation_facts     TEXT,
            macro_factors          TEXT,
            portfolio_fit          TEXT,
            sources                TEXT,   -- JSON: [{type,url,date,period}]
            data_quality           TEXT,   -- JSON: {facts,estimates,inferences,unverified,pending}
            proposed_action        VARCHAR, -- PENDING|INITIATE|MAINTAIN|INCREASE|REDUCE|CLOSE|DISCARD
            review_trigger         TEXT,
            next_review_at         DATE,
            report_text            TEXT,
            position_size          DOUBLE DEFAULT 0,
            created_at             TIMESTAMP DEFAULT current_timestamp
        );

        -- Watchlist funnel: research stage -> accept/dismiss decision gate.
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker      VARCHAR NOT NULL PRIMARY KEY,
            stage       VARCHAR NOT NULL,
            source      VARCHAR DEFAULT 'unknown',
            analysis_id INTEGER,
            reason      TEXT,
            added_at    TIMESTAMP DEFAULT current_timestamp,
            status      VARCHAR DEFAULT 'active'   -- active | accepted | dismissed
        );
    """)
        # Migration for pre-existing portfolio databases
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS closed_seq START 1;
            CREATE TABLE IF NOT EXISTS closed_positions (
                id            INTEGER PRIMARY KEY DEFAULT nextval('closed_seq'),
                ticker        VARCHAR NOT NULL,
                entry_price   DOUBLE NOT NULL,
                exit_price    DOUBLE,
                entry_date    DATE,
                closed_at     TIMESTAMP DEFAULT current_timestamp,
                verdict       VARCHAR,
                source        VARCHAR DEFAULT 'unknown',
                gain_pct      DOUBLE,
                analysis_id   INTEGER
            );
        """)

        # Migration for pre-existing portfolio databases
        _cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='paper_portfolio'"
        ).fetchall()]
        if "analysis_id" not in _cols:
            con.execute("ALTER TABLE paper_portfolio ADD COLUMN analysis_id INTEGER")
        if "status" not in _cols:
            con.execute("ALTER TABLE paper_portfolio ADD COLUMN status VARCHAR DEFAULT 'active'")


# ---------------------------------------------------------------------------
# Live price cache (TTL)
# ---------------------------------------------------------------------------

_price_cache: dict[str, tuple[float, dict]] = {}
_price_cache_lock = threading.Lock()


def _cache_get(ticker: str) -> Optional[dict]:
    """Return a cached price payload if still within PRICE_TTL, else None."""
    with _price_cache_lock:
        hit = _price_cache.get(ticker)
    if hit is None:
        return None
    ts, payload = hit
    ttl = NEGATIVE_PRICE_TTL if not payload.get("price") else PRICE_TTL
    if time.time() - ts > ttl:
        return None
    return payload


def _cache_put(ticker: str, payload: dict) -> None:
    with _price_cache_lock:
        _price_cache[ticker] = (time.time(), payload)
    # Opportunistic pruning so the dict cannot grow without bound across a
    # long-lived process.
    if len(_price_cache) > 2048:
        cutoff = time.time() - PRICE_TTL
        with _price_cache_lock:
            for k in [k for k, (ts, _) in _price_cache.items() if ts < cutoff]:
                _price_cache.pop(k, None)


def _fetch_price(ticker: str) -> dict:
    """Fetch a single ticker's live price from yfinance (blocking).

    Returns {"price": float, "currency": str}.  On any failure the price is 0
    so callers can treat it as "unknown" rather than exploding the whole
    evaluation for one bad symbol.
    """
    cached = _cache_get(ticker)
    if cached is not None:
        return cached
    payload = {"price": 0.0, "currency": "USD"}
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
        currency = info.get("currency", "USD") or "USD"
        payload = {"price": float(price or 0), "currency": currency}
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
    _cache_put(ticker, payload)
    return payload


# Bounded pool used by the async evaluate endpoint to fan out price lookups
# without blocking the event loop.
_price_pool = ThreadPoolExecutor(max_workers=PRICE_FETCH_WORKERS,
                                 thread_name_prefix="price")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    yield
    _price_pool.shutdown(wait=False)


app = FastAPI(
    title="PrimoGreedy Data API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_key(x_api_key: Optional[str] = Header(None)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SeenTickerIn(BaseModel):
    ticker: str
    region: str = "USA"


class TradeIn(BaseModel):
    ticker: str
    entry_price: float
    date: str  # YYYY-MM-DD
    verdict: str
    source: str = "unknown"
    position_size: float = 0.0
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    broker_status: str = "none"


class TradeFillUpdate(BaseModel):
    order_id: str
    fill_price: Optional[float] = None
    broker_status: str = "filled"


class AnalysisIn(BaseModel):
    """Structured analysis adhering to the 3-layer framework (proposal.md)."""
    ticker: str
    stage: Optional[str] = None            # RESEARCH|WATCHLIST|BUY_CANDIDATE|DISCARD
    verdict: Optional[str] = None          # STRONG BUY|BUY|WATCH|AVOID
    source: str = "unknown"
    price: Optional[float] = None
    currency: str = "USD"
    business_summary: Optional[str] = None
    financial_evolution: Optional[str] = None
    cash_flow: Optional[str] = None
    debt_liquidity: Optional[str] = None
    dilution: Optional[str] = None
    valuation_method: Optional[str] = None
    valuation_bear: Optional[str] = None
    valuation_base: Optional[str] = None
    valuation_bull: Optional[str] = None
    valuation_range_low: Optional[float] = None
    valuation_range_high: Optional[float] = None
    margin_of_safety: Optional[str] = None
    risks: Optional[str] = None
    invalidation_facts: Optional[str] = None
    macro_factors: Optional[str] = None
    portfolio_fit: Optional[str] = None
    sources: Optional[list] = None          # [{type,url,date,period}]
    data_quality: Optional[dict] = None     # {facts,estimates,inferences,unverified,pending}
    proposed_action: Optional[str] = None   # INITIATE|MAINTAIN|INCREASE|REDUCE|CLOSE|DISCARD|PENDING
    review_trigger: Optional[str] = None
    next_review_at: Optional[str] = None    # YYYY-MM-DD
    report_text: Optional[str] = None
    position_size: Optional[float] = None


class DecisionIn(BaseModel):
    action: Optional[str] = None
    position_size: Optional[float] = None
    entry_price: Optional[float] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request):
    """Liveness + dependency health.

    Returns ``status: "ok"`` when the DB is reachable.  ``degraded`` when the
    database answers but the analysis crons have gone stale (no new analysis
    within CRON_STALE_AFTER) or the DB is unreachable.
    """
    body = {
        "status": "ok",
        "db": DB_PATH,
        "price_cache_entries": len(_price_cache),
        "price_ttl_seconds": PRICE_TTL,
    }
    try:
        with get_db_ctx() as con:
            last = con.execute(
                "SELECT max(analyzed_at) FROM analyses"
            ).fetchone()[0]
            body["last_analysis_at"] = str(last) if last else None
            if last is not None:
                age = (datetime.now(timezone.utc)
                       - last.replace(tzinfo=timezone.utc)).total_seconds()
                body["last_analysis_age_seconds"] = round(age)
                body["crons_healthy"] = age < CRON_STALE_AFTER
                if not body["crons_healthy"]:
                    body["status"] = "degraded"
    except Exception as exc:
        logger.warning("health: DB check failed: %s", exc)
        body["status"] = "degraded"
        body["db_error"] = str(exc)
    return body


@app.get("/metrics")
def metrics(request: Request):
    """Lightweight JSON observability snapshot (no auth, like /health).

    Deliberately Prometheus-free: the VPS runs a single container and this is
    scraped by hand / by the monitoring agent.  Returns DB row counts, cron
    freshness, cache stats and process uptime.
    """
    out = {
        "price_cache_entries": len(_price_cache),
        "price_ttl_seconds": PRICE_TTL,
        "price_fetch_workers": PRICE_FETCH_WORKERS,
        "cron_stale_after_seconds": CRON_STALE_AFTER,
        "process_uptime_seconds": round(time.time() - _PROCESS_STARTED_AT),
    }
    try:
        with get_db_ctx() as con:
            for table in ("analyses", "paper_portfolio", "closed_positions",
                          "seen_tickers", "watchlist", "agent_runs"):
                out[f"{table}_rows"] = con.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            by_source = con.execute(
                "SELECT source, COUNT(*) FROM analyses GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
            out["analyses_by_source"] = {s: n for s, n in by_source}
            last = con.execute("SELECT max(analyzed_at) FROM analyses").fetchone()[0]
            out["last_analysis_at"] = str(last) if last else None

            # Review-gate backlog: actionable decisions vs. watches/rejections.
            wl = con.execute(
                """SELECT w.stage, w.status, a.proposed_action, COUNT(*)
                   FROM watchlist w JOIN analyses a ON a.id = w.analysis_id
                   GROUP BY 1, 2, 3"""
            ).fetchall()
            actionable = backlog = 0
            for stage, status, proposed, n in wl:
                if status != "active":
                    continue
                if _derive_action(stage, proposed) in _ACTIONABLE_ACTIONS:
                    actionable += n
                else:
                    backlog += n
            out["review_actionable"] = actionable
            out["review_non_actionable"] = backlog
    except Exception as exc:
        logger.warning("metrics: DB query failed: %s", exc)
        out["db_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Seen Tickers
# ---------------------------------------------------------------------------

@app.get("/seen-tickers")
def get_seen_tickers(x_api_key: Optional[str] = Header(None)):
    """Return seen tickers with verdict-aware TTL.

    BUY / STRONG BUY verdicts stay locked for 30 days.
    AVOID / WATCH / unknown verdicts expire after 14 days so the
    agent can re-evaluate tickers whose fundamentals may have changed.
    """
    if x_api_key:
        verify_key(x_api_key)

    long_cutoff = datetime.fromtimestamp(
        time.time() - MEMORY_TTL_LONG, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    short_cutoff = datetime.fromtimestamp(
        time.time() - MEMORY_TTL_SHORT, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")

    with get_db_ctx() as con:
        rows = con.execute("""
            WITH latest_verdicts AS (
                SELECT ticker, verdict,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM paper_portfolio
            )
            SELECT st.ticker, epoch(st.seen_at) AS ts
            FROM seen_tickers st
            LEFT JOIN latest_verdicts lv ON st.ticker = lv.ticker AND lv.rn = 1
            WHERE CASE
                WHEN lv.verdict IN ('BUY', 'STRONG BUY') THEN st.seen_at >= ?
                ELSE st.seen_at >= ?
            END
        """, [long_cutoff, short_cutoff]).fetchall()

    return {row[0]: row[1] for row in rows}


@app.post("/seen-tickers")
@limiter.limit("120/minute")
def mark_seen(request: Request, body: SeenTickerIn, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        con.execute(
            """INSERT INTO seen_tickers (ticker, region, seen_at)
               VALUES (?, ?, now())
               ON CONFLICT (ticker) DO UPDATE SET seen_at = now(), region = ?""",
            [body.ticker, body.region, body.region],
        )
    return {"status": "ok", "ticker": body.ticker}


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@app.get("/portfolio")
def get_portfolio(x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            """SELECT ticker, entry_price, date, verdict, source 
               FROM paper_portfolio 
               WHERE source NOT IN ('deploy-test', 'test_suite', 'vps_test', 'test', 'e2e_test', 'smoke_test')
                 AND ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
               ORDER BY date DESC"""
        ).fetchall()

    return [
        {
            "ticker": r[0],
            "entry_price": r[1],
            "date": str(r[2]),
            "verdict": r[3],
            "source": r[4],
        }
        for r in rows
    ]


@app.post("/portfolio")
@limiter.limit("60/minute")
def record_trade(request: Request, body: TradeIn, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        existing = con.execute(
            "SELECT COUNT(*) FROM paper_portfolio WHERE ticker = ?",
            [body.ticker],
        ).fetchone()[0]

        if existing > 0:
            con.execute(
                """UPDATE paper_portfolio
                   SET entry_price = ?, date = ?, verdict = ?, source = ?,
                       position_size = ?, order_id = ?, fill_price = ?,
                       broker_status = ?, created_at = current_timestamp
                   WHERE ticker = ?""",
                [body.entry_price, body.date, body.verdict, body.source,
                 body.position_size, body.order_id, body.fill_price,
                 body.broker_status, body.ticker],
            )
            return {"status": "updated", "ticker": body.ticker}

        try:
            con.execute(
                """INSERT INTO paper_portfolio
                   (ticker, entry_price, date, verdict, source, position_size, order_id, fill_price, broker_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [body.ticker, body.entry_price, body.date, body.verdict,
                 body.source, body.position_size, body.order_id,
                 body.fill_price, body.broker_status],
            )
        except duckdb.ConstraintException:
            return {"status": "duplicate", "ticker": body.ticker, "date": body.date}

    return {"status": "ok", "ticker": body.ticker}


@app.patch("/portfolio/{ticker}/fill")
@limiter.limit("60/minute")
def update_trade_fill(request: Request, ticker: str, body: TradeFillUpdate, x_api_key: str = Header(...)):
    """Update a trade's broker fill information after order execution."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        con.execute(
            """UPDATE paper_portfolio
               SET order_id = ?, fill_price = ?, broker_status = ?
               WHERE ticker = ? AND (order_id = ? OR date = current_date)""",
            [body.order_id, body.fill_price, body.broker_status,
             ticker, body.order_id],
        )
    return {"status": "ok", "ticker": ticker}


@app.delete("/portfolio/{ticker}")
@limiter.limit("60/minute")
def delete_portfolio_entry(request: Request, ticker: str, x_api_key: str = Header(...)):
    """Delete all portfolio entries for a given ticker."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        before = con.execute("SELECT COUNT(*) FROM paper_portfolio WHERE ticker = ?", [ticker]).fetchone()[0]
        con.execute("DELETE FROM paper_portfolio WHERE ticker = ?", [ticker])
    return {"status": "ok", "ticker": ticker, "deleted": before}


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(active|delisted)$")


@app.patch("/portfolio/{ticker}/status")
@limiter.limit("60/minute")
def update_portfolio_status(request: Request, ticker: str, body: StatusUpdate, x_api_key: str = Header(...)):
    """Mark a portfolio ticker as active or delisted."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        updated = con.execute(
            "SELECT COUNT(*) FROM paper_portfolio WHERE ticker = ?", [ticker]
        ).fetchone()[0]
        if updated:
            con.execute(
                "UPDATE paper_portfolio SET status = ? WHERE ticker = ?",
                [body.status, ticker],
            )
    return {"status": "ok", "ticker": ticker, "portfolio_status": body.status, "updated": updated}


@app.get("/portfolio/closed")
def list_closed_positions(x_api_key: Optional[str] = Header(None)):
    """Return archived positions closed via the review gate."""
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            """SELECT ticker, entry_price, exit_price, entry_date, closed_at,
                      verdict, source, gain_pct, analysis_id
               FROM closed_positions
               ORDER BY closed_at DESC"""
        ).fetchall()
    return [
        {
            "ticker": r[0], "entry_price": r[1], "exit_price": r[2],
            "entry_date": str(r[3]) if r[3] else None,
            "closed_at": str(r[4]) if r[4] else None,
            "verdict": r[5], "source": r[6], "gain_pct": r[7],
            "analysis_id": r[8],
        }
        for r in rows
    ]


@app.post("/portfolio/deduplicate")
@limiter.limit("10/minute")
def deduplicate_portfolio(request: Request, x_api_key: str = Header(...)):
    """Keep only the most recent entry per ticker, removing older duplicates."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        before = con.execute("SELECT COUNT(*) FROM paper_portfolio").fetchone()[0]
        con.execute("""
            DELETE FROM paper_portfolio
            WHERE rowid NOT IN (
                SELECT max(rowid) FROM paper_portfolio GROUP BY ticker
            )
        """)
        after = con.execute("SELECT COUNT(*) FROM paper_portfolio").fetchone()[0]
    return {"status": "ok", "before": before, "after": after, "removed": before - after}


# ---------------------------------------------------------------------------
# Analyses & Watchlist (3-layer framework: research -> watchlist -> decision)
# ---------------------------------------------------------------------------

_ANALYSIS_COLS = [
    "id", "ticker", "analyzed_at", "stage", "verdict", "source",
    "price", "currency", "business_summary", "financial_evolution", "cash_flow",
    "debt_liquidity", "dilution", "valuation_method", "valuation_bear",
    "valuation_base", "valuation_bull", "valuation_range_low",
    "valuation_range_high", "margin_of_safety", "risks", "invalidation_facts",
    "macro_factors", "portfolio_fit", "sources", "data_quality",
    "proposed_action", "review_trigger", "next_review_at", "report_text",
    "position_size",
]

_ANALYSIS_SELECT = ", ".join(_ANALYSIS_COLS)
_ANALYSIS_SELECT_A = ", ".join(f"a.{c}" for c in _ANALYSIS_COLS)


def _parse_json(val):
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        return val


def _analysis_dict(r):
    d = dict(zip(_ANALYSIS_COLS, r))
    d["analyzed_at"] = str(d["analyzed_at"])
    d["sources"] = _parse_json(d.get("sources"))
    d["data_quality"] = _parse_json(d.get("data_quality"))
    if d.get("next_review_at"):
        d["next_review_at"] = str(d["next_review_at"])
    return d


def _derive_stage(verdict: str) -> str:
    """Map legacy verdict labels onto the 4-stage funnel."""
    upper = (verdict or "").upper()
    if "STRONG BUY" in upper or "BUY" in upper:
        return "BUY_CANDIDATE"
    if "WATCH" in upper:
        return "WATCHLIST"
    if "AVOID" in upper or "DISCARD" in upper:
        return "DISCARD"
    return "RESEARCH"


def _stage_to_verdict(stage: str) -> str:
    return {
        "BUY_CANDIDATE": "BUY",
        "WATCHLIST": "WATCH",
        "DISCARD": "AVOID",
        "RESEARCH": "WATCH",
    }.get(stage or "", "WATCH")


def _derive_action(stage: str, proposed: str) -> str:
    if proposed and proposed not in ("PENDING", ""):
        return proposed
    return {
        "BUY_CANDIDATE": "INITIATE",
        "WATCHLIST": "PENDING",
        "RESEARCH": "PENDING",
        "DISCARD": "DISCARD",
    }.get(stage or "", "PENDING")


def _coerce_date(val):
    """Best-effort parse of an LLM-supplied date into YYYY-MM-DD, else None.

    Free-tier models often return prose (e.g. "after Q3 earnings") for
    ``next_review_at``.  Binding that straight into the DATE column raised a
    DuckDB ConversionException and 500'd the entire analysis, so the analysis
    was silently lost.  Anything unparseable becomes NULL instead.
    """
    if not val:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
                "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning("Unparseable next_review_at %r -> NULL", s[:80])
    return None


@app.post("/analyses")
@limiter.limit("120/minute")
def submit_analysis(request: Request, body: AnalysisIn, x_api_key: str = Header(...)):
    """Store a structured analysis and stage the ticker on the watchlist."""
    verify_key(x_api_key)
    stage = body.stage or _derive_stage(body.verdict)
    next_review = _coerce_date(body.next_review_at)

    with _db_write_lock, get_db_ctx() as con:
        insert_cols = [c for c in _ANALYSIS_COLS if c not in ("id", "analyzed_at")]
        placeholders = ", ".join(["?"] * len(insert_cols))
        con.execute(
            f"INSERT INTO analyses ({', '.join(insert_cols)}) VALUES ({placeholders})",
            [
                body.ticker, stage, body.verdict, body.source,
                body.price, body.currency, body.business_summary,
                body.financial_evolution, body.cash_flow, body.debt_liquidity,
                body.dilution, body.valuation_method, body.valuation_bear,
                body.valuation_base, body.valuation_bull, body.valuation_range_low,
                body.valuation_range_high, body.margin_of_safety, body.risks,
                body.invalidation_facts, body.macro_factors, body.portfolio_fit,
                json.dumps(body.sources) if body.sources else None,
                json.dumps(body.data_quality) if body.data_quality else None,
                body.proposed_action, body.review_trigger, next_review,
                body.report_text, body.position_size,
            ],
        )
        aid = con.execute("SELECT currval('analysis_seq')").fetchone()[0]

        reason = body.verdict or ""
        if body.valuation_base:
            reason = f"{reason} | {body.valuation_base}"
        con.execute(
            """INSERT INTO watchlist (ticker, stage, source, analysis_id, reason, added_at, status)
               VALUES (?, ?, ?, ?, ?, now(), 'active')
               ON CONFLICT (ticker) DO UPDATE SET
                   stage = ?, source = ?, analysis_id = ?, reason = ?,
                   added_at = now(), status = 'active'""",
            [body.ticker, stage, body.source, aid, reason,
             stage, body.source, aid, reason],
        )

    return {"status": "ok", "ticker": body.ticker, "analysis_id": aid, "stage": stage}


@app.get("/analyses")
def list_analyses(ticker: Optional[str] = None, limit: int = 50,
                  x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        if ticker:
            rows = con.execute(
                f"SELECT {_ANALYSIS_SELECT} FROM analyses WHERE ticker = ? ORDER BY analyzed_at DESC LIMIT ?",
                [ticker, min(limit, 200)],
            ).fetchall()
        else:
            rows = con.execute(
                f"SELECT {_ANALYSIS_SELECT} FROM analyses ORDER BY analyzed_at DESC LIMIT ?",
                [min(limit, 200)],
            ).fetchall()
    return [_analysis_dict(r) for r in rows]


@app.get("/analyses/{ticker}")
def get_analysis(ticker: str, x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        row = con.execute(
            f"SELECT {_ANALYSIS_SELECT} FROM analyses WHERE ticker = ? ORDER BY analyzed_at DESC LIMIT 1",
            [ticker],
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No analysis found")
    return _analysis_dict(row)


@app.get("/watchlist")
def get_watchlist(x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            f"""SELECT {_ANALYSIS_SELECT_A}, w.stage AS wl_stage, w.status AS wl_status
                FROM watchlist w
                JOIN analyses a ON a.id = w.analysis_id
                WHERE w.ticker NOT LIKE 'TEST%'
                ORDER BY w.added_at DESC"""
        ).fetchall()
    out = []
    for r in rows:
        d = _analysis_dict(r[:len(_ANALYSIS_COLS)])
        d["watchlist_stage"] = r[len(_ANALYSIS_COLS)]
        d["watchlist_status"] = r[len(_ANALYSIS_COLS) + 1]
        d["pending_action"] = _derive_action(d["watchlist_stage"], d.get("proposed_action"))
        out.append(d)
    return out


# Actions that genuinely require a human decision.  PENDING (still watching)
# and DISCARD (the system already rejected it) do not.  Leaving them in made
# 102 of 125 "pending" rows pure noise, burying the real 23 decisions.
_ACTIONABLE_ACTIONS = ("INITIATE", "INCREASE", "REDUCE", "CLOSE")


@app.get("/portfolio/reviews")
def portfolio_reviews(include: str = "actionable",
                      x_api_key: Optional[str] = Header(None)):
    """Watchlist rows awaiting a human accept/close decision.

    Defaults to *actionable* rows only (INITIATE/INCREASE/REDUCE/CLOSE),
    oldest first, so the queue reflects real work.  Pass ``include=all`` to
    inspect the complete funnel (watches and rejections included).
    """
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            f"""SELECT {_ANALYSIS_SELECT_A}, w.stage AS wl_stage,
                       w.status AS wl_status, w.added_at
                FROM watchlist w
                JOIN analyses a ON a.id = w.analysis_id
                WHERE w.status = 'active' AND w.ticker NOT LIKE 'TEST%'
                ORDER BY w.added_at ASC"""
        ).fetchall()
    out = []
    for r in rows:
        d = _analysis_dict(r[:len(_ANALYSIS_COLS)])
        d["watchlist_stage"] = r[len(_ANALYSIS_COLS)]
        d["watchlist_status"] = r[len(_ANALYSIS_COLS) + 1]
        added_at = r[len(_ANALYSIS_COLS) + 2]
        d["pending_action"] = _derive_action(d["watchlist_stage"], d.get("proposed_action"))
        d["age_days"] = (datetime.now() - added_at).days if added_at else None
        d["needs_review"] = d["pending_action"] in _ACTIONABLE_ACTIONS
        if include != "all" and not d["needs_review"]:
            continue
        out.append(d)
    return out


@app.post("/watchlist/{ticker}/accept")
@limiter.limit("60/minute")
def accept_decision(request: Request, ticker: str, body: DecisionIn, x_api_key: str = Header(...)):
    """Accept an analysis: record the paper trade, close the review. No auto orders."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        wl = con.execute(
            "SELECT ticker, stage, source, analysis_id, status FROM watchlist WHERE ticker = ?",
            [ticker],
        ).fetchone()
        if not wl:
            raise HTTPException(status_code=404, detail="Ticker not on watchlist")
        if wl[4] in ("accepted", "dismissed"):
            return {"status": "already_" + wl[4], "ticker": ticker}

        analysis = con.execute(
            f"SELECT {_ANALYSIS_SELECT} FROM analyses WHERE id = ?",
            [wl[3]],
        ).fetchone()
        if not analysis:
            analysis = con.execute(
                f"SELECT {_ANALYSIS_SELECT} FROM analyses WHERE ticker = ? ORDER BY analyzed_at DESC LIMIT 1",
                [ticker],
            ).fetchone()
        if not analysis:
            raise HTTPException(status_code=404, detail="No analysis available")
        d = _analysis_dict(analysis)

        action = body.action or d.get("proposed_action") or _derive_action(d["stage"], None)

        # CLOSE means "exit this position".  Accepting it must REMOVE the
        # paper-portfolio row, not insert/refresh one at the current price
        # (the old code did the latter, which silently opened a position
        # instead of closing it).
        if action == "CLOSE":
            existing_rows = con.execute(
                """SELECT entry_price, date, verdict, source
                   FROM paper_portfolio WHERE ticker = ?""",
                [ticker],
            ).fetchall()
            exit_price = body.entry_price or d.get("price")
            for ep, edate, everdict, esource in existing_rows:
                gain = (
                    round(((exit_price - ep) / ep) * 100, 2)
                    if (exit_price is not None and ep and ep > 0)
                    else None
                )
                con.execute(
                    """INSERT INTO closed_positions
                       (ticker, entry_price, exit_price, entry_date, verdict, source, gain_pct, analysis_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [ticker, ep, exit_price, edate, everdict, esource, gain, d["id"]],
                )
            con.execute("DELETE FROM paper_portfolio WHERE ticker = ?", [ticker])
            con.execute(
                "UPDATE watchlist SET status = 'accepted' WHERE ticker = ?", [ticker]
            )
            return {
                "status": "closed", "ticker": ticker, "action": "CLOSE",
                "closed_positions": len(existing_rows), "exit_price": exit_price,
                "analysis_id": d["id"],
            }

        entry_price = body.entry_price or d.get("price")
        if not entry_price:
            raise HTTPException(status_code=400, detail="No price: pass entry_price or set price in the analysis")
        verdict_label = d.get("verdict") or _stage_to_verdict(d.get("stage"))
        position_size = body.position_size if body.position_size is not None else (d.get("position_size") or 0.0)
        today = datetime.now().strftime("%Y-%m-%d")

        existing = con.execute(
            "SELECT COUNT(*) FROM paper_portfolio WHERE ticker = ?",
            [ticker],
        ).fetchone()[0]
        if existing > 0:
            con.execute(
                """UPDATE paper_portfolio
                   SET entry_price = ?, date = ?, verdict = ?, source = ?,
                       position_size = ?, broker_status = 'accepted', created_at = current_timestamp,
                       analysis_id = ?
                   WHERE ticker = ?""",
                [entry_price, today, verdict_label, wl[2],
                 position_size, d["id"], ticker],
            )
        else:
            con.execute(
                """INSERT INTO paper_portfolio
                   (ticker, entry_price, date, verdict, source, position_size,
                    broker_status, analysis_id)
                   VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?)""",
                [ticker, entry_price, today, verdict_label, wl[2],
                 position_size, d["id"]],
            )
        con.execute(
            "UPDATE watchlist SET status = 'accepted' WHERE ticker = ?",
            [ticker],
        )

    return {
        "status": "accepted", "ticker": ticker, "action": action,
        "entry_price": entry_price, "position_size": position_size,
        "analysis_id": d["id"],
    }


@app.post("/watchlist/{ticker}/dismiss")
@limiter.limit("60/minute")
def dismiss_decision(request: Request, ticker: str, x_api_key: str = Header(...)):
    """Dismiss an analysis without recording any trade."""
    verify_key(x_api_key)
    with _db_write_lock, get_db_ctx() as con:
        con.execute("UPDATE watchlist SET status = 'dismissed' WHERE ticker = ?", [ticker])
    return {"status": "dismissed", "ticker": ticker}


@app.get("/portfolio/evaluate")
@limiter.limit("30/minute")
async def evaluate_portfolio(request: Request, x_api_key: Optional[str] = Header(None)):
    """Fetch live prices and compute P&L for all portfolio entries.

    Prices are served from a TTL cache and fetched concurrently in a bounded
    thread pool, so this never blocks the event loop and never stampedes
    yfinance for a large portfolio.
    """
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            """SELECT ticker, entry_price, date, verdict, source 
               FROM paper_portfolio 
               WHERE source NOT IN ('deploy-test', 'test_suite', 'vps_test', 'test', 'e2e_test', 'smoke_test')
                 AND ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
               ORDER BY date"""
        ).fetchall()

    if not rows:
        return {"report": "Paper Portfolio is empty.", "trades": []}

    prices = await _fetch_prices_for_tickers([r[0] for r in rows])

    trades = []
    total_roi = 0.0
    winners = 0
    valid = 0

    for r in rows:
        ticker, entry, date, verdict, source = r[0], r[1], str(r[2]), r[3], r[4]
        info = prices.get(ticker) or {}
        price = info.get("price", 0) or 0
        currency = info.get("currency", "USD")

        # Pence → Pounds
        if ticker.endswith(".L") or currency in ("GBp", "GBX"):
            price = price / 100

        if price > 0 and entry > 0:
            gain = ((price - entry) / entry) * 100
            if gain > 0:
                winners += 1
            total_roi += gain
            valid += 1
            trades.append({
                "ticker": ticker, "date": date, "entry": entry,
                "current": round(price, 2), "gain_pct": round(gain, 2),
                "verdict": verdict,
            })
        else:
            trades.append({
                "ticker": ticker, "date": date, "entry": entry,
                "current": None, "gain_pct": None, "verdict": verdict,
            })

    avg_roi = total_roi / valid if valid else 0
    win_rate = (winners / valid * 100) if valid else 0

    return {
        "total_calls": len(rows),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_roi, 2),
        "trades": trades,
    }


def _fetch_benchmarks() -> dict[str, dict]:
    """Fetch 52-week change for IWM (small-cap) and SPY, TTL-cached."""
    out = {}
    for sym in ("IWM", "SPY"):
        key = f"__BENCH__{sym}"
        cached = _cache_get(key)
        if cached is not None:
            out[sym] = cached
            continue
        payload = {"price": 0.0, "change_52w_pct": None}
        try:
            info = yf.Ticker(sym).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            chg = info.get("52WeekChange")
            if chg is None:
                chg = info.get("SandP52WeekChange")
            payload = {
                "price": round(float(price or 0), 2),
                "change_52w_pct": round(float(chg) * 100, 2) if chg is not None else None,
            }
        except Exception as exc:
            logger.debug("Benchmark fetch failed for %s: %s", sym, exc)
        _cache_put(key, payload)
        out[sym] = payload
    return out


async def _fetch_prices_for_tickers(tickers: list[str]) -> dict[str, dict]:
    """Fetch live prices for unique tickers concurrently via the TTL cache."""
    unique = list(dict.fromkeys(tickers))

    async def _one(t: str) -> dict:
        return await run_in_threadpool(_fetch_price, t)

    results = await asyncio.gather(*(_one(t) for t in unique))
    return dict(zip(unique, results))


def _region_for_ticker(ticker: str) -> str:
    """Infer market region from ticker exchange suffix."""
    upper = ticker.upper()
    if upper.endswith(".L"):
        return "UK"
    if upper.endswith((".TO", ".V", ".CN", ".NE")):
        return "Canada"
    if upper.endswith(".AX"):
        return "Australia"
    if "." in upper:
        return "Other"
    return "USA"


def _summarize_group(items: list[dict]) -> dict:
    """Aggregate win rate and mean/median return for a group of trades."""
    gains = sorted(t["gain_pct"] for t in items if t["gain_pct"] is not None)
    priced = len(gains)
    winners = sum(1 for g in gains if g > 0)
    avg_ret = round(sum(gains) / priced, 2) if priced else 0.0
    if priced == 0:
        med_ret = 0.0
    elif priced % 2 == 1:
        med_ret = round(gains[priced // 2], 2)
    else:
        med_ret = round((gains[priced // 2 - 1] + gains[priced // 2]) / 2, 2)
    return {
        "count": len(items),
        "priced_count": priced,
        "winners": winners,
        "win_rate_pct": round((winners / priced * 100), 1) if priced else 0.0,
        "avg_return_pct": avg_ret,
        "median_return_pct": med_ret,
    }


@app.get("/portfolio/attribution")
@limiter.limit("30/minute")
async def portfolio_attribution(request: Request, x_api_key: Optional[str] = Header(None)):
    """Break down realized portfolio P&L by region, verdict, and source."""
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            """SELECT ticker, entry_price, date, verdict, source, COALESCE(status, 'active')
               FROM paper_portfolio
               WHERE source NOT IN ('deploy-test', 'test_suite', 'vps_test', 'test', 'e2e_test', 'smoke_test')
                 AND ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
               ORDER BY date"""
        ).fetchall()
        closed_rows = con.execute(
            "SELECT gain_pct FROM closed_positions"
        ).fetchall()

    if not rows:
        return {
            "total_calls": 0,
            "overall": _summarize_group([]),
            "by_region": {},
            "by_verdict": {},
            "by_source": {},
            "top_winners": [],
            "top_losers": [],
            "closed_summary": _summarize_group([{"gain_pct": cr[0]} for cr in closed_rows]),
            "benchmarks": {},
        }

    prices = await _fetch_prices_for_tickers([r[0] for r in rows])
    evaluated = []
    for ticker, entry, date, verdict, source, pstatus in rows:
        info = prices.get(ticker) or {}
        price = info.get("price", 0) or 0
        currency = info.get("currency", "USD")
        if ticker.endswith(".L") or currency in ("GBp", "GBX"):
            price = price / 100
        gain = round(((price - entry) / entry) * 100, 2) if (price > 0 and entry > 0) else None
        evaluated.append({
            "ticker": ticker,
            "region": _region_for_ticker(ticker),
            "verdict": verdict or "UNKNOWN",
            "source": source or "unknown",
            "status": pstatus,
            "date": str(date),
            "entry": entry,
            "current": round(price, 4) if price > 0 else None,
            "gain_pct": gain,
        })

    def _group_by(key: str) -> dict[str, dict]:
        buckets: dict[str, list[dict]] = {}
        for item in evaluated:
            buckets.setdefault(item[key], []).append(item)
        return {k: _summarize_group(v) for k, v in sorted(buckets.items())}

    priced_items = sorted(
        (t for t in evaluated if t["gain_pct"] is not None),
        key=lambda t: t["gain_pct"],
        reverse=True,
    )

    benchmarks = await run_in_threadpool(_fetch_benchmarks)
    return {
        "total_calls": len(evaluated),
        "overall": _summarize_group(evaluated),
        "by_region": _group_by("region"),
        "by_verdict": _group_by("verdict"),
        "by_source": _group_by("source"),
        "top_winners": priced_items[:5],
        "top_losers": list(reversed(priced_items[-5:])) if priced_items else [],
        "closed_summary": _summarize_group([{"gain_pct": cr[0]} for cr in closed_rows]),
        "benchmarks": benchmarks,
    }


# ---------------------------------------------------------------------------
# Portfolio Summary (lightweight, no yFinance calls)
# ---------------------------------------------------------------------------

@app.get("/portfolio/summary")
def portfolio_summary(x_api_key: Optional[str] = Header(None)):
    """Aggregated portfolio stats without live price lookups."""
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        trades = con.execute(
            """SELECT ticker, entry_price, date, verdict, source,
                      position_size, order_id, fill_price, broker_status,
                      COALESCE(status, 'active')
               FROM paper_portfolio 
               WHERE source NOT IN ('deploy-test', 'test_suite', 'vps_test', 'test', 'e2e_test', 'smoke_test')
                 AND ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
               ORDER BY date DESC"""
        ).fetchall()

        seen_count = con.execute("""
            SELECT COUNT(*) FROM seen_tickers
            WHERE ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
        """).fetchone()[0]

        runs = con.execute(
            """SELECT id, ticker, timestamp, status, region
               FROM agent_runs 
               WHERE ticker NOT LIKE 'TEST%' AND ticker != 'PYTH' AND ticker != 'GHACT'
               ORDER BY timestamp DESC LIMIT 20"""
        ).fetchall()

    by_verdict: dict[str, int] = {}
    by_source: dict[str, int] = {}
    recent_trades = []

    for r in trades:
        v = _extract_verdict_label(r[3])
        by_verdict[v] = by_verdict.get(v, 0) + 1
        src = r[4] or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

        trade_obj = {
            "ticker": r[0], "entry_price": r[1], "date": str(r[2]),
            "verdict": v, "source": src,
            "position_size": r[5], "order_id": r[6],
            "fill_price": r[7], "broker_status": r[8] or "none",
            "status": r[9] or "active",
        }
        recent_trades.append(trade_obj)

    recent_runs = [
        {"id": r[0], "ticker": r[1], "timestamp": str(r[2]),
         "status": r[3], "region": r[4]}
        for r in runs
    ]

    return {
        "total_trades": len(trades),
        "by_verdict": by_verdict,
        "by_source": by_source,
        "seen_tickers_count": seen_count,
        "recent_trades": recent_trades,
        "recent_runs": recent_runs,
    }


def _extract_verdict_label(verdict_text: str) -> str:
    """Pull the verdict keyword from the full verdict text."""
    upper = (verdict_text or "").upper()
    if "STRONG BUY" in upper:
        return "STRONG BUY"
    if "BUY" in upper:
        return "BUY"
    if "WATCH" in upper:
        return "WATCH"
    if "AVOID" in upper:
        return "AVOID"
    return "OTHER"


# ---------------------------------------------------------------------------
# Dashboard (public — no API key, behind Tailscale)
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the live portfolio dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Dashboard HTML (inline — no static file dependencies)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PrimoGreedy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e17;--s1:#111827;--s2:#1a2236;--bd:#1e293b;--cy:#22d3ee;--pu:#a78bfa;
--gn:#34d399;--rd:#f87171;--yl:#fbbf24;--tx:#e2e8f0;--td:#94a3b8;--tw:#f8fafc}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--tx);line-height:1.5}
.wrap{max-width:1320px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;flex-wrap:wrap;gap:12px}
header h1{font-size:28px;font-weight:800;letter-spacing:-1px}
header h1 span{background:linear-gradient(135deg,var(--cy),var(--pu));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hdr-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.badge{font-size:12px;padding:5px 14px;border:1px solid var(--bd);border-radius:99px;color:var(--gn);
background:rgba(52,211,153,.06);backdrop-filter:blur(8px);display:inline-flex;align-items:center;gap:6px}
.badge.warn{color:var(--yl);background:rgba(251,191,36,.08)}
.badge .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
.btn{background:var(--s2);border:1px solid var(--bd);border-radius:8px;color:var(--cy);padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;transition:border-color .2s}
.btn:hover{border-color:var(--cy)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:20px;
position:relative;overflow:hidden;transition:border-color .3s,transform .3s;animation:fadeUp .4s ease-out both}
.card:hover{border-color:var(--cy);transform:translateY(-2px)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,var(--cy),var(--pu),var(--gn));opacity:.6}
.card .label{font-size:11px;text-transform:uppercase;letter-spacing:1.4px;color:var(--td);margin-bottom:6px;font-weight:600}
.card .val{font-size:30px;font-weight:800;color:var(--tw);letter-spacing:-1px}
.card .sub{font-size:12px;color:var(--td);margin-top:6px}
.card .val.green{background:linear-gradient(135deg,#34d399,#2dd4bf);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.card .val.red{color:var(--rd)}

/* Panels */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}
.panel{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:22px;animation:fadeUp .5s ease-out both}
.panel h2{font-size:15px;margin-bottom:14px;color:var(--tw);font-weight:700;letter-spacing:-.3px;display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
canvas{max-height:250px}

/* Attribution & Movers */
.movers{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:600px){.movers{grid-template-columns:1fr}}
.mover-box{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:12px}
.mover-box h3{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--td);margin-bottom:8px}
.mover-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(30,41,59,.5);font-size:12px}
.mover-row:last-child{border-bottom:none}
.reg-pill{font-size:10px;padding:2px 6px;border-radius:6px;background:rgba(148,163,184,.12);color:var(--td);font-weight:600}

/* Filters */
.filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.input{background:var(--s2);border:1px solid var(--bd);border-radius:8px;color:var(--tx);padding:7px 11px;font-size:12px;font-family:'JetBrains Mono',monospace}
.input:focus{outline:none;border-color:var(--cy)}
select.input{font-family:'Inter',sans-serif;cursor:pointer}

/* Table */
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 10px;border-bottom:2px solid var(--bd);color:var(--td);font-weight:700;
text-transform:uppercase;letter-spacing:.8px;font-size:10px;cursor:pointer;user-select:none;transition:color .2s;white-space:nowrap}
th:hover{color:var(--cy)}
th.sorted::after{content:' \25be';color:var(--cy)}
th.sorted.asc::after{content:' \25b4'}
td{padding:9px 10px;border-bottom:1px solid rgba(30,41,59,.4);white-space:nowrap}
tr:hover td{background:rgba(34,211,238,.04)}
.ticker{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--cy);text-decoration:none;transition:color .2s}
.ticker:hover{color:var(--pu)}
.verdict-buy{color:var(--gn);font-weight:700}.verdict-watch{color:var(--yl);font-weight:700}
.verdict-avoid{color:var(--rd);font-weight:700}.verdict-strong{color:#2dd4bf;font-weight:800}
.gain-pos{color:var(--gn);font-weight:600}.gain-neg{color:var(--rd);font-weight:600}
.broker-filled{color:var(--gn);font-size:11px;font-weight:600}.broker-pending{color:var(--yl);font-size:11px}
.broker-none{color:var(--td);font-size:11px}

.refresh{font-size:11px;color:var(--td);text-align:center;margin-top:20px;padding:14px 0}
.loading{text-align:center;padding:40px;color:var(--td)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Primo<span>Greedy</span> Dashboard</h1>
  <div class="hdr-right">
    <div class="badge" id="healthBadge"><span class="dot"></span>Checking pipeline...</div>
    <button class="btn" id="refreshBtn">Refresh Now</button>
  </div>
</header>

<div class="cards" id="cards"><div class="loading">Loading...</div></div>

<div class="grid2">
  <div class="panel"><h2>Verdict Distribution</h2><canvas id="verdictChart"></canvas></div>
  <div class="panel"><h2>Position Sizing (Kelly %)</h2><canvas id="sizingChart"></canvas></div>
</div>

<div class="grid2">
  <div class="panel">
    <h2>Attribution by Region <span id="attrNote" style="color:var(--td);font-weight:400;font-size:12px"></span></h2>
    <canvas id="regionChart" style="margin-bottom:14px"></canvas>
    <div style="overflow-x:auto">
      <table>
        <thead><tr><th>Region</th><th>Trades</th><th>Priced</th><th>Win Rate</th><th>Avg P/L</th><th>Median P/L</th></tr></thead>
        <tbody id="regionBody"></tbody>
      </table>
    </div>
  </div>
  <div class="panel">
    <h2>Top Movers</h2>
    <div class="movers">
      <div class="mover-box"><h3>Top 5 Winners</h3><div id="topWinners"></div></div>
      <div class="mover-box"><h3>Top 5 Losers</h3><div id="topLosers"></div></div>
    </div>
  </div>
</div>

<div class="panel" style="margin-bottom:28px">
  <h2>Portfolio <span id="tradeCount" style="color:var(--td);font-weight:400;font-size:13px"></span></h2>
  <div class="filters">
    <input id="filterSearch" class="input" type="text" placeholder="Filter ticker (e.g. HCWC, .L)..." style="min-width:210px"/>
    <select id="filterRegion" class="input">
      <option value="ALL">All Regions</option>
      <option value="USA">USA</option>
      <option value="UK">UK (.L)</option>
      <option value="Canada">Canada (.TO/.V)</option>
      <option value="Australia">Australia (.AX)</option>
    </select>
    <select id="filterVerdict" class="input">
      <option value="ALL">All Verdicts</option>
      <option value="STRONG BUY">STRONG BUY</option>
      <option value="BUY">BUY</option>
      <option value="WATCH">WATCH</option>
    </select>
    <select id="filterOutcome" class="input">
      <option value="ALL">All Outcomes</option>
      <option value="WIN">Winners (&gt;0%)</option>
      <option value="LOSS">Losers (&le;0%)</option>
      <option value="UNPRICED">Unpriced</option>
    </select>
  </div>
  <div style="overflow-x:auto">
    <table id="portfolioTable">
      <thead><tr>
        <th data-col="ticker">Ticker</th><th data-col="region">Region</th><th data-col="date">Date</th>
        <th data-col="entry_price">Entry</th><th data-col="current">Now</th>
        <th data-col="gain_pct">P/L</th><th data-col="verdict">Verdict</th>
        <th data-col="position_size">Kelly %</th><th data-col="broker_status">Broker</th>
        <th data-col="source">Source</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div class="panel" style="margin-bottom:28px">
  <h2>Watchlist &amp; Reviews &mdash; review gate <span id="reviewCount" style="color:var(--td);font-weight:400;font-size:13px"></span></h2>
  <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <input id="apiKey" class="input" type="password" placeholder="API key (only needed to accept/dismiss)" style="min-width:250px"/>
    <button id="saveKey" class="btn">Save key</button>
    <span id="keyHint" style="color:var(--td);font-size:11px"></span>
  </div>
  <div style="overflow-x:auto">
    <table id="reviewTable">
      <thead><tr>
        <th>Ticker</th><th>Stage</th><th>Verdict</th><th>Pending Action</th>
        <th>Price</th><th>Val. Range</th><th>Margin of Safety</th>
        <th>Sources</th><th>Next Review</th><th>Decision</th>
      </tr></thead>
      <tbody id="reviewBody"></tbody>
    </table>
  </div>
</div>

<footer style="text-align:center;padding:24px 0 8px;border-top:1px solid var(--bd);margin-top:24px">
  <div style="margin-bottom:10px">
    <a href="https://ciscoponce.github.io/primogreedy/" target="_blank" rel="noopener"
       style="color:var(--cy);text-decoration:none;font-size:13px;font-weight:600;margin:0 16px">About the Project</a>
    <a href="https://github.com/CiscoPonce/primogreedy" target="_blank" rel="noopener"
       style="color:var(--pu);text-decoration:none;font-size:13px;font-weight:600;margin:0 16px">GitHub Repo</a>
    <a href="/metrics" target="_blank" rel="noopener"
       style="color:var(--gn);text-decoration:none;font-size:13px;font-weight:600;margin:0 16px">Raw Metrics</a>
  </div>
  <div class="refresh" id="refreshNote"></div>
</footer>
</div>

<script>
const H = {};
let sortCol = 'date', sortAsc = false;
let cachedTrades = [];
let cachedEvals = [];

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/[&<>"']/g, function(m) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m];
  });
}

function regionOf(ticker) {
  const u = (ticker || '').toUpperCase();
  if (u.endsWith('.L')) return 'UK';
  if (u.endsWith('.TO') || u.endsWith('.V') || u.endsWith('.CN') || u.endsWith('.NE')) return 'Canada';
  if (u.endsWith('.AX')) return 'Australia';
  if (u.includes('.')) return 'Other';
  return 'USA';
}

function currencyPrefix(ticker) {
  const u = (ticker || '').toUpperCase();
  if (u.endsWith('.L')) return '\u00a3';
  if (u.endsWith('.TO') || u.endsWith('.V') || u.endsWith('.CN') || u.endsWith('.NE')) return 'C$';
  if (u.endsWith('.AX')) return 'A$';
  return '$';
}

function fmtPrice(val, ticker) {
  if (val === null || val === undefined || isNaN(val)) return '\u2014';
  const n = Number(val);
  return currencyPrefix(ticker) + (Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2));
}

function fmtAge(secs) {
  if (secs === null || secs === undefined) return 'unknown';
  if (secs < 3600) return Math.round(secs / 60) + 'm ago';
  return (secs / 3600).toFixed(1) + 'h ago';
}

async function load() {
  try {
    const [sumR, seenR, evalR, attrR, healthR, metricsR] = await Promise.all([
      fetch('/portfolio/summary', {headers: H}),
      fetch('/seen-tickers', {headers: H}),
      fetch('/portfolio/evaluate', {headers: H}),
      fetch('/portfolio/attribution', {headers: H}),
      fetch('/health', {headers: H}),
      fetch('/metrics', {headers: H})
    ]);
    const sum = await sumR.json();
    const seen = await seenR.json();
    const evalResult = await evalR.json();
    const attr = await attrR.json();
    const health = await healthR.json();
    const metrics = await metricsR.json();

    cachedTrades = sum.recent_trades || [];
    cachedEvals = evalResult.trades || [];

    renderHealth(health);
    renderCards(sum, seen, evalResult, attr, metrics);
    renderVerdictChart(sum.by_verdict);
    renderSizingChart(cachedTrades);
    renderAttribution(attr);
    renderTable();

    document.getElementById('refreshNote').textContent =
      'Last refresh: ' + new Date().toLocaleTimeString() + ' \u2014 auto-refreshes every 5 min';
  } catch(e) {
    document.getElementById('cards').innerHTML =
      '<div class="loading">Error loading data: ' + escapeHtml(e.message) + '</div>';
  }
}

function renderHealth(h) {
  const badge = document.getElementById('healthBadge');
  const ok = h && h.status === 'ok' && h.crons_healthy;
  badge.className = ok ? 'badge' : 'badge warn';
  const age = h && h.last_analysis_age_seconds !== undefined ? fmtAge(h.last_analysis_age_seconds) : 'n/a';
  badge.innerHTML = `<span class="dot"></span>${ok ? 'Healthy' : 'Degraded'} \u00b7 Last cron ${escapeHtml(age)}`;
}

function renderCards(sum, seen, ev, attr, metrics) {
  const buys = (sum.by_verdict['BUY']||0) + (sum.by_verdict['STRONG BUY']||0);
  const watches = sum.by_verdict['WATCH']||0;
  const avoids = sum.by_verdict['AVOID']||0;
  const seenCount = Object.keys(seen).length;
  const ov = (attr && attr.overall) ? attr.overall : {};
  const avgReturn = ov.avg_return_pct !== undefined ? ov.avg_return_pct : (ev.avg_return || 0);
  const medReturn = ov.median_return_pct !== undefined ? ov.median_return_pct : 0;
  const winRate = ov.win_rate_pct !== undefined ? ov.win_rate_pct : (ev.win_rate || 0);
  const returnClass = avgReturn >= 0 ? 'green' : 'red';
  const returnSign = avgReturn >= 0 ? '+' : '';
  const medSign = medReturn >= 0 ? '+' : '';
  const sourcesStr = Object.entries(sum.by_source||{}).map(([k,v])=>escapeHtml(k)+': '+v).join(' \u00b7 ');
  document.getElementById('cards').innerHTML = `
    <div class="card"><div class="label">Total Trades</div><div class="val">${sum.total_trades}</div>
      <div class="sub">${buys} buys \u00b7 ${watches} watch \u00b7 ${avoids} avoid</div></div>
    <div class="card"><div class="label">Win Rate</div>
      <div class="val green">${winRate.toFixed(1)}%</div>
      <div class="sub">${ov.winners || 0} winners of ${ov.priced_count || 0} priced</div></div>
    <div class="card"><div class="label">Mean / Median P/L</div>
      <div class="val ${returnClass}">${returnSign}${avgReturn.toFixed(1)}%</div>
      <div class="sub">Median: ${medSign}${medReturn.toFixed(1)}%${(() => {
        const b = (attr && attr.benchmarks) ? attr.benchmarks : {};
        const iwm = b.IWM && b.IWM.change_52w_pct != null ? 'IWM 52w: ' + (b.IWM.change_52w_pct >= 0 ? '+' : '') + b.IWM.change_52w_pct.toFixed(1) + '%' : '';
        return iwm ? ' · ' + iwm : '';
      })()}</div></div>
    <div class="card"><div class="label">Analyses / Closed</div><div class="val">${metrics.analyses_rows || 0}</div>
      <div class="sub">${seenCount} seen · ${metrics.closed_positions_rows || 0} closed</div></div>
    <div class="card"><div class="label">Sources</div><div class="val">${Object.keys(sum.by_source||{}).length}</div>
      <div class="sub">${sourcesStr}</div></div>`;
}

let vChart, sChart, rChart;
function renderVerdictChart(bv) {
  const labels = Object.keys(bv||{}), data = Object.values(bv||{});
  const colors = labels.map(l => l==='BUY'?'#34d399':l==='STRONG BUY'?'#2dd4bf':l==='WATCH'?'#fbbf24':'#f87171');
  if (vChart) vChart.destroy();
  vChart = new Chart(document.getElementById('verdictChart'), {
    type:'doughnut', data:{labels, datasets:[{data, backgroundColor:colors, borderWidth:0}]},
    options:{plugins:{legend:{labels:{color:'#94a3b8',font:{size:12}}}},cutout:'65%'}
  });
}

function renderSizingChart(trades) {
  const filtered = trades.filter(t => t.position_size > 0);
  const labels = filtered.map(t => t.ticker);
  const data = filtered.map(t => t.position_size);
  const colors = filtered.map(t => {
    const v = (t.verdict||'').toUpperCase();
    return v.includes('STRONG')?'#2dd4bf':v.includes('BUY')?'#34d399':v.includes('WATCH')?'#fbbf24':'#f87171';
  });
  if (sChart) sChart.destroy();
  sChart = new Chart(document.getElementById('sizingChart'), {
    type:'bar', data:{labels, datasets:[{label:'Kelly %', data, backgroundColor:colors, borderRadius:6}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{ticks:{color:'#94a3b8',font:{size:10}},grid:{display:false}},
      y:{ticks:{color:'#94a3b8',callback:v=>v+'%'},grid:{color:'rgba(30,41,59,.4)'}}
    }}
  });
}

function renderAttribution(attr) {
  const byReg = (attr && attr.by_region) ? attr.by_region : {};
  const regions = Object.keys(byReg);
  const avgs = regions.map(r => byReg[r].avg_return_pct);
  const meds = regions.map(r => byReg[r].median_return_pct);
  if (rChart) rChart.destroy();
  rChart = new Chart(document.getElementById('regionChart'), {
    type:'bar',
    data:{
      labels: regions,
      datasets:[
        {label:'Avg Return %', data:avgs, backgroundColor:'#22d3ee', borderRadius:5},
        {label:'Median Return %', data:meds, backgroundColor:'#a78bfa', borderRadius:5}
      ]
    },
    options:{
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
      scales:{
        x:{ticks:{color:'#94a3b8',font:{size:11}},grid:{display:false}},
        y:{ticks:{color:'#94a3b8',callback:v=>v+'%'},grid:{color:'rgba(30,41,59,.4)'}}
      }
    }
  });

  document.getElementById('regionBody').innerHTML = regions.map(r => {
    const g = byReg[r];
    const ac = g.avg_return_pct >= 0 ? 'gain-pos' : 'gain-neg';
    const mc = g.median_return_pct >= 0 ? 'gain-pos' : 'gain-neg';
    const as = g.avg_return_pct >= 0 ? '+' : '';
    const ms = g.median_return_pct >= 0 ? '+' : '';
    return `<tr>
      <td><strong>${escapeHtml(r)}</strong></td>
      <td>${g.count}</td><td>${g.priced_count}</td>
      <td>${g.win_rate_pct.toFixed(1)}%</td>
      <td class="${ac}">${as}${g.avg_return_pct.toFixed(1)}%</td>
      <td class="${mc}">${ms}${g.median_return_pct.toFixed(1)}%</td>
    </tr>`;
  }).join('');

  const renderMover = (m, isWin) => {
    const yf = encodeURIComponent(m.ticker);
    const cls = isWin ? 'gain-pos' : 'gain-neg';
    const sign = m.gain_pct >= 0 ? '+' : '';
    return `<div class="mover-row">
      <div>
        <a class="ticker" href="https://finance.yahoo.com/quote/${yf}" target="_blank" rel="noopener">${escapeHtml(m.ticker)}</a>
        <span class="reg-pill">${escapeHtml(m.region)}</span>
        <div style="color:var(--td);font-size:10px">${fmtPrice(m.entry, m.ticker)} \u2192 ${fmtPrice(m.current, m.ticker)}</div>
      </div>
      <span class="${cls}">${sign}${m.gain_pct.toFixed(1)}%</span>
    </div>`;
  };

  document.getElementById('topWinners').innerHTML =
    (attr.top_winners || []).map(m => renderMover(m, true)).join('') || '<div style="color:var(--td);font-size:12px">No priced winners yet</div>';
  document.getElementById('topLosers').innerHTML =
    (attr.top_losers || []).map(m => renderMover(m, false)).join('') || '<div style="color:var(--td);font-size:12px">No priced losers yet</div>';
}

function renderTable() {
  const evalMap = {};
  cachedEvals.forEach(e => { evalMap[e.ticker] = e; });

  const search = (document.getElementById('filterSearch').value || '').trim().toUpperCase();
  const fReg = document.getElementById('filterRegion').value;
  const fVer = document.getElementById('filterVerdict').value;
  const fOut = document.getElementById('filterOutcome').value;

  const merged = cachedTrades.map(t => {
    const ev = evalMap[t.ticker] || {};
    return {
      ...t,
      region: regionOf(t.ticker),
      current: ev.current !== undefined ? ev.current : null,
      gain_pct: ev.gain_pct !== undefined ? ev.gain_pct : null
    };
  }).filter(t => {
    if (search && !t.ticker.toUpperCase().includes(search)) return false;
    if (fReg !== 'ALL' && t.region !== fReg) return false;
    if (fVer !== 'ALL' && (t.verdict || '').toUpperCase() !== fVer) return false;
    if (fOut === 'WIN' && (t.gain_pct === null || t.gain_pct <= 0)) return false;
    if (fOut === 'LOSS' && (t.gain_pct === null || t.gain_pct > 0)) return false;
    if (fOut === 'UNPRICED' && t.gain_pct !== null) return false;
    return true;
  });

  const sorted = [...merged].sort((a,b) => {
    let av = a[sortCol], bv = b[sortCol];
    const aNull = (av === null || av === undefined);
    const bNull = (bv === null || bv === undefined);
    if (aNull && bNull) return 0;
    if (aNull) return 1;
    if (bNull) return -1;
    if (typeof av === 'string') { av = av.toLowerCase(); bv = String(bv).toLowerCase(); }
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });

  document.getElementById('tradeCount').textContent = `(${sorted.length} of ${cachedTrades.length} positions)`;

  document.querySelectorAll('th[data-col]').forEach(th => {
    th.classList.remove('sorted','asc');
    if (th.dataset.col === sortCol) {
      th.classList.add('sorted');
      if (sortAsc) th.classList.add('asc');
    }
  });

  document.getElementById('tbody').innerHTML = sorted.map(t => {
    const vc = verdictClass(t.verdict);
    const bc = t.broker_status === 'filled' ? 'broker-filled' :
               t.broker_status === 'none' ? 'broker-none' : 'broker-pending';
    const currentStr = fmtPrice(t.current, t.ticker);
    const gainStr = t.gain_pct !== null
      ? `<span class="${t.gain_pct >= 0 ? 'gain-pos' : 'gain-neg'}">${t.gain_pct >= 0 ? '+' : ''}${t.gain_pct.toFixed(1)}%</span>`
      : '\u2014';
    const safeTicker = escapeHtml(t.ticker);
    const yf = encodeURIComponent(t.ticker);
    return `<tr>
      <td><a class="ticker" href="https://finance.yahoo.com/quote/${yf}" target="_blank" rel="noopener">${safeTicker}</a></td>
      <td><span class="reg-pill">${escapeHtml(t.region)}</span></td>
      <td>${escapeHtml(t.date)}</td>
      <td>${fmtPrice(t.entry_price, t.ticker)}</td>
      <td>${currentStr}</td>
      <td>${gainStr}</td>
      <td class="${vc}">${escapeHtml(t.verdict)}</td>
      <td>${t.position_size > 0 ? t.position_size.toFixed(1)+'%' : '\u2014'}</td>
      <td class="${bc}">${escapeHtml(t.broker_status || 'none')}</td>
      <td>${escapeHtml(t.source)}</td></tr>`;
  }).join('') || '<tr><td colspan="10" style="color:var(--td)">No matching positions.</td></tr>';
}

function verdictClass(v) {
  const u = (v||'').toUpperCase();
  if (u.includes('STRONG')) return 'verdict-strong';
  if (u.includes('BUY')) return 'verdict-buy';
  if (u.includes('WATCH')) return 'verdict-watch';
  return 'verdict-avoid';
}

/* ---------- Watchlist & Reviews (review gate) ---------- */
let API_KEY = localStorage.getItem('pg_key') || '';
if (API_KEY) document.getElementById('apiKey').value = API_KEY;

async function loadReviews() {
  try {
    const resp = await fetch('/portfolio/reviews');
    const reviews = await resp.json();
    const rows = Array.isArray(reviews) ? reviews : [];
    document.getElementById('reviewCount').textContent = `(${rows.length} pending)`;
    document.getElementById('reviewBody').innerHTML = rows.map(r => {
      const safeTicker = escapeHtml(r.ticker);
      const yf = encodeURIComponent(r.ticker);
      const stage = escapeHtml(r.stage || r.watchlist_stage || '\u2014');
      const verdict = escapeHtml(r.verdict || '\u2014');
      const action = escapeHtml(r.pending_action || r.proposed_action || '\u2014');
      const priceStr = r.price ? fmtPrice(r.price, r.ticker) : '\u2014';
      const rangeStr = (r.valuation_range_low && r.valuation_range_high)
        ? fmtPrice(r.valuation_range_low, r.ticker)+' \u2013 '+fmtPrice(r.valuation_range_high, r.ticker)
        : '\u2014';
      const mosShort = r.margin_of_safety ? escapeHtml(String(r.margin_of_safety).slice(0, 90)) : '\u2014';
      const srcs = Array.isArray(r.sources) ? r.sources.length : (r.sources ? 1 : 0);
      const next = escapeHtml(r.next_review_at || '\u2014');
      return `<tr>
        <td><a class="ticker" href="https://finance.yahoo.com/quote/${yf}" target="_blank" rel="noopener">${safeTicker}</a>${r.age_days != null ? ' <span style="color:var(--td);font-size:10px">'+r.age_days+'d</span>' : ''}</td>
        <td>${stage}</td>
        <td class="${verdictClass(r.verdict)}">${verdict}</td>
        <td>${action}</td>
        <td>${priceStr}</td>
        <td>${rangeStr}</td>
        <td>${mosShort}</td>
        <td>${srcs}</td>
        <td>${next}</td>
        <td style="white-space:nowrap">
          <button onclick="decide('${safeTicker}','accept')" style="background:rgba(52,211,153,.12);border:1px solid #34d399;color:#34d399;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;margin-right:6px">${r.pending_action === 'CLOSE' ? 'Close' : 'Accept'}</button>
          <button onclick="decide('${safeTicker}','dismiss')" style="background:rgba(248,113,113,.12);border:1px solid #f87171;color:#f87171;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer">Dismiss</button>
        </td></tr>`;
    }).join('') || '<tr><td colspan="10" style="color:var(--td)">No pending reviews \u2014 the review gate is clear.</td></tr>';
  } catch(e) {
    document.getElementById('reviewBody').innerHTML =
      '<tr><td colspan="10" style="color:var(--rd)">Error loading reviews: ' + escapeHtml(e.message) + '</td></tr>';
  }
}

async function decide(ticker, decision) {
  if (!API_KEY) {
    API_KEY = prompt('Enter the VPS API key to ' + decision + ' ' + ticker + ':') || '';
    if (!API_KEY) return;
    localStorage.setItem('pg_key', API_KEY);
    document.getElementById('apiKey').value = API_KEY;
  }
  try {
    const url = '/watchlist/' + encodeURIComponent(ticker) + '/' + (decision === 'accept' ? 'accept' : 'dismiss');
    const init = {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Key': API_KEY}
    };
    if (decision === 'accept') init.body = JSON.stringify({});
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (resp.ok) {
      document.getElementById('keyHint').textContent = '';
      load(); loadReviews();
    } else {
      document.getElementById('keyHint').textContent = 'Error: ' + (data.detail || resp.status);
    }
  } catch(e) {
    document.getElementById('keyHint').textContent = 'Error: ' + escapeHtml(e.message);
  }
}

document.getElementById('saveKey').addEventListener('click', () => {
  API_KEY = document.getElementById('apiKey').value.trim();
  localStorage.setItem('pg_key', API_KEY);
  document.getElementById('keyHint').textContent = API_KEY ? 'Key saved.' : 'Key cleared.';
});

document.getElementById('refreshBtn').addEventListener('click', () => { load(); loadReviews(); });
['filterSearch', 'filterRegion', 'filterVerdict', 'filterOutcome'].forEach(id => {
  document.getElementById(id).addEventListener('input', renderTable);
  document.getElementById(id).addEventListener('change', renderTable);
});

document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = true; }
    renderTable();
  });
});

load();
loadReviews();
setInterval(load, 5 * 60 * 1000);
setInterval(loadReviews, 5 * 60 * 1000);
</script>
</body>
</html>"""
