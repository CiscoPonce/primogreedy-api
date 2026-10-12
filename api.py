"""PrimoGreedy Data API — FastAPI wrapper around DuckDB.

Serves seen tickers, paper portfolio, and agent run logs over HTTP.
Secured with X-API-Key header. Runs behind Tailscale (private network).

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Optional

import duckdb
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DUCKDB_PATH", "/home/ubuntu/primogreedy/data.duckdb")
API_KEY = os.getenv("VPS_API_KEY", "")
if not API_KEY:
    logger.warning("VPS_API_KEY environment variable is not set!")

MEMORY_TTL_LONG = 30 * 24 * 60 * 60   # 30 days — BUY / STRONG BUY
MEMORY_TTL_SHORT = 14 * 24 * 60 * 60  # 14 days — AVOID / WATCH / unknown


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> duckdb.DuckDBPyConnection:
    """Return a fresh connection (DuckDB is single-writer, but reads are fine)."""
    return duckdb.connect(DB_PATH)


@contextmanager
def get_db_ctx():
    """Context manager for DuckDB connections to ensure proper closure."""
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
        _cols = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='paper_portfolio'"
        ).fetchall()
        if "analysis_id" not in [c[0] for c in _cols]:
            con.execute("ALTER TABLE paper_portfolio ADD COLUMN analysis_id INTEGER")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="PrimoGreedy Data API",
    version="1.0.0",
    lifespan=lifespan,
)


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
def health():
    return {"status": "ok", "db": DB_PATH}


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
def mark_seen(body: SeenTickerIn, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    with get_db_ctx() as con:
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
def record_trade(body: TradeIn, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    with get_db_ctx() as con:
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
def update_trade_fill(ticker: str, body: TradeFillUpdate, x_api_key: str = Header(...)):
    """Update a trade's broker fill information after order execution."""
    verify_key(x_api_key)
    with get_db_ctx() as con:
        con.execute(
            """UPDATE paper_portfolio
               SET order_id = ?, fill_price = ?, broker_status = ?
               WHERE ticker = ? AND (order_id = ? OR date = current_date)""",
            [body.order_id, body.fill_price, body.broker_status,
             ticker, body.order_id],
        )
    return {"status": "ok", "ticker": ticker}


@app.delete("/portfolio/{ticker}")
def delete_portfolio_entry(ticker: str, x_api_key: str = Header(...)):
    """Delete all portfolio entries for a given ticker."""
    verify_key(x_api_key)
    with get_db_ctx() as con:
        before = con.execute("SELECT COUNT(*) FROM paper_portfolio WHERE ticker = ?", [ticker]).fetchone()[0]
        con.execute("DELETE FROM paper_portfolio WHERE ticker = ?", [ticker])
    return {"status": "ok", "ticker": ticker, "deleted": before}


@app.post("/portfolio/deduplicate")
def deduplicate_portfolio(x_api_key: str = Header(...)):
    """Keep only the most recent entry per ticker, removing older duplicates."""
    verify_key(x_api_key)
    with get_db_ctx() as con:
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
def submit_analysis(body: AnalysisIn, x_api_key: str = Header(...)):
    """Store a structured analysis and stage the ticker on the watchlist."""
    verify_key(x_api_key)
    stage = body.stage or _derive_stage(body.verdict)
    next_review = _coerce_date(body.next_review_at)

    with get_db_ctx() as con:
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


@app.get("/portfolio/reviews")
def portfolio_reviews(x_api_key: Optional[str] = Header(None)):
    """Active watchlist items awaiting an accept/dismiss decision."""
    if x_api_key:
        verify_key(x_api_key)
    with get_db_ctx() as con:
        rows = con.execute(
            f"""SELECT {_ANALYSIS_SELECT_A}, w.stage AS wl_stage, w.status AS wl_status
                FROM watchlist w
                JOIN analyses a ON a.id = w.analysis_id
                WHERE w.status = 'active' AND w.ticker NOT LIKE 'TEST%'
                ORDER BY a.analyzed_at DESC"""
        ).fetchall()
    out = []
    for r in rows:
        d = _analysis_dict(r[:len(_ANALYSIS_COLS)])
        d["watchlist_stage"] = r[len(_ANALYSIS_COLS)]
        d["watchlist_status"] = r[len(_ANALYSIS_COLS) + 1]
        d["pending_action"] = _derive_action(d["watchlist_stage"], d.get("proposed_action"))
        if d["pending_action"] in ("MAINTAIN",):
            continue
        out.append(d)
    return out


@app.post("/watchlist/{ticker}/accept")
def accept_decision(ticker: str, body: DecisionIn, x_api_key: str = Header(...)):
    """Accept an analysis: record the paper trade, close the review. No auto orders."""
    verify_key(x_api_key)
    with get_db_ctx() as con:
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
def dismiss_decision(ticker: str, x_api_key: str = Header(...)):
    """Dismiss an analysis without recording any trade."""
    verify_key(x_api_key)
    with get_db_ctx() as con:
        con.execute("UPDATE watchlist SET status = 'dismissed' WHERE ticker = ?", [ticker])
    return {"status": "dismissed", "ticker": ticker}


@app.get("/portfolio/evaluate")
def evaluate_portfolio(x_api_key: Optional[str] = Header(None)):
    """Fetch live prices and compute P&L for all portfolio entries."""
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

    trades = []
    total_roi = 0.0
    winners = 0
    valid = 0

    for r in rows:
        ticker, entry, date, verdict, source = r[0], r[1], str(r[2]), r[3], r[4]
        try:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
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
        except Exception as exc:
            logger.warning(f"Error evaluating ticker {ticker}: {exc}")
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
                      position_size, order_id, fill_price, broker_status
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
.wrap{max-width:1260px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;flex-wrap:wrap;gap:12px}
header h1{font-size:28px;font-weight:800;letter-spacing:-1px}
header h1 span{background:linear-gradient(135deg,var(--cy),var(--pu));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.badge{font-size:12px;padding:4px 14px;border:1px solid var(--bd);border-radius:99px;color:var(--gn);
background:rgba(52,211,153,.06);backdrop-filter:blur(8px)}
.badge .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--gn);margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:22px;
position:relative;overflow:hidden;transition:border-color .3s,transform .3s;animation:fadeUp .4s ease-out both}
.card:hover{border-color:var(--cy);transform:translateY(-2px)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,var(--cy),var(--pu),var(--gn));opacity:.6}
.card:nth-child(1){animation-delay:.05s}.card:nth-child(2){animation-delay:.1s}
.card:nth-child(3){animation-delay:.15s}.card:nth-child(4){animation-delay:.2s}
.card:nth-child(5){animation-delay:.25s}
.card .label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--td);margin-bottom:6px;font-weight:600}
.card .val{font-size:34px;font-weight:800;color:var(--tw);letter-spacing:-1px}
.card .sub{font-size:12px;color:var(--td);margin-top:6px}
.card .val.green{background:linear-gradient(135deg,#34d399,#2dd4bf);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.card .val.red{color:var(--rd)}

/* Panels */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:32px}
@media(max-width:768px){.grid2{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}
.panel{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:24px;animation:fadeUp .5s ease-out both}
.panel h2{font-size:15px;margin-bottom:16px;color:var(--tw);font-weight:700;letter-spacing:-.3px}
canvas{max-height:260px}

/* Table */
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 12px;border-bottom:2px solid var(--bd);color:var(--td);font-weight:700;
text-transform:uppercase;letter-spacing:.8px;font-size:10px;cursor:pointer;user-select:none;transition:color .2s}
th:hover{color:var(--cy)}
th.sorted::after{content:' ▾';color:var(--cy)}
th.sorted.asc::after{content:' ▴'}
td{padding:10px 12px;border-bottom:1px solid rgba(30,41,59,.4)}
tr:hover td{background:rgba(34,211,238,.04)}
.ticker{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--cy);text-decoration:none;transition:color .2s}
.ticker:hover{color:var(--pu)}
.verdict-buy{color:var(--gn);font-weight:700}.verdict-watch{color:var(--yl);font-weight:700}
.verdict-avoid{color:var(--rd);font-weight:700}.verdict-strong{color:#2dd4bf;font-weight:800}
.gain-pos{color:var(--gn);font-weight:600}.gain-neg{color:var(--rd);font-weight:600}
.broker-filled{color:var(--gn);font-size:11px;font-weight:600}.broker-pending{color:var(--yl);font-size:11px}
.broker-none{color:var(--td);font-size:11px}

/* Activity */
.activity{list-style:none;max-height:320px;overflow-y:auto}
.activity li{padding:8px 0;border-bottom:1px solid rgba(30,41,59,.3);font-size:13px;display:flex;justify-content:space-between;align-items:center}
.activity .ts{color:var(--td);font-size:11px;font-family:'JetBrains Mono',monospace}
.refresh{font-size:11px;color:var(--td);text-align:center;margin-top:24px;padding:16px 0;border-top:1px solid var(--bd)}
.loading{text-align:center;padding:40px;color:var(--td)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Primo<span>Greedy</span> Dashboard</h1>
  <div class="badge"><span class="dot"></span>Live — Auto-refresh 5m</div>
</header>

<div class="cards" id="cards"><div class="loading">Loading...</div></div>

<div class="grid2">
  <div class="panel"><h2>Verdict Distribution</h2><canvas id="verdictChart"></canvas></div>
  <div class="panel"><h2>Position Sizing (Kelly %)</h2><canvas id="sizingChart"></canvas></div>
</div>

<div class="panel" style="margin-bottom:32px">
  <h2>Portfolio — <span id="tradeCount" style="color:var(--td);font-weight:400;font-size:13px"></span></h2>
  <div style="overflow-x:auto">
    <table id="portfolioTable">
      <thead><tr>
        <th data-col="ticker">Ticker</th><th data-col="date">Date</th>
        <th data-col="entry_price">Entry</th><th data-col="current">Now</th>
        <th data-col="gain_pct">P/L</th><th data-col="verdict">Verdict</th>
        <th data-col="position_size">Kelly %</th><th data-col="broker_status">Broker</th>
        <th data-col="source">Source</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div class="panel" style="margin-bottom:32px">
    <h2>Watchlist &amp; Reviews — review gate (<span id="reviewCount" style="color:var(--td);font-weight:400;font-size:13px"></span>)</h2>
    <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <input id="apiKey" type="password" placeholder="API key (only needed to accept/dismiss)"
             style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;color:var(--tx);
                    padding:8px 12px;font-size:12px;font-family:'JetBrains Mono',monospace;min-width:240px"/>
      <button id="saveKey" style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;color:var(--cy);
             padding:8px 14px;font-size:12px;cursor:pointer">Save key</button>
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

<footer style="text-align:center;padding:28px 0 8px;border-top:1px solid var(--bd);margin-top:32px">
  <div style="margin-bottom:12px">
    <a href="https://ciscoponce.github.io/primogreedy/" target="_blank" rel="noopener"
       style="color:var(--cy);text-decoration:none;font-size:13px;font-weight:600;margin:0 16px">📄 About the Project</a>
    <a href="https://github.com/CiscoPonce/primogreedy" target="_blank" rel="noopener"
       style="color:var(--pu);text-decoration:none;font-size:13px;font-weight:600;margin:0 16px">⭐ GitHub Repo</a>
  </div>
  <div class="refresh" id="refreshNote"></div>
</footer>
</div>

<script>
const H = {};
let sortCol = 'date', sortAsc = false;
let evalData = [];

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/[&<>"']/g, function(m) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m];
  });
}

async function load() {
  try {
    const [sumR, seenR, evalR] = await Promise.all([
      fetch('/portfolio/summary', {headers: H}),
      fetch('/seen-tickers', {headers: H}),
      fetch('/portfolio/evaluate', {headers: H})
    ]);
    const sum = await sumR.json();
    const seen = await seenR.json();
    const evalResult = await evalR.json();
    evalData = evalResult.trades || [];

    renderCards(sum, seen, evalResult);
    renderVerdictChart(sum.by_verdict);
    renderSizingChart(sum.recent_trades);
    renderTable(sum.recent_trades, evalData);

    document.getElementById('refreshNote').textContent =
      'Last refresh: ' + new Date().toLocaleTimeString() + ' — next in 5 min';
  } catch(e) {
    document.getElementById('cards').innerHTML =
      '<div class="loading">Error loading data: ' + escapeHtml(e.message) + '</div>';
  }
}

function renderCards(sum, seen, ev) {
  const buys = (sum.by_verdict['BUY']||0) + (sum.by_verdict['STRONG BUY']||0);
  const avoids = sum.by_verdict['AVOID']||0;
  const watches = sum.by_verdict['WATCH']||0;
  const seenCount = Object.keys(seen).length;
  const avgReturn = ev.avg_return !== undefined ? ev.avg_return : 0;
  const winRate = ev.win_rate !== undefined ? ev.win_rate : 0;
  const returnClass = avgReturn >= 0 ? 'green' : 'red';
  const returnSign = avgReturn >= 0 ? '+' : '';
  const sourcesStr = Object.entries(sum.by_source||{}).map(([k,v])=>escapeHtml(k)+': '+v).join(' · ');
  document.getElementById('cards').innerHTML = `
    <div class="card"><div class="label">Total Trades</div><div class="val">${sum.total_trades}</div>
      <div class="sub">${buys} buys · ${watches} watch · ${avoids} avoid</div></div>
    <div class="card"><div class="label">Buy Rate</div>
      <div class="val green">${sum.total_trades ? Math.round(buys/sum.total_trades*100) : 0}%</div>
      <div class="sub">${buys} actionable of ${sum.total_trades}</div></div>
    <div class="card"><div class="label">Avg Return</div>
      <div class="val ${returnClass}">${returnSign}${avgReturn.toFixed(1)}%</div>
      <div class="sub">Win rate: ${winRate.toFixed(0)}%</div></div>
    <div class="card"><div class="label">Seen Tickers</div><div class="val">${seenCount}</div>
      <div class="sub">Active in memory ledger</div></div>
    <div class="card"><div class="label">Sources</div><div class="val">${Object.keys(sum.by_source||{}).length}</div>
      <div class="sub">${sourcesStr}</div></div>`;
}

let vChart, sChart;
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

function renderTable(trades, evals) {
  // Merge eval data (current price + gain) into trades
  const evalMap = {};
  evals.forEach(e => { evalMap[e.ticker] = e; });

  const merged = trades.map(t => {
    const ev = evalMap[t.ticker] || {};
    return {...t, current: ev.current || null, gain_pct: ev.gain_pct !== undefined ? ev.gain_pct : null};
  });

  const sorted = [...merged].sort((a,b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = -9999;
    if (bv === null || bv === undefined) bv = -9999;
    if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv||'').toLowerCase(); }
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });

  document.getElementById('tradeCount').textContent = sorted.length + ' positions';

  // Update sort indicators
  document.querySelectorAll('th[data-col]').forEach(th => {
    th.classList.remove('sorted','asc');
    if (th.dataset.col === sortCol) {
      th.classList.add('sorted');
      if (sortAsc) th.classList.add('asc');
    }
  });

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = sorted.map(t => {
    const vc = verdictClass(t.verdict);
    const bc = t.broker_status === 'filled' ? 'broker-filled' :
               t.broker_status === 'none' ? 'broker-none' : 'broker-pending';  
    const currentStr = t.current !== null ? '$'+t.current.toFixed(2) : '—';
    const gainStr = t.gain_pct !== null
      ? `<span class="${t.gain_pct >= 0 ? 'gain-pos' : 'gain-neg'}">${t.gain_pct >= 0 ? '+' : ''}${t.gain_pct.toFixed(1)}%</span>`
      : '—';
    const safeTicker = escapeHtml(t.ticker);
    const safeDate = escapeHtml(t.date);
    const safeVerdict = escapeHtml(t.verdict);
    const safeBrokerStatus = escapeHtml(t.broker_status || 'none');
    const safeSource = escapeHtml(t.source);
    const yf = encodeURIComponent(t.ticker);
    return `<tr>
      <td><a class="ticker" href="https://finance.yahoo.com/quote/${yf}" target="_blank" rel="noopener">${safeTicker}</a></td>
      <td>${safeDate}</td>
      <td>$${t.entry_price.toFixed(2)}</td>
      <td>${currentStr}</td>
      <td>${gainStr}</td>
      <td class="${vc}">${safeVerdict}</td>
      <td>${t.position_size > 0 ? t.position_size.toFixed(1)+'%' : '—'}</td>
      <td class="${bc}">${safeBrokerStatus}</td>
      <td>${safeSource}</td></tr>`;
  }).join('');
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

async function loadReviews() {
  try {
    const resp = await fetch('/portfolio/reviews');
    const reviews = await resp.json();
    const rows = Array.isArray(reviews) ? reviews : [];
    document.getElementById('reviewCount').textContent = rows.length + ' pending';
    document.getElementById('reviewBody').innerHTML = rows.map(r => {
      const safeTicker = escapeHtml(r.ticker);
      const yf = encodeURIComponent(r.ticker);
      const stage = escapeHtml(r.stage || r.watchlist_stage || '—');
      const verdict = escapeHtml(r.verdict || '—');
      const action = escapeHtml(r.pending_action || r.proposed_action || '—');
      const priceStr = r.price ? '$'+r.price.toFixed(2) : '—';
      const rangeStr = (r.valuation_range_low && r.valuation_range_high)
        ? '$'+r.valuation_range_low.toFixed(2)+' – $'+r.valuation_range_high.toFixed(2)
        : '—';
      const mosShort = r.margin_of_safety ? escapeHtml(String(r.margin_of_safety).slice(0, 90)) : '—';
      const srcs = Array.isArray(r.sources) ? r.sources.length : (r.sources ? 1 : 0);
      const next = r.next_review_at || '—';
      return `<tr>
        <td><a class="ticker" href="https://finance.yahoo.com/quote/${yf}" target="_blank" rel="noopener">${safeTicker}</a></td>
        <td>${stage}</td>
        <td class="${verdictClass(r.verdict)}">${verdict}</td>
        <td>${action}</td>
        <td>${priceStr}</td>
        <td>${rangeStr}</td>
        <td>${mosShort}</td>
        <td>${srcs}</td>
        <td>${next}</td>
        <td style="white-space:nowrap">
          <button onclick="decide('${safeTicker}','accept')" style="background:rgba(52,211,153,.12);border:1px solid #34d399;color:#34d399;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;margin-right:6px">Accept</button>
          <button onclick="decide('${safeTicker}','dismiss')" style="background:rgba(248,113,113,.12);border:1px solid #f87171;color:#f87171;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer">Dismiss</button>
        </td></tr>`;
    }).join('') || '<tr><td colspan="10" style="color:var(--td)">No pending reviews — the review gate is clear.</td></tr>';
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

document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = true; }
    load();
  });
});

load();
loadReviews();
setInterval(load, 5 * 60 * 1000);
setInterval(loadReviews, 5 * 60 * 1000);
</script>
</body>
</html>"""
