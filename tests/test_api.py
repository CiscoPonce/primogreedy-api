"""Integration tests for the PrimoGreedy API endpoints.

Uses FastAPI's TestClient against a throwaway DuckDB (see conftest.py).
yfinance is never called for real here.
"""

import api

# A minimal payload that satisfies the AnalysisIn model.
def _analysis_payload(ticker="AAPL", **over):
    payload = {"ticker": ticker, "verdict": "BUY", "source": "pytest"}
    payload.update(over)
    return payload


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "db" in body


def test_metrics_shape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    for key in ("analyses_rows", "paper_portfolio_rows", "seen_tickers_rows",
                "last_analysis_at", "process_uptime_seconds"):
        assert key in body


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_write_requires_api_key(client):
    r = client.post("/analyses", json=_analysis_payload())
    assert r.status_code in (401, 422)  # missing header -> 422 by FastAPI


def test_write_rejects_bad_api_key(client):
    r = client.post("/analyses", json=_analysis_payload(),
                    headers={"x-api-key": "wrong"})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Analyses round-trip
# ---------------------------------------------------------------------------

def test_submit_and_read_analysis(client, auth_headers, clean_tables):
    r = client.post("/analyses",
                    json=_analysis_payload("TESTAA", verdict="WATCH"),
                    headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "WATCHLIST"

    r = client.get("/analyses/TESTAA", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "TESTAA"
    assert body["verdict"] == "WATCH"


def test_submit_analysis_with_prose_next_review_does_not_500(client, auth_headers, clean_tables):
    # Regression: unparseable next_review_at used to raise and lose the analysis.
    r = client.post(
        "/analyses",
        json=_analysis_payload("TESTBB", next_review_at="after Q3 earnings"),
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text

    r = client.get("/analyses/TESTBB", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["next_review_at"] is None


def test_submit_analysis_records_iso_next_review(client, auth_headers, clean_tables):
    r = client.post(
        "/analyses",
        json=_analysis_payload("TESTCC", next_review_at="2026-12-31"),
        headers=auth_headers,
    )
    assert r.status_code == 200
    r = client.get("/analyses/TESTCC", headers=auth_headers)
    assert r.json()["next_review_at"] == "2026-12-31"


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def test_record_portfolio_trade(client, auth_headers, clean_tables):
    r = client.post("/portfolio",
                    json={"ticker": "PGTEST1", "entry_price": 10.0,
                          "date": "2026-01-02", "verdict": "BUY"},
                    headers=auth_headers)
    assert r.status_code == 200, r.text

    r = client.get("/portfolio", headers=auth_headers)
    assert r.status_code == 200
    tickers = [t["ticker"] for t in r.json()]
    assert "PGTEST1" in tickers


def test_evaluate_portfolio_empty_ok(client, auth_headers, clean_tables):
    r = client.get("/portfolio/evaluate", headers=auth_headers)
    assert r.status_code == 200
    # Either the explicit empty marker or zero trades.
    body = r.json()
    assert body.get("trades", []) == [] or "report" in body


def test_seen_tickers_roundtrip(client, auth_headers, clean_tables):
    r = client.post("/seen-tickers", json={"ticker": "TESTEE"},
                    headers=auth_headers)
    assert r.status_code == 200
    r = client.get("/seen-tickers", headers=auth_headers)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

def test_dashboard_is_public(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "PrimoGreedy" in r.text


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_evaluate_endpoint_is_rate_limited(client, auth_headers, clean_tables):
    # The endpoint is capped at 30/minute; the 31st call must be 429.
    api.limiter.enabled = True
    api.limiter.reset()
    try:
        statuses = []
        for _ in range(35):
            statuses.append(client.get("/portfolio/evaluate",
                                       headers=auth_headers).status_code)
        assert 429 in statuses, f"expected a 429 in {statuses}"
    finally:
        api.limiter.enabled = False
        api.limiter.reset()


# ---------------------------------------------------------------------------
# Review gate: the queue must contain real decisions only
# ---------------------------------------------------------------------------

def test_reviews_surface_only_actionable_items(client, auth_headers, clean_tables):
    client.post("/analyses", json={"ticker": "REVBUY", "stage": "BUY_CANDIDATE",
                                   "verdict": "BUY", "proposed_action": "INITIATE",
                                   "price": 2.0, "source": "pytest"},
                headers=auth_headers)
    client.post("/analyses", json={"ticker": "REVWATCH", "stage": "WATCHLIST",
                                   "verdict": "WATCH", "proposed_action": "PENDING",
                                   "source": "pytest"}, headers=auth_headers)
    client.post("/analyses", json={"ticker": "REVAVOID", "stage": "DISCARD",
                                   "verdict": "AVOID", "proposed_action": "DISCARD",
                                   "source": "pytest"}, headers=auth_headers)
    client.post("/analyses", json={"ticker": "REVCLOSE", "stage": "DISCARD",
                                   "verdict": "AVOID", "proposed_action": "CLOSE",
                                   "price": 1.0, "source": "pytest"},
                headers=auth_headers)

    r = client.get("/portfolio/reviews", headers=auth_headers)
    assert r.status_code == 200
    tickers = {x["ticker"] for x in r.json()}
    assert {"REVBUY", "REVCLOSE"} <= tickers
    assert "REVWATCH" not in tickers
    assert "REVAVOID" not in tickers

    row = next(x for x in r.json() if x["ticker"] == "REVBUY")
    assert row["needs_review"] is True
    assert row["age_days"] is not None
    assert row["pending_action"] == "INITIATE"

    # The full funnel is still inspectable.
    r = client.get("/portfolio/reviews?include=all", headers=auth_headers)
    all_tickers = {x["ticker"] for x in r.json()}
    assert {"REVBUY", "REVCLOSE", "REVWATCH", "REVAVOID"} <= all_tickers

    # Non-actionable rows are explicitly marked as such.
    tag = next(x for x in r.json() if x["ticker"] == "REVWATCH")
    assert tag["needs_review"] is False


def test_accept_close_removes_position_instead_of_opening_one(client, auth_headers, clean_tables):
    client.post("/portfolio", json={"ticker": "REVPOS", "entry_price": 5.0,
                                    "date": "2026-01-02", "verdict": "BUY",
                                    "source": "pytest"}, headers=auth_headers)
    client.post("/analyses", json={"ticker": "REVPOS", "stage": "DISCARD",
                                   "verdict": "AVOID", "proposed_action": "CLOSE",
                                   "price": 4.0, "source": "pytest"},
                headers=auth_headers)

    r = client.post("/watchlist/REVPOS/accept", json={}, headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "closed"
    assert body["closed_positions"] == 1

    # The position must be gone — and no replacement row created.
    listed = client.get("/portfolio", headers=auth_headers).json()
    assert "REVPOS" not in {t["ticker"] for t in listed}


def test_metrics_reports_review_backlog(client, auth_headers, clean_tables):
    client.post("/analyses", json={"ticker": "METBUY", "stage": "BUY_CANDIDATE",
                                   "verdict": "BUY", "proposed_action": "INITIATE",
                                   "source": "pytest"}, headers=auth_headers)
    client.post("/analyses", json={"ticker": "METWATCH", "stage": "WATCHLIST",
                                   "verdict": "WATCH", "proposed_action": "PENDING",
                                   "source": "pytest"}, headers=auth_headers)
    m = client.get("/metrics").json()
    assert m["review_actionable"] >= 1
    assert m["review_non_actionable"] >= 1
