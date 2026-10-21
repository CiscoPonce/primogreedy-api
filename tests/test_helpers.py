"""Unit tests for the pure helpers in api.py.

These cover the logic that directly gates whether an analysis is stored or
silently lost, so regressions here are expensive and hard to notice.
"""

import pytest

import api


# ---------------------------------------------------------------------------
# _derive_stage — legacy verdict label -> 4-stage funnel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict,expected", [
    ("STRONG BUY", "BUY_CANDIDATE"),
    ("BUY", "BUY_CANDIDATE"),
    ("strong buy", "BUY_CANDIDATE"),
    ("WATCH", "WATCHLIST"),
    ("watch closely", "WATCHLIST"),
    ("AVOID", "DISCARD"),
    ("DISCARD", "DISCARD"),
    ("", "RESEARCH"),
    (None, "RESEARCH"),
    ("something unexpected", "RESEARCH"),
])
def test_derive_stage(verdict, expected):
    assert api._derive_stage(verdict) == expected


def test_derive_stage_strong_buy_is_not_downgraded():
    # "STRONG BUY" contains "BUY"; must still map to BUY_CANDIDATE.
    assert api._derive_stage("STRONG BUY") == "BUY_CANDIDATE"


# ---------------------------------------------------------------------------
# _stage_to_verdict — inverse mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage,expected", [
    ("BUY_CANDIDATE", "BUY"),
    ("WATCHLIST", "WATCH"),
    ("DISCARD", "AVOID"),
    ("RESEARCH", "WATCH"),
    ("", "WATCH"),
    (None, "WATCH"),
    ("garbage", "WATCH"),
])
def test_stage_to_verdict(stage, expected):
    assert api._stage_to_verdict(stage) == expected


# ---------------------------------------------------------------------------
# _derive_action
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage,proposed,expected", [
    ("BUY_CANDIDATE", None, "INITIATE"),
    ("WATCHLIST", None, "PENDING"),
    ("RESEARCH", None, "PENDING"),
    ("DISCARD", None, "DISCARD"),
    ("WATCHLIST", "PENDING", "PENDING"),   # PENDING is not a real override
    ("WATCHLIST", "", "PENDING"),
    ("DISCARD", "CLOSE", "CLOSE"),         # explicit proposal wins
    ("BUY_CANDIDATE", "INCREASE", "INCREASE"),
])
def test_derive_action(stage, proposed, expected):
    assert api._derive_action(stage, proposed) == expected


# ---------------------------------------------------------------------------
# _extract_verdict_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("STRONG BUY", "STRONG BUY"),
    ("BUY", "BUY"),
    ("WATCH", "WATCH"),
    ("AVOID", "AVOID"),
    ("BUY — high conviction but valuation stretched", "BUY"),
    ("", "OTHER"),
    (None, "OTHER"),
    ("no keyword here", "OTHER"),
])
def test_extract_verdict_label(text, expected):
    assert api._extract_verdict_label(text) == expected


def test_extract_verdict_label_prefers_strong_buy():
    # "STRONG BUY" also contains "BUY"; must not collapse to plain BUY.
    assert api._extract_verdict_label("it is a STRONG BUY") == "STRONG BUY"


# ---------------------------------------------------------------------------
# _coerce_date — LLM date coercion (regression: bad dates used to 500)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    ("2026-03-15", "2026-03-15"),
    ("Report due 2026-03-15 after earnings", "2026-03-15"),
    ("2026/03/15", "2026-03-15"),
    ("15/03/2026", "2026-03-15"),
    ("03/15/2026", "2026-03-15"),
    ("March 15, 2026", "2026-03-15"),
    ("Mar 15, 2026", "2026-03-15"),
    ("15 March 2026", "2026-03-15"),
    ("15 Mar 2026", "2026-03-15"),
    ("2026-13-45", None),      # invalid month/day
    ("after Q3 earnings", None),
    ("", None),
    (None, None),
    ("   ", None),
])
def test_coerce_date(val, expected):
    assert api._coerce_date(val) == expected


def test_coerce_date_embedded_iso_wins():
    # An ISO date embedded in prose should be extracted even with other numbers.
    assert api._coerce_date("re-check around 2026-04-01, after the 10-K") == "2026-04-01"


# ---------------------------------------------------------------------------
# Price cache TTL logic
# ---------------------------------------------------------------------------

def test_cache_roundtrip_within_ttl():
    api._price_cache.clear()
    api._cache_put("AAPL", {"price": 123.0, "currency": "USD"})
    got = api._cache_get("AAPL")
    assert got == {"price": 123.0, "currency": "USD"}


def test_cache_miss_for_unknown_ticker():
    api._price_cache.clear()
    assert api._cache_get("NOPE") is None


def test_cache_expiry_honours_ttl(monkeypatch):
    api._price_cache.clear()
    api._cache_put("AAPL", {"price": 1.0, "currency": "USD"})
    # Pretend the entry was written long ago.
    ts, payload = api._price_cache["AAPL"]
    api._price_cache["AAPL"] = (ts - api.PRICE_TTL - 1, payload)
    assert api._cache_get("AAPL") is None


def test_cache_pruning_bounds_size(monkeypatch):
    api._price_cache.clear()
    # Force a tiny TTL so the newly written entry is immediately stale.
    monkeypatch.setattr(api, "PRICE_TTL", 0)
    for i in range(2100):
        api._cache_put(f"T{i}", {"price": 1.0, "currency": "USD"})
    # Pruning should have dropped the stale entries.
    assert len(api._price_cache) < 2100
    api._price_cache.clear()
