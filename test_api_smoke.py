#!/usr/bin/env python3
"""Smoke tests for AppleSupport AI Agent FastAPI server and pipeline."""

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/health returns 200 and system metadata."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "AppleSupport" in data["agent"]
    assert "FAISS" in data["vector_index"]


def test_serve_frontend_index():
    """Verify GET / returns the HTML single-page frontend."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "AppleSupport AI Agent" in resp.text
    assert "Agent Pipeline Execution" in resp.text


def test_support_endpoint_autohandle():
    """Verify POST /api/support handles routine troubleshooting queries via AUTO-HANDLE."""
    payload = {
        "query": "My iPhone battery is draining so fast after updating to iOS 11. What can I do?",
        "k": 3,
        "use_remote": False,
    }
    resp = client.post("/api/support", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Core response schema assertions
    assert data["query"] == payload["query"]
    assert data["intent"] in ["battery_charging", "software_update"]
    assert isinstance(data["intent_confidence"], float)
    assert len(data["evidence_ids"]) == 3
    assert len(data["evidence"]) == 3
    assert isinstance(data["reply"], str) and len(data["reply"]) > 0

    # Escalation decision
    assert data["escalation"]["decision"] == "AUTO-HANDLE"
    assert data["escalation"]["risk_level"] == "low"
    assert "safe for automated response" in data["escalation"]["reason"].lower()

    # Pipeline stages
    assert len(data["pipeline_stages"]) == 6
    for stage in data["pipeline_stages"]:
        assert stage["status"] == "completed"
        assert "detail" in stage


def test_support_endpoint_escalation():
    """Verify POST /api/support escalates high-risk queries with explicit signals."""
    payload = {
        "query": "I need to talk to a human representative right now, my credit card was charged twice!",
        "k": 3,
        "use_remote": False,
    }
    resp = client.post("/api/support", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["escalation"]["decision"] == "HUMAN ESCALATION"
    assert data["escalation"]["risk_level"] == "high"
    signals = data["escalation"]["signals_triggered"]
    assert any(s in signals for s in ["explicit_human_request", "payment_refund_dispute"])
    assert len(data["pipeline_stages"]) == 6


def test_empty_query_rejected():
    """Verify empty query inputs are rejected with 400 Bad Request."""
    resp = client.post("/api/support", json={"query": "   ", "k": 3})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
