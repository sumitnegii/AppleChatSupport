#!/usr/bin/env python3
"""Smoke unit tests for the escalation decision layer."""

import pytest
from escalation_decision import decide_escalation, extract_escalation_signals


def test_explicit_human_request_escalates():
    query = "I want to speak with a human agent, not a bot."
    res = decide_escalation(query, intent="connectivity", top_score=0.45)
    assert res["decision"] == "human_escalation"
    assert res["risk_level"] == "high"
    assert "explicit_human_request" in res["signals_triggered"]
    assert "human" in res["reason"].lower()


def test_account_security_escalates():
    query = "My Apple ID has been disabled and I am locked out of iCloud."
    res = decide_escalation(query, intent="account_apple_id", top_score=0.50)
    assert res["decision"] == "human_escalation"
    assert res["risk_level"] == "high"
    assert "account_security_access" in res["signals_triggered"]


def test_payment_billing_dispute_escalates():
    query = "Why did you charge my card twice for this subscription?"
    res = decide_escalation(query, intent="payments_billing", top_score=0.42)
    assert res["decision"] == "human_escalation"
    assert res["risk_level"] == "high"
    assert "payment_refund_dispute" in res["signals_triggered"]


def test_severe_hardware_safety_escalates():
    query = "My phone is burning up and the battery looks swollen!"
    res = decide_escalation(query, intent="battery_charging", top_score=0.48)
    assert res["decision"] == "human_escalation"
    assert res["risk_level"] == "high"
    assert "severe_hardware_safety_data_loss" in res["signals_triggered"]


def test_ambiguous_intent_escalates():
    query = "help pls"
    res = decide_escalation(query, intent="ambiguous", top_score=0.20)
    assert res["decision"] == "human_escalation"
    assert "ambiguous_intent" in res["signals_triggered"] or "insufficient_retrieval_grounding" in res["signals_triggered"]


def test_low_retrieval_score_escalates():
    query = "How do I install custom android rom onto my apple tv?"
    res = decide_escalation(query, intent="software_update", top_score=0.18)
    assert res["decision"] == "human_escalation"
    assert "insufficient_retrieval_grounding" in res["signals_triggered"]
    assert "0.180" in res["reason"] or "similarity" in res["reason"].lower()


def test_unsupported_commitment_draft_escalates():
    query = "My screen is cracked after dropping it."
    draft = "We are sorry, we will replace your phone free of charge under warranty."
    res = decide_escalation(query, intent="device_issue", top_score=0.35, draft_reply=draft)
    assert res["decision"] == "human_escalation"
    assert "unsupported_commitments" in res["signals_triggered"]


def test_clean_troubleshooting_auto_handles():
    query = "My battery drains quickly after updating to iOS 11.1.1."
    res = decide_escalation(query, intent="battery_charging", top_score=0.45)
    assert res["decision"] == "auto_handle"
    assert res["risk_level"] == "low"
    assert len(res["signals_triggered"]) == 0
    assert "auto_handle" in res["decision"]
    assert "routine troubleshooting" in res["reason"].lower()


def test_low_intent_confidence_escalates():
    """Verify that near-random intent confidence (< 0.13) escalates to human."""
    query = "banana spaceship tomorrow morning"
    res = decide_escalation(
        query=query,
        intent="media_accessories",
        top_score=0.45,
        intent_confidence=0.1097,
    )
    assert res["decision"] == "human_escalation"
    assert "low_intent_confidence" in res["signals_triggered"]
    assert "random chance" in res["reason"].lower()
