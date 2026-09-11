#!/usr/bin/env python3
"""Smoke unit tests for the LLM judge evaluation module."""

import numpy as np
import pytest
from llm_judge_eval import (
    build_judge_prompt,
    parse_judge_response,
    compute_metrics,
    compute_confusion_matrix,
    DIMENSIONS,
)


def test_build_judge_prompt():
    query = "battery drain on iPhone 6s"
    evidence = "Conversation ID 123: AppleSupport suggested updating iOS."
    reply = "Please update to iOS 11.1 and test again."

    prompt = build_judge_prompt(query, evidence, reply)
    assert query in prompt
    assert evidence in prompt
    assert reply in prompt
    assert "0 = No unsupported claims" in prompt
    assert "1 = Minor or limited unsupported claim(s)" in prompt
    assert "2 = Significant or multiple unsupported claims" in prompt
    for dim in DIMENSIONS:
        assert dim in prompt


def test_parse_judge_response_clean_json():
    raw = '{"factual_grounding": 2, "helpfulness": 2, "relevance": 2, "unsupported_claims": 0, "overall_quality": 2, "rationale": "Perfect response."}'
    parsed = parse_judge_response(raw)

    assert parsed['factual_grounding'] == 2
    assert parsed['helpfulness'] == 2
    assert parsed['relevance'] == 2
    assert parsed['unsupported_claims'] == 0
    assert parsed['overall_quality'] == 2
    assert parsed['rationale'] == "Perfect response."


def test_parse_judge_response_markdown_json():
    raw = '''```json
{
  "factual_grounding": 1,
  "helpfulness": 0,
  "relevance": 2,
  "unsupported_claims": 1,
  "overall_quality": 0,
  "rationale": "Truncated reply."
}
```'''
    parsed = parse_judge_response(raw)
    assert parsed['factual_grounding'] == 1
    assert parsed['helpfulness'] == 0
    assert parsed['relevance'] == 2
    assert parsed['unsupported_claims'] == 1
    assert parsed['overall_quality'] == 0
    assert parsed['rationale'] == "Truncated reply."


def test_parse_judge_response_clamping():
    raw = '{"factual_grounding": 5, "helpfulness": -1, "relevance": 2, "unsupported_claims": 0, "overall_quality": 1}'
    parsed = parse_judge_response(raw)
    assert parsed['factual_grounding'] == 2
    assert parsed['helpfulness'] == 0


def test_compute_metrics_perfect_match():
    y_true = [0, 1, 2, 1, 2]
    y_pred = [0, 1, 2, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m['exact_agreement'] == 1.0
    assert m['adjacent_agreement'] == 1.0
    assert m['mae'] == 0.0
    assert m['bias'] == 0.0
    assert m['correlation'] == 1.0
    assert m['quadratic_weighted_kappa'] == 1.0


def test_compute_metrics_adjacent():
    y_true = [1, 2, 1, 0]
    y_pred = [2, 1, 0, 1]
    m = compute_metrics(y_true, y_pred)
    assert m['exact_agreement'] == 0.0
    assert m['adjacent_agreement'] == 1.0
    assert m['mae'] == 1.0


def test_compute_confusion_matrix():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == 4
    assert cm[0, 0] == 1
    assert cm[1, 2] == 1
    assert cm[2, 2] == 1
    assert cm[2, 1] == 1
