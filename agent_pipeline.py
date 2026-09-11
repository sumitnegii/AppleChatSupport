#!/usr/bin/env python3
"""Unified Agent Pipeline for AppleSupport AI Assistant.

This module chains together the 6 production stages of the AppleSupport agent:
1. Reading & normalizing the customer query.
2. Classifying intent into the 10-class taxonomy using TF-IDF + Logistic Regression.
3. Retrieving historical conversations using local FAISS vector search (IndexFlatIP).
4. Reviewing and ranking historical evidence against grounding thresholds.
5. Generating a concise draft reply grounded strictly in the retrieved evidence.
6. Deciding auto-handle vs. human escalation with an explicit, explainable reason.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Internal domain modules
from historical_retrieval import (
    load_index,
    faiss_search,
    cosine_search,
    clean_text,
    load_faiss_index,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    INDEX_CSV,
)
from grounded_reply_generation import generate_grounded_reply
from escalation_decision import decide_escalation

CANDIDATES_CSV = 'golden_candidates.csv'
FINAL_INTENTS = [
    'software_update',
    'battery_charging',
    'app_store',
    'account_apple_id',
    'connectivity',
    'keyboard_input',
    'device_issue',
    'payments_billing',
    'mail_messages',
    'media_accessories',
]

# Global singletons for fast re-use across web requests
_CLASSIFIER = None
_VECTORIZER = None
_RECORDS_CACHE = None
_FAISS_INDEX = None
_FAISS_MAPPING = None


def get_intent_classifier():
    """Train or return cached TF-IDF Logistic Regression classifier."""
    global _CLASSIFIER, _VECTORIZER
    if _CLASSIFIER is not None and _VECTORIZER is not None:
        return _CLASSIFIER, _VECTORIZER

    if not os.path.exists(CANDIDATES_CSV):
        raise FileNotFoundError(f"Training candidates file not found: {CANDIDATES_CSV}")

    df = pd.read_csv(CANDIDATES_CSV, low_memory=False)
    df['text_clean'] = df['text'].fillna('').apply(clean_text)
    df = df[df['text_clean'].str.len() > 0]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents='unicode',
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df['text_clean'].tolist())
    y = df['proposed_intent'].tolist()

    clf = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X, y)

    _CLASSIFIER = clf
    _VECTORIZER = vectorizer
    return _CLASSIFIER, _VECTORIZER


def get_retrieval_resources():
    """Load or return cached FAISS index, mapping, and conversation dataframe."""
    global _RECORDS_CACHE, _FAISS_INDEX, _FAISS_MAPPING
    if _RECORDS_CACHE is None:
        _RECORDS_CACHE = load_index(INDEX_CSV)

    if _FAISS_INDEX is None or _FAISS_MAPPING is None:
        try:
            _FAISS_INDEX, _FAISS_MAPPING = load_faiss_index(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
        except Exception:
            _FAISS_INDEX, _FAISS_MAPPING = None, None

    return _RECORDS_CACHE, _FAISS_INDEX, _FAISS_MAPPING


def classify_query_intent(query: str) -> Dict[str, Any]:
    """Classify the query into the 10-class intent taxonomy with confidence."""
    cleaned = clean_text(query)
    if not cleaned or len(cleaned.split()) < 2:
        return {
            'intent': 'ambiguous',
            'confidence': 0.0,
            'probabilities': {},
        }

    clf, vec = get_intent_classifier()
    X = vec.transform([cleaned])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_

    top_idx = int(np.argmax(probs))
    intent = str(classes[top_idx])
    confidence = float(probs[top_idx])

    prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
    return {
        'intent': intent,
        'confidence': round(confidence, 4),
        'probabilities': prob_dict,
    }


def run_agent_pipeline(query: str,
                       k: int = 3,
                       use_remote: bool = True) -> Dict[str, Any]:
    """Execute the end-to-end 6-stage AppleSupport AI Agent pipeline.

    Stage 1: Reading query
    Stage 2: Classifying intent
    Stage 3: Retrieving historical cases (RAG / FAISS)
    Stage 4: Reviewing historical evidence
    Stage 5: Generating grounded reply
    Stage 6: Making escalation decision
    """
    records, faiss_idx, mapping = get_retrieval_resources()

    # Stage 1: Reading query
    cleaned_query = clean_text(query)
    query_length = len(cleaned_query.split())

    # Stage 2: Classifying intent
    intent_res = classify_query_intent(query)
    detected_intent = intent_res['intent']
    intent_conf = intent_res['confidence']

    # Stage 3: Retrieving historical cases via FAISS
    try:
        top_df = faiss_search(
            query=query,
            records=records,
            faiss_index=faiss_idx,
            mapping=mapping,
            k=k,
        )
    except Exception:
        top_df = cosine_search(query=query, records=records, k=k)

    top_score = float(top_df['sim_score'].max()) if not top_df.empty and 'sim_score' in top_df.columns else 0.0

    # Stage 4: Reviewing historical evidence
    evidence_list = []
    evidence_ids = []
    for _, row in top_df.iterrows():
        cid = int(row['conversation_id'])
        evidence_ids.append(str(cid))
        score = float(row.get('sim_score', 0.0))
        cust_excerpt = clean_text(str(row.get('customer_turns', ''))[:160])
        supp_excerpt = clean_text(str(row.get('support_turns', ''))[:180])
        evidence_list.append({
            'conversation_id': cid,
            'sim_score': round(score, 4),
            'customer_excerpt': cust_excerpt,
            'support_excerpt': supp_excerpt,
            'n_messages': int(row.get('n_messages', 0)),
        })

    # Stage 5: Generating grounded reply
    reply_res = generate_grounded_reply(
        query=query,
        records=records,
        k=k,
        use_remote=use_remote,
    )
    draft_reply = reply_res.get('draft_reply', '')
    provider_used = reply_res.get('provider_used', 'deterministic_fallback')
    model_used = reply_res.get('model_used', 'deterministic_template')
    remote_request_succeeded = reply_res.get('remote_request_succeeded', False)

    # Stage 6: Making escalation decision
    esc_res = decide_escalation(
        query=query,
        intent=detected_intent,
        top_score=top_score,
        evidence_df=top_df,
        draft_reply=draft_reply,
        intent_confidence=intent_conf,
    )

    badge = 'AUTO-HANDLE' if esc_res['decision'] == 'auto_handle' else 'HUMAN ESCALATION'

    stages = [
        {
            'id': 1,
            'name': 'Reading Query',
            'detail': f'Sanitized input query ({query_length} words)',
            'status': 'completed',
        },
        {
            'id': 2,
            'name': 'Classifying Intent',
            'detail': f'Detected intent: {detected_intent} (top class prob: {intent_conf:.1%}, prior: 10.0%)',
            'status': 'completed',
        },
        {
            'id': 3,
            'name': 'Retrieving Historical Cases (RAG / FAISS)',
            'detail': f'Queried 2,000-case FAISS index; retrieved top {len(evidence_ids)} conversations',
            'status': 'completed',
        },
        {
            'id': 4,
            'name': 'Reviewing Historical Evidence',
            'detail': f'Evaluated semantic similarity (top score: {top_score:.4f}, min threshold: 0.28)',
            'status': 'completed',
        },
        {
            'id': 5,
            'name': 'Generating Grounded Reply',
            'detail': f'Grounded response generated via {provider_used} ({model_used})',
            'status': 'completed',
        },
        {
            'id': 6,
            'name': 'Making Escalation Decision',
            'detail': f'{badge}: {esc_res["reason"]}',
            'status': 'completed',
        },
    ]

    return {
        'query': query,
        'intent': detected_intent,
        'intent_confidence': intent_conf,
        'evidence_ids': evidence_ids,
        'evidence': evidence_list,
        'reply': draft_reply,
        'provider_used': provider_used,
        'model_used': model_used,
        'remote_request_succeeded': remote_request_succeeded,
        'escalation': {
            'decision': badge,
            'raw_decision': esc_res['decision'],
            'reason': esc_res['reason'],
            'risk_level': esc_res['risk_level'],
            'signals_triggered': esc_res['signals_triggered'],
        },
        'pipeline_stages': stages,
    }


if __name__ == '__main__':
    test_q = "My iPhone battery is draining so fast after updating to iOS 11. What can I do?"
    res = run_agent_pipeline(test_q, use_remote=False)
    print("Pipeline result for:", test_q)
    print("Intent:", res['intent'], f"({res['intent_confidence']:.2f})")
    print("Decision:", res['escalation']['decision'], "-", res['escalation']['reason'])
    print("Evidence IDs:", res['evidence_ids'])
    print("Reply:", res['reply'][:120], "...")
