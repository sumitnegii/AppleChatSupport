#!/usr/bin/env python3
"""Escalation Decision Layer for AppleSupport AI Agent.

Decides between 'auto_handle' and 'human_escalation' with an explicit, explainable
reason based on observable query signals, intent classification, and retrieval evidence.

Core Design Principles:
1. Conservative Safety: Financial, account security, hardware safety, and data loss
   cases are immediately escalated to human agents.
2. Explainability: Every decision is accompanied by an explicit, human-readable reason
   and a list of triggered risk signals.
3. Grounding Verification: Queries with insufficient historical evidence or low similarity
   scores are escalated rather than risking ungrounded hallucinations.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd


# Default threshold for minimum retrieval relevance required to safely auto-handle
MIN_RETRIEVAL_SIMILARITY_THRESHOLD = 0.28

# Minimum classifier probability required to safely verify intent (uniform random prior is 0.10 for 10 classes)
MIN_INTENT_CONFIDENCE_THRESHOLD = 0.13

# Intent categories that inherently mandate human handling due to risk/PII
MANDATORY_HUMAN_INTENTS = {
    'payments_billing',
    'account_apple_id',
    'ambiguous',
}

# Regex patterns for high-risk customer signals
PATTERNS = {
    'explicit_human_request': re.compile(
        r'\b(speak to|talk to|connect me to|transfer to|need a|want a)?\s*'
        r'(human|real person|agent|representative|advisor|operator|manager|supervisor|someone)\b',
        re.IGNORECASE
    ),
    'account_security_access': re.compile(
        r'\b(locked out|lockout|stolen|hacked|compromised|two[\s-]factor|2fa|security question|'
        r'unauthorized access|disabled account|account disabled|activation lock|passcode forgot|'
        r'forgot password|reset password|apple id locked|find my iphone|lost mode)\b',
        re.IGNORECASE
    ),
    'payment_refund_dispute': re.compile(
        r'\b(unrecognized charge|charged twice|double charge|unauthorized charge|refund|dispute|'
        r'credit card|debit card|bank statement|fraud|fraudulent|overcharged|billing error|'
        r'gift card balance|subscription fee|charged for|cancel subscription|money stolen)\b',
        re.IGNORECASE
    ),
    'severe_hardware_safety_data_loss': re.compile(
        r'\b(burning|burning up|smoke|smoking|spark|sparking|fire|exploded|swollen battery|battery swollen|'
        r'shattered|cracked screen|water damage|dropped in water|lost all my data|photos deleted|'
        r'lost everything|data lost|bricked|wont turn on at all|phone completely dead)\b',
        re.IGNORECASE
    ),
    'unsupported_commitments': re.compile(
        r'\b(we will refund|we can refund|we will replace|we guarantee|send you a new|'
        r'warranty will cover|free repair|compensation)\b',
        re.IGNORECASE
    ),
}


def extract_escalation_signals(text: str,
                               intent: Optional[str] = None,
                               top_score: Optional[float] = None,
                               draft_reply: Optional[str] = None,
                               intent_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
    """Inspect observable text, intent, and retrieval signals for escalation risks."""
    signals = []
    text_clean = str(text).lower()

    # 1. Explicit Human Request
    if PATTERNS['explicit_human_request'].search(text_clean):
        signals.append({
            'signal': 'explicit_human_request',
            'severity': 'high',
            'description': 'Customer explicitly requested to interact with a human agent or representative.'
        })

    # 2. Account / Security / Access
    if (intent == 'account_apple_id') or PATTERNS['account_security_access'].search(text_clean):
        signals.append({
            'signal': 'account_security_access',
            'severity': 'high',
            'description': 'Account security, credentials, or Apple ID access issue requires authenticated human support.'
        })

    # 3. Payment / Billing / Financial Dispute
    if (intent == 'payments_billing') or PATTERNS['payment_refund_dispute'].search(text_clean):
        signals.append({
            'signal': 'payment_refund_dispute',
            'severity': 'high',
            'description': 'Financial, billing, or charge dispute involves monetary adjustments or payment details.'
        })

    # 4. Severe Hardware Safety or Data Loss
    if PATTERNS['severe_hardware_safety_data_loss'].search(text_clean):
        signals.append({
            'signal': 'severe_hardware_safety_data_loss',
            'severity': 'high',
            'description': 'Hardware safety hazard (overheating/battery swelling) or catastrophic data loss.'
        })

    # 5. Ambiguous Intent / Low Classifier Confidence
    if intent == 'ambiguous' or len(text.strip().split()) < 3:
        signals.append({
            'signal': 'ambiguous_intent',
            'severity': 'medium',
            'description': 'Customer query intent is ambiguous or underspecified.'
        })
    elif intent_confidence is not None and intent_confidence < MIN_INTENT_CONFIDENCE_THRESHOLD:
        signals.append({
            'signal': 'low_intent_confidence',
            'severity': 'medium',
            'description': f'Classifier confidence ({intent_confidence:.1%}) is near random chance ({MIN_INTENT_CONFIDENCE_THRESHOLD:.1%}); intent is unverified.'
        })

    # 6. Insufficient Retrieval Grounding
    if top_score is not None and top_score < MIN_RETRIEVAL_SIMILARITY_THRESHOLD:
        signals.append({
            'signal': 'insufficient_retrieval_grounding',
            'severity': 'medium',
            'description': f'Top retrieval similarity ({top_score:.3f}) is below confidence threshold ({MIN_RETRIEVAL_SIMILARITY_THRESHOLD}).'
        })

    # 7. Unsupported Commitments in Draft Reply
    if draft_reply and PATTERNS['unsupported_commitments'].search(draft_reply.lower()):
        signals.append({
            'signal': 'unsupported_commitments',
            'severity': 'high',
            'description': 'Draft reply contains financial or replacement commitments requiring supervisor sign-off.'
        })

    return signals


def decide_escalation(query: str,
                      intent: Optional[str] = None,
                      top_score: Optional[float] = None,
                      evidence_df: Optional[pd.DataFrame] = None,
                      draft_reply: Optional[str] = None,
                      intent_confidence: Optional[float] = None) -> Dict[str, Any]:
    """Decide whether to auto-handle or escalate to human with a stated reason.

    Returns:
        dict with:
            - decision: 'auto_handle' | 'human_escalation'
            - reason: explicit explanation string
            - risk_level: 'high' | 'medium' | 'low'
            - signals_triggered: list of signal names
    """
    # If evidence_df is provided and top_score was not passed, compute max score
    if top_score is None and evidence_df is not None and not evidence_df.empty:
        if 'sim_score' in evidence_df.columns:
            top_score = float(evidence_df['sim_score'].max())

    # Extract all matching risk signals
    signals = extract_escalation_signals(
        text=query,
        intent=intent,
        top_score=top_score,
        draft_reply=draft_reply,
        intent_confidence=intent_confidence,
    )

    signal_names = [s['signal'] for s in signals]
    high_severity = [s for s in signals if s['severity'] == 'high']
    medium_severity = [s for s in signals if s['severity'] == 'medium']

    # Escalation Rule 1: Any High Severity Signal -> Escalate Immediately
    if high_severity:
        primary = high_severity[0]
        return {
            'decision': 'human_escalation',
            'reason': primary['description'],
            'risk_level': 'high',
            'signals_triggered': signal_names,
            'metadata': {
                'intent': intent,
                'top_retrieval_score': top_score,
            }
        }

    # Escalation Rule 2: Any Medium Severity Signal -> Escalate
    if medium_severity:
        primary = medium_severity[0]
        return {
            'decision': 'human_escalation',
            'reason': primary['description'],
            'risk_level': 'medium',
            'signals_triggered': signal_names,
            'metadata': {
                'intent': intent,
                'top_retrieval_score': top_score,
            }
        }

    # Auto-handle Rule: No Risk Signals, Clear Intent, Sufficient Historical Grounding
    score_str = f"relevance: {top_score:.3f}" if top_score is not None else "grounded"
    intent_str = f"'{intent}'" if intent else "diagnosed"

    return {
        'decision': 'auto_handle',
        'reason': (
            f"Routine troubleshooting query with clear {intent_str} intent and sufficient "
            f"historical evidence grounding ({score_str}); safe for automated response."
        ),
        'risk_level': 'low',
        'signals_triggered': [],
        'metadata': {
            'intent': intent,
            'top_retrieval_score': top_score,
        }
    }


def format_escalation_report(results: List[Dict[str, Any]], output_path: str = 'apple_support_escalation_demo.txt') -> str:
    """Generate a clean text report summarizing escalation decisions."""
    lines = []
    lines.append("=" * 80)
    lines.append("AppleSupport AI Agent — Escalation Decision Demonstration")
    lines.append("=" * 80)
    lines.append(f"Total Examples Evaluated: {len(results)}")
    
    auto_count = sum(1 for r in results if r['decision'] == 'auto_handle')
    esc_count = sum(1 for r in results if r['decision'] == 'human_escalation')
    lines.append(f"Auto-handled: {auto_count} ({auto_count / max(1, len(results)) * 100:.1f}%)")
    lines.append(f"Escalated to Human: {esc_count} ({esc_count / max(1, len(results)) * 100:.1f}%)")
    lines.append("-" * 80)
    lines.append("")

    for idx, r in enumerate(results, start=1):
        score_val = r.get('top_score')
        score_disp = f"{score_val:.4f}" if score_val is not None else "N/A"
        lines.append(f"[{idx}] Query: {r['query']}")
        lines.append(f"    Intent: {r.get('intent', 'N/A')} | Retrieval Score: {score_disp}")
        lines.append(f"    Decision: {r['decision'].upper()} (Risk: {r['risk_level'].upper()})")
        lines.append(f"    Reason: {r['reason']}")
        if r['signals_triggered']:
            lines.append(f"    Signals: {', '.join(r['signals_triggered'])}")
        lines.append("")

    lines.append("=" * 80)
    content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(content)

    return output_path
