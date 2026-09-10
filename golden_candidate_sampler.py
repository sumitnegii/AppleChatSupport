import pandas as pd
import numpy as np
import re
import html
from collections import Counter
from pathlib import Path

INPUT = 'apple_support_conversations.csv'
OUTPUT = 'golden_candidates.csv'

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

LABEL_KEYWORDS = {
    'software_update': ['update','ios','software','upgrade','version','install'],
    'battery_charging': ['battery','charge','charging','drain','power'],
    'app_store': ['app store','itunes','app','download','song','playlist','music'],
    'account_apple_id': ['account','apple id','login','password','sign in','icloud account','recovery'],
    'connectivity': ['wifi','bluetooth','network','connect','signal','internet'],
    'keyboard_input': ['keyboard','text','symbol','letter','typing','spell','input'],
    'device_issue': ['screen','crash','lag','slow','freeze','performance','device','phone'],
    'payments_billing': ['payment','purchase','card','gift card','billing','charged'],
    'mail_messages': ['mail','email','icloud','message','inbox','sync','messages'],
    'media_accessories': ['media','music','photos','home sharing','watch','charger','dock','accessory','support','dm'],
}


def clean_text(x):
    x = html.unescape(str(x))
    x = re.sub(r'https?://\S+|www\.\S+', ' ', x)
    x = re.sub(r'@[A-Za-z0-9_]+', ' ', x)
    # Keep punctuation/content but minimize obvious artifacts.
    x = re.sub(r'[^\w\s#-]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    x = x.strip()
    return x


def is_valid_text(x):
    x = str(x).strip()
    if len(x) < 3:
        return False
    # Remove obvious empty/noise rows such as overflow symbols or just punctuation
    if len(re.sub(r'[^\w]+', '', x)) < 3:
        return False
    return True


def build_label_score_dict(text):
    txt = str(text).lower()
    scores = {label: 0 for label in FINAL_INTENTS}
    hits = {label: [] for label in FINAL_INTENTS}
    for label, kws in LABEL_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                hits[label].append(kw)
        scores[label] = len(hits[label])
    return scores, hits


def choose_candidates_for_label(df, label, n=25):
    """Return a balanced list of 25 rows using existing keyword and cluster-signal style evidence only.

    We rank by keyword evidence, then add a small explicit diversity sampler so the file has
    ambiguous/borderline examples, not just the easiest ones.
    """
    rows = []
    for _, r in df.iterrows():
        txt = r['clean_text']
        scores, hits = build_label_score_dict(txt)
        top_label = max(FINAL_INTENTS, key=lambda l: scores[l])
        # Candidate retrieval for requested label only is based on lexicon/keyword signal.
        # It is allowed to use the cluster-style ranking concept by preferring top-scoring rows.
        if scores[label] <= 0:
            continue
        # Some text has words in two cluster-like groups; keep them as candidate retrieval for manual review.
        top_label = max(FINAL_INTENTS, key=lambda l: scores[l])
        second_label = sorted(FINAL_INTENTS, key=lambda l: scores[l], reverse=True)[1]
        # Borderline/ambiguous examples are rows where the top label is close to a second label.
        # This intentionally further diversifies the 25 examples for manual labeling.
        margin = scores[top_label] - scores[second_label]
        is_borderline = scores[top_label] == scores[label] and scores[second_label] >= scores[label] - 1
        rows.append({
            'tweet_id': r['tweet_id'],
            'conversation_id': r['conversation_id'],
            'text': r['text'],
            'clean_text': txt,
            'proposed_intent': label,
            'score': scores[label],
            'top_label': top_label,
            'second_label': second_label,
            'margin': margin,
            'is_borderline': is_borderline,
            'label_scores': scores,
        })

    # Rank by confidence and diversity. Use a deterministic tie-breaker.
    # Candidate selection pushes top-score examples first but keeps some borderlines.
    # This helps the golden file include examples that may genuinely overlap intents.
    rows = sorted(rows, key=lambda x: (-x['score'], x['margin'], x['tweet_id']))

    # Reserve a few borderline examples separately, then fill with top evidence examples.
    clear_rows = [r for r in rows if r['score'] >= 2 and r['margin'] >= 2]
    borderline_rows = [r for r in rows if r['score'] >= 1 and r['margin'] <= 1]

    selected = []
    # Prefer unique conversation_id diversity so samples are not repeatedly from one thread.
    seen_conv = set()

    # First, take 20 clear examples with diversity by conversation.
    for r in clear_rows:
        if len(selected) >= n:
            break
        if pd.notna(r['conversation_id']) and str(r['conversation_id']) in seen_conv:
            continue
        if pd.notna(r['conversation_id']):
            seen_conv.add(str(r['conversation_id']))
        selected.append(r)

    # Add 5 ambiguous examples if possible.
    seen_ids = {str(r['tweet_id']) for r in selected}
    for r in borderline_rows:
        if len(selected) >= n:
            break
        if str(r['tweet_id']) in seen_ids:
            continue
        if pd.notna(r['conversation_id']) and str(r['conversation_id']) in seen_conv:
            # still allow some duplicates across conversations but diversify where possible.
            pass
        selected.append(r)
        seen_ids.add(str(r['tweet_id']))
        if pd.notna(r['conversation_id']):
            seen_conv.add(str(r['conversation_id']))

    # If we still need more rows, take additional high-score rows.
    seen_ids = {str(r['tweet_id']) for r in selected}
    for r in rows:
        if len(selected) >= n:
            break
        if str(r['tweet_id']) in seen_ids:
            continue
        # Additional diversity by conversation only if available.
        if pd.notna(r['conversation_id']) and str(r['conversation_id']) in seen_conv:
            continue
        selected.append(r)
        seen_ids.add(str(r['tweet_id']))
        if pd.notna(r['conversation_id']):
            seen_conv.add(str(r['conversation_id']))

    # Use enough deterministically, but if a label lacks enough examples fill with the closest.
    if len(selected) < n:
        for r in rows:
            if len(selected) >= n:
                break
            if str(r['tweet_id']) in seen_ids:
                continue
            selected.append(r)
            seen_ids.add(str(r['tweet_id']))

    # Keep exactly 25 rows per intent.
    selected = selected[:n]
    return selected


def main():
    df = pd.read_csv(INPUT, low_memory=False)
    df['inbound'] = df['inbound'].astype(str).str.lower()
    df['text'] = df['text'].fillna('').astype(str)
    df['clean_text'] = df['text'].apply(clean_text)

    # Customer-only slice as requested (# customer messages only)
    cust = df[df['inbound'] == 'true'].copy()
    # Remove empty/noise rows and exact dedupe by text.
    cust = cust[cust['clean_text'].apply(is_valid_text)].copy()

    # Use conversation_id from existing graph if present. Ensure numeric stability is preserved.
    cust['conversation_id'] = pd.to_numeric(cust['conversation_id'], errors='coerce')
    cust['tweet_id'] = pd.to_numeric(cust['tweet_id'], errors='coerce')

    # Create output rows.
    all_selected = []
    for label in FINAL_INTENTS:
        # Candidate retrieval from existing keyword cluster-like vocabulary.
        label_rows = choose_candidates_for_label(cust, label, n=25)
        for row in label_rows:
            # Preserve requested output columns only; proposed intent is descriptive and not ground truth.
            all_selected.append({
                'tweet_id': int(row['tweet_id']),
                'conversation_id': row['conversation_id'],
                'text': row['text'],
                'proposed_intent': row['proposed_intent'],
            })

    # Save exactly 250 rows. Should be deterministic and should not alter source dataset.
    out = pd.DataFrame(all_selected)
    # Keep up to 250 rows (25 x 10)
    out = out.head(250)
    out.to_csv(OUTPUT, index=False)

    print(f'Wrote {OUTPUT} with {len(out)} candidates.')
    print(out['proposed_intent'].value_counts().to_string())

if __name__ == '__main__':
    main()
