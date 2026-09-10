import pandas as pd
import numpy as np
import re
import html
from collections import Counter

INPUT = 'apple_support_conversations.csv'
REPORT = 'intent_labeling_review_report.txt'

# Proposed label vocabulary
LABEL_KEYWORDS = {
    'software_update': ['update', 'ios', 'software', 'upgrade', 'version', 'install'],
    'battery_charging': ['battery', 'charge', 'charging', 'drain', 'power'],
    'app_store_itunes': ['app store', 'itunes', 'app', 'download', 'song', 'playlist', 'music'],
    'account_apple_id': ['account', 'apple id', 'login', 'password', 'sign in', 'icloud account', 'recovery'],
    'connectivity': ['wifi', 'bluetooth', 'network', 'connect', 'signal', 'internet'],
    'keyboard_input': ['keyboard', 'text', 'symbol', 'letter', 'typing', 'spell', 'input'],
    'device_performance': ['screen', 'crash', 'lag', 'slow', 'freeze', 'performance', 'device', 'phone'],
    'payments_billing': ['payment', 'purchase', 'card', 'gift card', 'billing', 'charged'],
    'mail_messages_icloud': ['mail', 'email', 'icloud', 'message', 'inbox', 'sync', 'messages'],
    'media_accessories_support': ['media', 'music', 'photos', 'home sharing', 'watch', 'charger', 'dock', 'accessory', 'support', 'dm'],
}

LABEL_ORDER = list(LABEL_KEYWORDS.keys())


def clean_text(text):
    """Minimal customer-message cleaning used for labeling only."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'[^\w\s#\-\']+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_valid(text):
    text = text.strip()
    if len(text) < 3:
        return False
    if len(re.sub(r'[^\w]+', '', text)) < 3:
        return False
    return True


def assign_label(text):
    """Rule-based lexicon classifier for the requested 10-label AppleSupport taxonomy."""
    txt = str(text).lower()
    scores = []
    for label, kws in LABEL_KEYWORDS.items():
        hits = 0
        for kw in kws:
            if kw in txt:
                hits += 1
        if hits > 0:
            scores.append((label, hits))

    if not scores:
        # fallback to a safe generic support label
        return 'media_accessories_support'

    # Highest hit count wins; tie keeps label-order stable.
    scores.sort(key=lambda x: (-x[1], LABEL_ORDER.index(x[0])))
    return scores[0][0]


def main():
    df = pd.read_csv(INPUT, low_memory=False)
    df['inbound'] = df['inbound'].astype(str).str.lower()
    df['author_id'] = df['author_id'].astype(str).str.strip()
    df['text'] = df['text'].fillna('').astype(str)

    # Customer-only slice
    customer = df[df['inbound'] == 'true'].copy()
    customer['clean_text'] = customer['text'].apply(clean_text)
    customer = customer[customer['clean_text'].apply(is_valid)].copy()

    # Proposed label scoring assignment.
    customer['assigned_label'] = customer['clean_text'].apply(assign_label)

    # Proposed distribution.
    label_counts = customer['assigned_label'].value_counts().sort_index()

    # Build report.
    lines = []
    lines.append('AppleSupport Customer Intent Labeling Report')
    lines.append('Source: apple_support_conversations.csv')
    lines.append('Rows used: %d' % len(customer))
    lines.append('Cleaning: minimal URL/mention removal, whitespace collapse, punctuation normalization')
    lines.append('Labels used: ' + ', '.join(LABEL_ORDER))
    lines.append('')
    lines.append('Proposed label distribution:')
    for label in LABEL_ORDER:
        c = int(label_counts.get(label, 0))
        lines.append(f'{label}: {c}')
    lines.append('')
    lines.append('20 examples per intent for manual review:')

    # Write 20 examples per label
    for label in LABEL_ORDER:
        lines.append('')
        lines.append('### ' + label)
        sample = customer[customer['assigned_label'] == label][
            ['tweet_id', 'author_id', 'clean_text']
        ].drop_duplicates().head(20)

        for _, row in sample.iterrows():
            lines.append(f"tweet_id={int(row['tweet_id'])} | author_id={row['author_id']} | text={row['clean_text']}")

    # Also add cluster/theme grouping evidence in the report.
    lines.append('')
    lines.append('Theme-backed rule groups (source for label scores):')
    for label, kws in LABEL_KEYWORDS.items():
        lines.append(f'{label}: {kws}')

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print('Wrote report:', REPORT)
    print('Rows used:', len(customer))
    print('Label distribution:')
    print(label_counts.to_string())

if __name__ == '__main__':
    main()
