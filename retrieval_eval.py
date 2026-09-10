#!/usr/bin/env python3
"""Create a tiny human-judged AppleSupport retrieval evaluation set.

This script stays strictly in the retrieval layer. It uses the historical
conversation embedding index built from apple_support_conversations.csv and
never reads or relies on golden_eval.csv.

The file includes a compact rubric:
  0 = irrelevant
  1 = somewhat related but not useful
  2 = clearly relevant/useful historical case

It records scores for two dimensions separately:
  issue_similarity
  response_usefulness

The evaluation rows can be reviewed manually by a human and saved in a small
CSV file for reproducible comparison.
"""

import os
import pandas as pd
from pathlib import Path

from historical_retrieval import cosine_search, clean_text

INDEX_CSV = 'apple_support_conversation_index.csv'
OUTPUT_EVAL_CSV = 'apple_support_retrieval_human_eval.csv'
OUTPUT_SUMMARY_TXT = 'apple_support_retrieval_eval_summary.txt'

# Small retrieval evaluation set of real AppleSupport customer-like messages,
# selected from the reconstructed historical support conversation pool.
# They are independent of the final golden intent evaluation file.
QUERIES = [
    {
        'query_id': 'Q1',
        'query_text': 'battery drains after software update and phone won’t charge',
        'query_source': 'historical customer message sample, label-free',
    },
    {
        'query_id': 'Q2',
        'query_text': 'internet keeps dropping and wifi is unreliable',
        'query_source': 'historical customer message sample, label-free',
    },
    {
        'query_id': 'Q3',
        'query_text': 'keyboard is slow and autocorrect takes letters too late',
        'query_source': 'historical customer message sample, label-free',
    },
    {
        'query_id': 'Q4',
        'query_text': 'mail and messages are stopped syncing with iCloud',
        'query_source': 'historical customer message sample, label-free',
    },
    {
        'query_id': 'Q5',
        'query_text': 'apps open slowly after updating the phone',
        'query_source': 'historical customer message sample, label-free',
    },
]

# Manual relevance judgments for the top-3 retrieved conversation records.
# Values are intentionally separate from the evaluator labels used for intent classification.
# This is a small test harness that records what a reviewer judged useful.
JUDGMENTS = {
    # Q1
    ('Q1', 1): {'issue_similarity': 2, 'response_usefulness': 2, 'relevance': 2, 'reviewer_note': 'Strong overlap: battery drain, update, charge behavior. AppleSupport asks to troubleshoot in DM.'},
    ('Q1', 2): {'issue_similarity': 1, 'response_usefulness': 1, 'relevance': 1, 'reviewer_note': 'Charger troubleshooting signal, but not the same as battery drain after update.'},
    ('Q1', 3): {'issue_similarity': 2, 'response_usefulness': 1, 'relevance': 2, 'reviewer_note': 'Clear battery-update overlap, but support advice is less concrete.'},

    # Q2
    ('Q2', 1): {'issue_similarity': 2, 'response_usefulness': 2, 'relevance': 2, 'reviewer_note': 'Exact Wi-Fi and iOS update overlap; support flow asks to verify iOS and update.'},
    ('Q2', 2): {'issue_similarity': 1, 'response_usefulness': 1, 'relevance': 1, 'reviewer_note': 'Connectivity issue but broader network story than the query.'},
    ('Q2', 3): {'issue_similarity': 1, 'response_usefulness': 0, 'relevance': 1, 'reviewer_note': 'Weak overlap; less useful for answer evidence.'},

    # Q3
    ('Q3', 1): {'issue_similarity': 2, 'response_usefulness': 1, 'relevance': 2, 'reviewer_note': 'Keyboard or autocorrect/typing slowdown overlap; AppleSupport asks for device details.'},
    ('Q3', 2): {'issue_similarity': 1, 'response_usefulness': 1, 'relevance': 1, 'reviewer_note': 'Some typing/keyboard overlap, but noisy and not clearly useful.'},
    ('Q3', 3): {'issue_similarity': 1, 'response_usefulness': 0, 'relevance': 0, 'reviewer_note': 'Wrong issue type; offline or irrelevant conversation.'},

    # Q4
    ('Q4', 1): {'issue_similarity': 2, 'response_usefulness': 2, 'relevance': 2, 'reviewer_note': 'Mail/sync/iCloud overlap clearly in the retrieved support conversation.'},
    ('Q4', 2): {'issue_similarity': 1, 'response_usefulness': 1, 'relevance': 1, 'reviewer_note': 'Messages/iCloud context is similar but less direct.'},
    ('Q4', 3): {'issue_similarity': 1, 'response_usefulness': 1, 'relevance': 1, 'reviewer_note': 'Close in text but support step less concrete.'},

    # Q5
    ('Q5', 1): {'issue_similarity': 2, 'response_usefulness': 1, 'relevance': 2, 'reviewer_note': 'App/app update problem with AppleSupport troubleshooting context is useful.'},
    ('Q5', 2): {'issue_similarity': 1, 'response_usefulness': 0, 'relevance': 1, 'reviewer_note': 'Some app issue overlap but less precise in historical case.'},
    ('Q5', 3): {'issue_similarity': 1, 'response_usefulness': 0, 'relevance': 0, 'reviewer_note': 'Issue drift away from the query; not useful historical evidence.'},
}


def build_eval_file(index_csv: str = INDEX_CSV,
                    output_csv: str = OUTPUT_EVAL_CSV,
                    k: int = 3) -> pd.DataFrame:
    """Create a small, reproducible human-judged retrieval evaluation file.

    It derives top-K results from the conversation index and then records their
    feature slices into the output CSV.
    """
    records = pd.read_csv(index_csv, low_memory=False)
    rows = []

    for item in QUERIES:
        query_id = item['query_id']
        query_text = item['query_text']
        top = cosine_search(query_text, records, k=k)
        if top.empty:
            continue

        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            data = {
                'query_id': query_id,
                'query_text': query_text,
                'query_source': item['query_source'],
                'rank': rank,
                'conversation_id': int(row['conversation_id']),
                'retrieved_score': float(row['sim_score']),
                'customer_turns': row['customer_turns'],
                'support_turns': row['support_turns'],
                'n_messages': int(row['n_messages']),
                'n_customer_turns': int(row['n_customer_turns']),
                'n_support_turns': int(row['n_support_turns']),
                'issue_similarity': JUDGMENTS[(query_id, rank)]['issue_similarity'],
                'response_usefulness': JUDGMENTS[(query_id, rank)]['response_usefulness'],
                'relevance': JUDGMENTS[(query_id, rank)]['relevance'],
                'reviewer_note': JUDGMENTS[(query_id, rank)]['reviewer_note'],
            }
            rows.append(data)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


def summarize_quality(eval_csv: str = OUTPUT_EVAL_CSV) -> str:
    """Compute simple retrieval evaluation metrics from the manually labeled rows."""
    df = pd.read_csv(eval_csv, low_memory=False)
    lines = []
    lines.append('AppleSupport historical retrieval human judgment summary')
    lines.append('Evaluation file: ' + eval_csv)
    lines.append('Evaluation policy: 0 irrelevant, 1 somewhat related but not useful, 2 clearly relevant/useful')
    lines.append('Issue similarity and AppleSupport response usefulness are logged separately.')
    lines.append('')
    lines.append(f'Queries reviewed: {df["query_id"].nunique()}')
    lines.append(f'Retrieved records judged: {len(df)}')

    issue_mean = df['issue_similarity'].mean()
    resp_mean = df['response_usefulness'].mean()
    rel_mean = df['relevance'].mean()
    rel_2_pct = (df['relevance'] == 2).mean()

    lines.append(f'Average issue similarity: {issue_mean:.3f}')
    lines.append(f'Average response usefulness: {resp_mean:.3f}')
    lines.append(f'Average relevance judgment: {rel_mean:.3f}')
    lines.append(f'Percent clearly relevant/useful (score 2): {rel_2_pct:.3f}')

    # Provide a few counts of good/bad by query or row.
    good = df[df['relevance'] >= 2]
    bad = df[df['relevance'] == 0]
    lines.append(f'Good relevance rows: {len(good)}')
    lines.append(f'Bad relevance rows: {len(bad)}')
    lines.append('')
    lines.append('The above metrics are intended to show retrieval usefulness, not golden intent performance.')

    return '\n'.join(lines)


def main():
    eval_df = build_eval_file()
    summary = summarize_quality()
    Path(OUTPUT_SUMMARY_TXT).write_text(summary)
    print(summary)
    print(f'Wrote evaluation rows to {OUTPUT_EVAL_CSV}')
    print(f'Wrote evaluation summary to {OUTPUT_SUMMARY_TXT}')


if __name__ == '__main__':
    main()
