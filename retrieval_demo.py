#!/usr/bin/env python3
"""Small demo for showing top-K historical AppleSupport retrieval.

This script intentionally reads the conversation-level embedding index and
prints the neighbors for a short, human-curated query list.
"""

import pandas as pd

from historical_retrieval import cosine_search, load_index, clean_text

INDEX_CSV = 'apple_support_conversation_index.csv'


def main():
    records = pd.read_csv(INDEX_CSV, low_memory=False)
    queries = [
        'battery drains after software update and phone won’t charge',
        'internet keeps dropping and wifi is unreliable',
        'keyboard is slow and autocorrect takes letters too late',
        'mail and messages are stopped syncing with iCloud',
    ]

    print('AppleSupport conversation retrieval demo')
    print('Index file:', INDEX_CSV)
    print('')
    for i, q in enumerate(queries, start=1):
        print(f'Query {i}: {q}')
        top = cosine_search(q, records, k=3)
        if top.empty:
            print('  No matches found')
            continue
        for j, row in top.iterrows():
            print(f'  Rank {j+1} | conv={int(row["conversation_id"])} | score={row["sim_score"]:.4f}')
            print('    Customer turns:', clean_text(row['customer_turns'][:260]))
            print('    Support turns:', clean_text(row['support_turns'][:260]))
            print('    Message count:', int(row['n_messages']))
        print('')


if __name__ == '__main__':
    main()
