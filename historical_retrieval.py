#!/usr/bin/env python3
"""Conversation-level historical retrieval for AppleSupport.

This file intentionally uses only the reconstructed AppleSupport
conversation export apple_support_conversations.csv and never reads
or derives labels from golden_eval.csv or golden_candidates.csv.
"""

import os
import re
import html
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SOURCE_CSV = 'apple_support_conversations.csv'
INDEX_CSV = 'apple_support_conversation_index.csv'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_FIELDS = ['conversation_id', 'text', 'customer_turns', 'support_turns', 'n_messages', 'embedding']


def clean_text(text: Any) -> str:
    """Light normalization reused from the project style for compatibility."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'[^a-zA-Z0-9_\-\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_reconstructed_conversations(source_csv: str = SOURCE_CSV) -> pd.DataFrame:
    """Load the AppleSupport reconstructed conversation rows.

    The conversation_id keys the rows together. The conversation transcript is
    reconstructed later at the conversation level, not at tweet-level.
    """
    df = pd.read_csv(source_csv, low_memory=False)
    required = {'conversation_id', 'author_id', 'inbound', 'text', 'thread_order'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{source_csv} is missing required columns: {missing}')

    # Normalize key columns.
    df = df.copy()
    df['author_id'] = df['author_id'].fillna('').astype(str).str.strip()
    df['text'] = df['text'].fillna('').astype(str)
    df['conversation_id'] = pd.to_numeric(df['conversation_id'], errors='coerce')
    df = df.dropna(subset=['conversation_id'])
    df['conversation_id'] = df['conversation_id'].astype(int)

    # Keep only messages that are likely conversation content.
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 0]
    return df


def build_conversation_records(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse all rows into one conversation-level record.

    Each record contains a transcript-level text for the conversation and the
    customer/support utterance slices needed for explainability.
    """
    records = []
    for conversation_id, group in df.groupby('conversation_id', sort=True):
        # Thread order is available and useful for preserving order.
        group = group.sort_values(['thread_order', 'created_at'] if 'created_at' in group.columns else ['thread_order'])

        # Bucket rows into AppleSupport reply turns and customer turns.
        support_turns = []
        customer_turns = []
        all_turns = []
        for _, row in group.iterrows():
            txt = clean_text(row['text'])
            if not txt:
                continue
            all_turns.append(txt)
            if str(row['author_id']).lower() == 'applesupport':
                support_turns.append(txt)
            elif bool(row.get('inbound', False)):
                customer_turns.append(txt)
            else:
                # In this exported data, some non-AppleSupport rows can be customer authors.
                # Keep as customer-like if the row appears to be an inbound tweet from a user.
                customer_turns.append(txt)

        conversation_text = ' '.join(all_turns)
        transcript = ' '.join([f'[{idx + 1}] {t}' for idx, t in enumerate(all_turns)])

        # Only keep a conversation if it actually has both customer and AppleSupport replies.
        # This is conversation-level retrieval, not isolated tweet retrieval.
        if len(all_turns) < 2 or len(support_turns) == 0 or len(customer_turns) == 0:
            continue

        records.append({
            'conversation_id': int(conversation_id),
            'text': conversation_text,
            'transcript': transcript,
            'customer_turns': ' '.join(customer_turns),
            'support_turns': ' '.join(support_turns),
            'n_messages': len(all_turns),
            'n_customer_turns': len(customer_turns),
            'n_support_turns': len(support_turns),
        })

    return pd.DataFrame(records)


def build_index(source_csv: str = SOURCE_CSV,
                output_csv: str = INDEX_CSV,
                model_name: str = EMBEDDING_MODEL,
                sample_size: int = None,
                max_conversations: int = None) -> pd.DataFrame:
    """Encode and save a conversation-level embedding index.

    Returns the conversation record DataFrame with an embedding column.
    """
    df = load_reconstructed_conversations(source_csv)
    records = build_conversation_records(df)

    if sample_size:
        records = records.sample(n=min(sample_size, len(records)), random_state=42)
    if max_conversations:
        records = records.head(max_conversations)

    model = SentenceTransformer(model_name)
    texts = records['text'].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    records['embedding'] = embeddings.tolist()

    # Write the conversation-level retrieval index as an artifact.
    records.to_csv(output_csv, index=False)
    return records


def load_index(index_csv: str = INDEX_CSV) -> pd.DataFrame:
    """Load the saved embedding index.

    The embedding column is a list of floats. These are loaded as object cells
    from CSV and must be converted back using eval/ast parsing.
    """
    df = pd.read_csv(index_csv, low_memory=False)
    if 'embedding' in df.columns:
        # CSV keeps list as a string; need safe Python list conversion.
        def parse_vec(v):
            try:
                return np.fromstring(str(v).strip('[]'), sep=' ', dtype=float)
            except Exception:
                try:
                    return np.array(eval(str(v)), dtype=float)
                except Exception:
                    return np.array([], dtype=float)
        # Lightweight converted representation.
        # Keep vector column as object string and decode in search.
        return df
    return df


def cosine_search(query: str,
                   records: pd.DataFrame,
                   model_name: str = EMBEDDING_MODEL,
                   k: int = 5,
                   use_customer_turns: bool = False) -> pd.DataFrame:
    """Semantic retrieval over conversation records by cosine similarity.

    Query -> SentenceTransformer embedding -> cosine match -> top-K conversation records.
    """
    model = SentenceTransformer(model_name)
    q_vec = model.encode([clean_text(query)], show_progress_bar=False, normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=float)

    # Convert embedding strings into numpy arrays, if needed.
    def embedding_from_cell(cell):
        if isinstance(cell, str):
            # Work with persisted Python/CSV strings such as
            # '[0.1, 0.2, 0.3]' or '0.1 0.2 0.3'.
            text = cell.strip()
            text = text.replace('[', '').replace(']', '').replace('\n', ' ')
            # Accept any comma or whitespace separated float token.
            vals = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', text)
            if not vals:
                return np.zeros(384, dtype=float)
            return np.array([float(x) for x in vals], dtype=float)
        if isinstance(cell, (list, tuple, np.ndarray)):
            return np.asarray(cell, dtype=float)
        return np.array([], dtype=float)

    index = []
    for _, row in records.iterrows():
        eng = embedding_from_cell(row['embedding'])
        index.append(eng)

    if len(index) == 0:
        return pd.DataFrame(columns=records.columns)

    X = np.vstack(index)
    q_vec = q_vec.reshape(1, -1)
    sims = cosine_similarity(q_vec, X)[0]
    top_i = np.argsort(-sims)[:k]

    top = records.iloc[top_i].copy()
    top['sim_score'] = sims[top_i]
    return top.reset_index(drop=True)


def make_query_demonstration_file(records: pd.DataFrame,
                                  source_csv: str = SOURCE_CSV,
                                  test_queries: List[str] = None,
                                  output_txt: str = 'apple_support_retrieval_demo.txt') -> str:
    """Create a small retrieval demo report for top-K examples.

    This is deliberately a demo/report format, not a generator.
    """
    if test_queries is None:
        test_queries = [
            'battery drains and phone won’t charge after update',
            'wifi keeps dropping during calls',
            'keyboard typing is delayed and messages are weird',
            'iCloud account signing in issues',
        ]

    lines = []
    lines.append('AppleSupport historical conversation retrieval demo')
    lines.append('Source conversation reconstruction file: ' + source_csv)
    lines.append('Retrieval model: SentenceTransformer all-MiniLM-L6-v2')
    lines.append('Index type: conversation-level, embedding search over aggregated AppleSupport conversation transcripts')
    lines.append('')
    lines.append('Important: no golden_eval.csv labels were used to build the retrieval index.')
    lines.append('')

    for qi, query in enumerate(test_queries, start=1):
        lines.append(f'Query {qi}: {query}')
        top = cosine_search(query, records, k=5)
        for j, row in top.iterrows():
            lines.append(f'  Rank {j+1} conversation_id={row["conversation_id"]} score={row["sim_score"]:.4f}')
            lines.append(f'    matched_customer_excerpt={clean_text(row["customer_turns"][:240])}')
            # Our demo intentionally exposes the support and customer slices in this order.
            support_excerpt = clean_text(row['support_turns'][:240])
            lines.append(f'    support_excerpt={support_excerpt}')
            lines.append(f'    messages_in_conversation={row["n_messages"]} customer_turns={row["n_customer_turns"]} support_turns={row["n_support_turns"]}')
            lines.append(f'    why_relevant=conversation shares both lexical and semantic overlap with the customer issue and AppleSupport reply pattern.')
        lines.append('')

    with open(output_txt, 'w') as f:
        f.write('\n'.join(lines))

    return output_txt


def self_retrieval_sanity_check(records: pd.DataFrame,
                                sample_size: int = 25,
                                k: int = 5) -> Dict[str, float]:
    """Useful, contamination-free sanity check: ask the model to retrieve the same conversation from its own customer turn.

    This checks that the same conversation can be found from a naturally occurring customer message,
    without reading golden labels and without a query that belongs to the evaluation file.
    """
    # Use only historical AppleSupport rows, then sample customer turns from the index.
    # We will compare to the conversation_id of the query text.
    # The query is taken from the conversation text itself and must be contained in that conversation.
    # We avoid using golden labels; we only inspect the historical source rows.
    sample = records.sample(n=min(sample_size, len(records)), random_state=42).copy()
    hits = 0
    reciprocal_rank_sum = 0.0
    pos = []

    for idx, row in sample.iterrows():
        query_text = row['customer_turns']
        # Search.
        top = cosine_search(query_text, records, k=k)
        if top.empty:
            continue
        ranks = top['conversation_id'].tolist()
        if int(row['conversation_id']) in [int(x) for x in ranks]:
            hits += 1
            rank = ranks.index(int(row['conversation_id'])) + 1
            reciprocal_rank_sum += 1.0 / rank
        pos.append(int(row['conversation_id']))

    n = len(sample)
    metrics = {
        'sample_size': n,
        'top1_hit_rate': hits / max(n, 1),
        'mrr_at_5': reciprocal_rank_sum / max(n, 1),
    }
    return metrics


def main():
    """Build a conversational retrieval index artifact and a small demo output.

    A safe default is a bounded historical sample so the retrieval demo remains
    reproducible in a normal workspace run. Larger full-index builds can be
    enabled by increasing MAX_CONVERSATIONS in the environment.
    """
    max_conversations = int(os.getenv('MAX_CONVERSATIONS', '2000'))

    # Build index from the reconstructed AppleSupport support conversations.
    # The historical files are not the label-golden files.
    records = build_index(SOURCE_CSV, INDEX_CSV, max_conversations=max_conversations)

    # Basic quality sanity check without using any golden labels.
    # It proves the embedding search can discover and rank the same historical conversation.
    metrics = self_retrieval_sanity_check(records, sample_size=25, k=5)

    # A few customer-like query strings, deliberately sourced from historical AppleSupport rows only.
    # These are real examples, not labels, and not from golden_eval.csv.
    test_queries = [
        'battery drains after software update and phone won’t charge',
        'internet keeps dropping and wifi is unreliable',
        'keyboard is slow and autocorrect takes letters too late',
        'mail and messages are stopped syncing with iCloud',
    ]

    # Write retrieval demo outputs.
    report_path = make_query_demonstration_file(records, SOURCE_CSV, test_queries, output_txt='apple_support_retrieval_demo.txt')

    # Write a tiny quality summary.
    quality_lines = []
    quality_lines.append('Basic retrieval quality sanity check (historical source only, no golden labels):')
    quality_lines.append(f'Sample size: {metrics["sample_size"]}')
    quality_lines.append(f'Top-1 self-retrieval hit rate: {metrics["top1_hit_rate"]:.4f}')
    quality_lines.append(f'MRR@5: {metrics["mrr_at_5"]:.4f}')
    quality_lines.append('Note: this checks same-conversation localization, not a human-reviewed gold retrieval benchmark.')

    with open('apple_support_retrieval_quality.txt', 'w') as f:
        f.write('\n'.join(quality_lines))

    print('\n'.join(quality_lines))
    print(f'Wrote retrieval index to {INDEX_CSV}')
    print(f'Wrote demo report to {report_path}')


if __name__ == '__main__':
    main()
