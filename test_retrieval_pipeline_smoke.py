import os
import pandas as pd
from historical_retrieval import build_index, cosine_search


def test_build_index_and_cosine_retrieval_smoke(tmp_path):
    idx_path = str(tmp_path / 'small_index.csv')
    records = build_index(
        source_csv='apple_support_conversations.csv',
        output_csv=idx_path,
        max_conversations=50,
    )

    assert not records.empty
    assert 'embedding' in records.columns
    assert 'conversation_id' in records.columns
    assert 'customer_turns' in records.columns

    q = records['customer_turns'].iloc[0]
    top = cosine_search(q, records, k=3)
    assert not top.empty
    assert len(top) == 3

    assert os.path.exists(idx_path)
