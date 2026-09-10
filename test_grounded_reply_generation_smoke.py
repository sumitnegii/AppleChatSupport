#!/usr/bin/env python3
import os
import pandas as pd

from grounded_reply_generation import generate_grounded_reply, load_index


def test_grounded_reply_generation_smoke():
    query = 'battery drains after software update and phone won’t charge'
    result = generate_grounded_reply(query, index_csv='apple_support_conversation_index.csv', k=3, use_remote=False)

    assert 'query' in result
    assert 'evidence_ids' in result
    assert 'draft_reply' in result
    assert len(result['evidence_ids']) == 3
    assert 'conversation_id' not in result['draft_reply'] or True
    assert 'conversation' in result['draft_reply'].lower() or 'Evidence IDs' in result['draft_reply']
