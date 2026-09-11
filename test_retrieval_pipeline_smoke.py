import os
import json
import numpy as np
import pandas as pd
import pytest

from historical_retrieval import (
    build_index,
    build_faiss_index,
    load_faiss_index,
    faiss_search,
    cosine_search,
    INDEX_CSV,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
)
from grounded_reply_generation import generate_grounded_reply


def test_faiss_index_build_and_load(tmp_path):
    """Test 1 & 2: Prove FAISS index builds and loads correctly from disk."""
    records = pd.read_csv(INDEX_CSV, low_memory=False).head(20)
    out_index = str(tmp_path / "test.index")
    out_map = str(tmp_path / "test_mapping.json")

    idx, mapping = build_faiss_index(
        records=records,
        output_index_path=out_index,
        output_mapping_path=out_map,
    )

    assert os.path.exists(out_index)
    assert os.path.exists(out_map)
    assert idx.ntotal == 20
    assert mapping["dim"] == 384
    assert mapping["ntotal"] == 20
    assert len(mapping["vector_to_conversation_id"]) == 20

    loaded_idx, loaded_map = load_faiss_index(index_path=out_index, mapping_path=out_map)
    assert loaded_idx.ntotal == 20
    assert loaded_map["dim"] == 384
    assert loaded_map["vector_to_conversation_id"] == mapping["vector_to_conversation_id"]


def test_faiss_search_top_k():
    """Test 3: Prove FAISS search returns top-K results with valid scores."""
    records = pd.read_csv(INDEX_CSV, low_memory=False)
    query = "battery drains quickly after iOS update"

    top = faiss_search(query, records, k=3)
    assert not top.empty
    assert len(top) == 3
    assert "sim_score" in top.columns
    assert "conversation_id" in top.columns
    # Scores should be in valid cosine range [-1, 1]
    assert all(-1.0 <= s <= 1.0 for s in top["sim_score"])
    # Scores should be sorted descending
    scores = top["sim_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_faiss_conversation_id_mapping():
    """Test 4: Prove returned conversation IDs map correctly to metadata."""
    records = pd.read_csv(INDEX_CSV, low_memory=False)
    with open(FAISS_MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    query = "wifi disconnects repeatedly"
    top = faiss_search(query, records, k=5)

    valid_cids = set(mapping["vector_to_conversation_id"])
    for _, row in top.iterrows():
        cid = int(row["conversation_id"])
        assert cid in valid_cids
        # Verify text and turns exist in metadata
        assert len(str(row["customer_turns"])) > 0
        assert len(str(row["support_turns"])) > 0


def test_faiss_vs_cosine_exact_match():
    """Test comparison: Prove FAISS search returns exact same top-K as cosine search."""
    records = pd.read_csv(INDEX_CSV, low_memory=False)
    test_queries = [
        "battery drains after software update and phone won’t charge",
        "internet keeps dropping and wifi is unreliable",
        "keyboard typing is delayed and autocorrect fails",
        "iCloud mail and messages sync problem",
    ]

    for q in test_queries:
        res_faiss = faiss_search(q, records, k=3)
        res_cosine = cosine_search(q, records, k=3)

        faiss_cids = res_faiss["conversation_id"].tolist()
        cosine_cids = res_cosine["conversation_id"].tolist()
        assert faiss_cids == cosine_cids, f"Mismatch for query: {q}"

        faiss_scores = np.array(res_faiss["sim_score"].tolist())
        cosine_scores = np.array(res_cosine["sim_score"].tolist())
        np.testing.assert_allclose(faiss_scores, cosine_scores, rtol=1e-4, atol=1e-4)


def test_grounded_reply_receives_correct_evidence():
    """Test 5: Prove grounded reply generation receives correct evidence from FAISS."""
    records = pd.read_csv(INDEX_CSV, low_memory=False)
    query = "battery drains after software update and phone won’t charge"

    top_faiss = faiss_search(query, records, k=3)
    expected_ids = [str(int(x)) for x in top_faiss["conversation_id"].tolist()]

    result = generate_grounded_reply(query, index_csv=INDEX_CSV, k=3, use_remote=False)

    assert result["evidence_ids"] == expected_ids
    assert len(result["evidence_ids"]) == 3
    # Check that at least one evidence ID is mentioned in the drafted reply
    assert any(eid in result["draft_reply"] for eid in expected_ids)
