import pandas as pd
import numpy as np
from collections import defaultdict, deque
import re

INPUT = "archive/twcs/twcs.csv"
OUTPUT = "apple_support_conversations.csv"


def parse_response_ids(value):
    """Parse a response_tweet_id cell that may contain one id or a comma-separated list."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text in {"nan", "", "None", "NaN"}:
        return []

    # Strip quotes and bracket artifacts if present.
    text = text.replace('"', '').replace('[', '').replace(']', '')
    ids = []
    for token in text.split(','):
        token = token.strip()
        if token.isdigit():
            ids.append(int(token))
    return ids


def get_apple_thread_ids(df):
    """Return the tweet IDs in the AppleSupport thread graph.

    Start from every AppleSupport author row. Walk:
    - parent edge: in_response_to_tweet_id
    - child edges: response_tweet_id (same row)
    - reverse parent lookup: child_by_parent

    Then keep only rows whose text mentions @AppleSupport or whose author is AppleSupport.
    This removes false cross-brand edges.
    """
    df = df.copy()
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    df["author_id"] = df["author_id"].astype(str).str.strip()

    row_by_tweet = {}
    for idx, row in df.iterrows():
        tid = row["tweet_id"]
        if pd.notna(tid):
            row_by_tweet[int(tid)] = idx

    # child_by_parent tells us which tweet replies to a parent tweet.
    child_by_parent = defaultdict(list)
    for idx, row in df.iterrows():
        parent = row["in_response_to_tweet_id"]
        tid = row["tweet_id"]
        if pd.notna(parent) and pd.notna(tid):
            child_by_parent[int(parent)].append(int(tid))

    # Seed all AppleSupport rows.
    seeds = []
    for idx, row in df.iterrows():
        if row["author_id"].lower() == "applesupport":
            tid = row["tweet_id"]
            if pd.notna(tid):
                seeds.append(int(tid))

    queue = deque(sorted(set(seeds)))
    closure = set()

    while queue:
        cur = queue.popleft()
        if cur in closure:
            continue
        closure.add(cur)

        if cur not in row_by_tweet:
            continue

        row = df.loc[row_by_tweet[cur]]

        # follow child edges in response_tweet_id
        for child_id in parse_response_ids(row["response_tweet_id"]):
            if child_id in row_by_tweet and child_id not in closure:
                queue.append(child_id)

        # follow parent edge in_response_to_tweet_id
        parent_id = row["in_response_to_tweet_id"]
        if pd.notna(parent_id):
            parent_id = int(parent_id)
            if parent_id in row_by_tweet and parent_id not in closure:
                queue.append(parent_id)

        # follow all replies that point to this current tweet
        for child_id in child_by_parent.get(cur, []):
            if child_id in row_by_tweet and child_id not in closure:
                queue.append(child_id)

    # Keep only AppleSupport and text-mention rows inside this closure.
    keep_ids = set()
    for tid in closure:
        if tid not in row_by_tweet:
            continue
        row = df.loc[row_by_tweet[tid]]
        text_has_apple = bool(str(row["text"]).find("@AppleSupport") >= 0 or str(row["text"]).find("@applesupport") >= 0)
        if row["author_id"].lower() == "applesupport" or text_has_apple:
            keep_ids.add(tid)

    return sorted(keep_ids)


def main():
    df = pd.read_csv(INPUT, low_memory=False, dtype={"response_tweet_id": str})

    # Normalize fields.
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    df["author_id"] = df["author_id"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    # Get AppleSupport thread tweet ids.
    thread_tweet_ids = get_apple_thread_ids(df)
    apple_df = df[df["tweet_id"].isin(thread_tweet_ids)].copy()

    # Create a conversation_id from the root customer tweet: the first customer tweet
    # with no in_response_to_tweet_id in the chain.
    row_by_tweet = {}
    for idx, row in df.iterrows():
        tid = row["tweet_id"]
        if pd.notna(tid):
            row_by_tweet[int(tid)] = idx

    def find_root_customer(tid):
        """Walk upward by in_response_to_tweet_id until we reach a no-parent customer tweet."""
        visited = set()
        cur = int(tid)
        while cur in row_by_tweet:
            if cur in visited:
                break
            visited.add(cur)
            row = df.loc[row_by_tweet[cur]]
            parent = row["in_response_to_tweet_id"]
            if pd.isna(parent):
                return cur
            if int(parent) not in row_by_tweet:
                return cur
            cur = int(parent)
        return int(tid)

    # Assign conversation_id and ordered thread position.
    apple_df["conversation_id"] = apple_df["tweet_id"].apply(lambda x: find_root_customer(int(x)) if pd.notna(x) else np.nan)
    apple_df = apple_df.sort_values(["conversation_id", "tweet_id"], kind="mergesort")
    apple_df["thread_order"] = apple_df.groupby("conversation_id").cumcount() + 1

    # Save.
    apple_df.to_csv(OUTPUT, index=False)

    # Inspection metrics.
    print("Shape of exported AppleSupport conversations:", apple_df.shape)
    print("Customer messages (inbound=True):", int((apple_df["inbound"] == True).sum()))
    print("AppleSupport messages (author_id=AppleSupport):", int((apple_df["author_id"].str.lower() == "applesupport").sum()))
    print("Missing response_tweet_id values:", int(apple_df["response_tweet_id"].isna().sum()))
    print("Missing in_response_to_tweet_id values:", int(apple_df["in_response_to_tweet_id"].isna().sum()))
    print("Number of reconstructed conversations:", apple_df["conversation_id"].nunique())

if __name__ == "__main__":
    main()