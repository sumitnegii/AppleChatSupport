# Decision Log

This log records the non-obvious engineering decisions and trade-offs made while building the AppleSupport AI Agent.

### 1. Dataset Selection: AppleSupport Tweets
- **Decision**: Used historical `@AppleSupport` conversations from the Kaggle Customer Support on Twitter dataset.
- **Why**: Real customer support tweets contain realistic noise (informal shorthand, frustration, typos, iOS version references) and verified agent replies, which synthetic support datasets lack.

### 2. 10 Data-Derived Intent Classes
- **Decision**: Grouped issues into 10 target intents (`software_update`, `battery_charging`, `app_store`, `account_apple_id`, `connectivity`, `keyboard_input`, `device_issue`, `payments_billing`, `mail_messages`, `media_accessories`) plus an explicit `ambiguous` review tag.
- **Why**: 40+ granular classes caused severe label sparsity on small sets, while 3 broad categories were not actionable for routing. 10 classes balanced operational specificity with enough support in a 250-row evaluation set.

### 3. Decoupling Escalation from Intent Classification
- **Decision**: Built escalation as an independent rule-based safety layer rather than treating "escalate" as an intent class.
- **Why**: Escalation depends on operational risk, not just topic. A `battery_charging` query could be a simple tip request (auto-handle) or a swollen, smoking battery (immediate safety escalation).

### 4. Conversation-Level Retrieval
- **Decision**: Indexed full conversations (customer turns + support turns grouped by `conversation_id`) instead of isolated tweets.
- **Why**: Standalone customer tweets often lack context ("it broke again"), and standalone agent tweets are often partial ("please DM us"). Full conversation transcripts preserve the verified troubleshooting path.

### 5. Embedding Model (`all-MiniLM-L6-v2`)
- **Decision**: Used `all-MiniLM-L6-v2` (384 dimensions) for both semantic classification and historical retrieval.
- **Why**: Runs locally on CPU with sub-10ms encoding latency, zero API rate limits, zero monetary cost, and consistent deterministic vector outputs.

### 6. Local FAISS Vector Index (`IndexFlatIP`)
- **Decision**: Stored conversation vectors in a local FAISS index using `IndexFlatIP` over L2-normalized embeddings.
- **Why**: Exact inner product on normalized vectors is mathematically identical to cosine similarity, runs in under 1ms locally, and avoids running external vector database servers.

### 7. 2,000-Conversation Subset for Retrieval Index
- **Decision**: Indexed a deterministic 2,000-conversation subset covering multi-turn cases across all 10 intents, rather than indexing all ~80k conversations.
- **Why**: Kept index construction and search fast enough to reproduce locally in under 15 minutes, with the index file under 3 MB and memory usage low.

### 8. Minimum Retrieval Similarity Threshold (0.28)
- **Decision**: Enforced a minimum cosine similarity threshold of 0.28 on the top retrieved case before allowing automated replies.
- **Why**: Out-of-domain queries (like asking to install Linux on Apple Watch) yield low retrieval scores. Escalating them prevents the model from hallucinating ungrounded steps.

### 9. Conservative Safety Escalation
- **Decision**: Automatically escalated payment disputes, account lockouts, hardware safety risks, explicit human requests, and low-retrieval queries.
- **Why**: Incorrect automated replies on credit card disputes or battery safety issues carry high real-world risk. Escalating these to human agents is safer than optimizing for a high auto-handle percentage.

### 10. Low-Confidence Safeguard for Intent Classifier (0.13)
- **Decision**: Added an escalation check when top intent probability is below 0.13.
- **Why**: The Logistic Regression probabilities are uncalibrated, but a score below 0.13 is near the 10% uniform random prior for 10 classes, signalling that the classifier is guessing.

### 11. 250-Row Human-Labelled Golden Set
- **Decision**: Built and reviewed a 250-example golden set (`golden_eval.csv`) with manual inspection and ambiguity flags.
- **Why**: Evaluating purely against LLM-generated pseudo-labels creates circular confirmation bias. Real human review was necessary to catch sarcasm, short queries, and ambiguous intents.

### 12. Aligned Rubrics for LLM-as-a-Judge
- **Decision**: Aligned the LLM judge prompt to match the exact 0–2 human scoring definitions, specifically for unsupported claims (0 = none, 1 = minor, 2 = significant).
- **Why**: When rubrics differ, agreement drops. Aligning the definitions gave 60% exact and 100% adjacent agreement on overall quality with human scores.

### 13. Resilient LLM Provider Fallback
- **Decision**: Used Groq as primary with Gemini, OpenAI, and deterministic template fallbacks.
- **Why**: Prevents evaluations or demos from breaking when third-party API keys hit quotas or are not provided in the environment.

### 14. Local Index over Hosted Vector Database
- **Decision**: Kept retrieval local (FAISS + CSV) rather than using cloud services like Pinecone or Weaviate.
- **Why**: Removes external infrastructure dependencies, API keys, and setup overhead, allowing anyone to clone and run the project locally.
